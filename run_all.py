#!/usr/bin/env python3
"""
Run every experiment of the paper except the batch-size study of
Section 7, which has its own script (exp24_batch_study.py).

    python run_all.py --resume            # full run, several jobs in parallel
    python run_all.py --resume --jobs 1   # one experiment at a time
    python run_all.py --quick             # short test run of every script
    python run_all.py --only exp19,exp20  # selected experiments
    python run_all.py --list              # show the plan and exit

--jobs N runs N experiments at the same time (default: half the CPU cores,
at most 6).  Every process is single-threaded and every run is seeded, so
the results do not depend on --jobs.  The output of each experiment goes to
results/logs/<id>_<mode>.txt, figures to results/figures and per-seed data
to results/data.  The timing experiments (exp5, exp19) run alone at the end.
"""

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
LOGS = os.path.join(RES, "logs")

# (id, script, supports --quick, estimated minutes on one core, description,
#  extra args, extra environment, required modules, run alone)
_LC = {"adult": "126025", "higgs": "167200", "volkert": "168331", "fmnist": "189908"}
PLAN = [
    ("exp7",   "exp7_retrained_and_ucb.py",     False,   2, "Refitted NN/RF; SDD with UCB", [], {}, [], False),
    ("exp13",  "exp13_naive_topq.py",           True,   25, "Naive top-q vs CL-min", [], {}, [], False),
    ("exp15",  "exp15_large_q_sdd.py",          True,   25, "Levy-10D SDD, q = 10/20/50", [], {}, [], False),
    ("exp16",  "exp16_lengthscale.py",          True,    2, "Suppression radius vs lengthscale", [], {}, [], False),
    ("exp17",  "exp17_lastlayer_bnn.py",        True,    3, "Last-layer BNN, uncertainty profile", [], {}, [], False),
    ("exp17b", "exp17b_bnn_perseed.py",         False,   5, "Last-layer BNN, 20 seeds", [], {}, [], False),
    ("exp18",  "exp18_refit_lengthscale.py",    True,    2, "Hyperparameters refitted per step", [], {}, [], False),
    ("exp20",  "exp20_sdd_start_sensitivity.py", False, 40, "SDD vs starts and GP configuration", [], {}, [], False),
    ("exp21",  "exp21_acquisition_monotonicity.py", False, 10, "Duplicates by acquisition/lie/incumbent", [], {}, [], False),
    ("exp22",  "exp22_noisy_hartmann.py",       False,  45, "Noisy Hartmann-6D", [], {}, [], False),
] + [
    (f"exp23A_{k}", "exp23_lcbench_hpo.py", False, 30, f"LCBench SDD ({k})", [],
     {"EXP_LCBENCH_TASKS": t}, ["onnxruntime"], False)
    for k, t in _LC.items()
] + [
    # timing experiments run alone at the end, so other jobs do not slow them
    ("exp5",   "exp5_wallclock.py",             False,   1, "Time per batch step", [], {}, [], True),
    ("exp19",  "exp19_mqrbf_power_function.py", False,   8, "MQ-RBF heuristic vs power function", [], {}, [], True),
]


def env_for_children(quick):
    env = dict(os.environ)
    env.update({
        "PYTHONUTF8": "1",              # Windows: UTF-8 for files and pipes
        "PYTHONIOENCODING": "utf-8",
        "MPLBACKEND": "Agg",            # never open plot windows
        "OMP_NUM_THREADS": "1",         # single-threaded BLAS
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "EXP_QUICK": "1" if quick else "0",
    })
    return env


def record_environment(path):
    info = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    pins = {"numpy": "2.5.1", "scipy": "1.18.0", "sklearn": "1.9.0",
            "matplotlib": "3.11.0", "torch": "2.11.0", "botorch": "0.18.1",
            "gpytorch": "1.15.2", "onnxruntime": "1.30.0"}
    versions = {}
    for mod in pins:
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "?")
        except Exception as e:                      # noqa: BLE001
            versions[mod] = f"NOT IMPORTABLE ({type(e).__name__})"
    info["versions"] = versions
    info["pinned"] = pins
    info["version_mismatches"] = {k: v for k, v in versions.items()
                                  if not str(v).startswith(pins[k])}
    try:
        import numpy as np
        info["numpy_blas"] = np.show_config(mode="dicts").get(
            "Build Dependencies", {}).get("blas", {})
    except Exception:                               # noqa: BLE001
        pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test of every script")
    ap.add_argument("--only", default="", help="comma list, e.g. exp19,exp20")
    ap.add_argument("--skip", default="", help="comma list to skip")
    ap.add_argument("--resume", action="store_true",
                    help="skip experiments that already finished successfully")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--jobs", default="auto",
                    help="experiments to run at the same time (default: auto)")
    args = ap.parse_args()
    cores = os.cpu_count() or 2
    if args.jobs == "auto":
        jobs = min(6, max(1, cores // 2))
    else:
        jobs = max(1, int(args.jobs))

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    plan = [p for p in PLAN if (not only or p[0] in only) and p[0] not in skip]

    if args.list:
        for pid, script, _, est, what, _a, _e, _r, solo in plan:
            print(f"  {pid:<14} ~{est:>4} min   {script:<34} {what}"
                  + ("   [alone]" if solo else ""))
        print(f"\n  total ~{sum(p[3] for p in plan) / 60:.1f} h on one core")
        return

    os.makedirs(LOGS, exist_ok=True)
    mode = "QUICK" if args.quick else "FULL"
    info = record_environment(os.path.join(RES, "environment.json"))
    print("=" * 78)
    print(f"Experiment run ({mode})  -  {info['timestamp']}")
    print(f"Python {sys.version.split()[0]} on {info['platform']}")
    if info["version_mismatches"]:
        print("Note: package versions differ from requirements.txt:")
        for k, v in info["version_mismatches"].items():
            print(f"   {k}: have {v}, pinned {info['pinned'][k]}")
    print("=" * 78)

    env = env_for_children(args.quick)
    status_path = os.path.join(RES, f"status_{mode.lower()}.json")
    status = {}
    if os.path.exists(status_path):
        try:
            with open(status_path, encoding="utf-8") as f:
                status = json.load(f)
        except (ValueError, OSError):
            # e.g. emptied by a Ctrl+C while it was being written
            print(f"NOTE: {os.path.relpath(status_path, HERE)} was unreadable; "
                  "starting a fresh status file (use --only / --skip to avoid "
                  "rerunning finished experiments).")
            status = {}

    todo = [p for p in plan
            if not (args.resume and status.get(p[0], {}).get("ok"))]
    for p in plan:
        if p not in todo:
            print(f"  {p[0]}: already done, skipping")

    # required packages must import in this Python
    missing = set()
    for p in todo:
        for mod in p[7]:
            try:
                __import__(mod)
            except Exception:                                 # noqa: BLE001
                missing.add(mod)
    if missing:
        pins = {"onnxruntime": "onnxruntime==1.30.0", "botorch": "botorch==0.18.1"}
        pkgs = " ".join(pins.get(m, m) for m in sorted(missing))
        print(f"\nMISSING PACKAGES for this Python ({sys.executable}): "
              f"{', '.join(sorted(missing))}")
        print("Install them into exactly this Python, then run again:")
        print(f'   "{sys.executable}" -m pip install {pkgs}')
        sys.exit(1)
    if any(p[0].startswith("exp23") for p in todo):
        import yahpo_data
        if not yahpo_data.ensure(["lcbench"]):
            sys.exit("The LCBench files could not be downloaded; see above.")

    def start(p):
        pid, script, has_quick, est, what, xargs, xenv, _req, _solo = p
        cmd = [sys.executable, os.path.join(HERE, script)] + list(xargs)
        if args.quick and has_quick:
            cmd.append("--quick")
        env_i = dict(env); env_i.update(xenv)
        log_path = os.path.join(LOGS, f"{pid}_{mode.lower()}.txt")
        log = open(log_path, "w", encoding="utf-8")
        log.write(f"# {' '.join(cmd)}\n# started {dt.datetime.now()}\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=HERE, env=env_i, stdout=log,
                                stderr=subprocess.STDOUT)
        return dict(p=p, proc=proc, log=log, t0=time.time(), path=log_path)

    def finish(r):
        rc = r["proc"].wait()
        r["log"].close()
        secs = time.time() - r["t0"]
        pid = r["p"][0]
        status[pid] = {"ok": rc == 0, "returncode": rc, "seconds": round(secs, 1),
                       "script": r["p"][1]}
        tmp = status_path + ".tmp"                 # atomic write
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        os.replace(tmp, status_path)
        tag = "OK    " if rc == 0 else "FAILED"
        print(f"  {dt.datetime.now():%H:%M}  {tag} {pid:<14} {secs / 60:6.1f} min"
              f"   (log: {os.path.relpath(r['path'], HERE)})")
        if rc != 0:
            with open(r["path"], encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-6:]
            for line in tail:
                print("           | " + line.rstrip()[:100])

    # longest first; the timing experiments wait until nothing else runs
    normal = [p for p in todo if not p[8]]
    solo_q = [p for p in todo if p[8]]
    queue = sorted(normal, key=lambda p: -p[3]) if jobs > 1 else list(normal)
    est_total = sum(p[3] for p in todo)
    print(f"\n{len(todo)} experiments, ~{est_total / 60:.1f} h of work on one core;"
          f" {cores} CPU cores: running {jobs} experiments at a time (--jobs {jobs}).")
    print("Output of each experiment goes to its log file in results/logs/.\n")

    t_all = time.time()
    running, last_beat = [], time.time()
    try:
        while queue or running or solo_q:
            while queue and len(running) < jobs:
                p = queue.pop(0)
                running.append(start(p))
                print(f"  {dt.datetime.now():%H:%M}  start  {p[0]:<14} (~{p[3]} min)  {p[4]}")
            if not queue and not running and solo_q:
                p = solo_q.pop(0)
                running.append(start(p))
                print(f"  {dt.datetime.now():%H:%M}  start  {p[0]:<14} (~{p[3]} min)  {p[4]}  [alone]")
            time.sleep(2)
            for r in [r for r in running if r["proc"].poll() is not None]:
                running.remove(r)
                finish(r)
            if time.time() - last_beat > 900 and running:
                last_beat = time.time()
                names = ", ".join(f"{r['p'][0]} ({(time.time() - r['t0']) / 60:.0f} min)"
                                  for r in running)
                print(f"  {dt.datetime.now():%H:%M}  still running: {names}; "
                      f"{len(queue)} waiting")
    except KeyboardInterrupt:
        print("\nInterrupted: stopping running experiments "
              "(run again with --resume to continue).")
        for r in running:
            r["proc"].terminate()
            r["log"].close()
        sys.exit(1)

    print("\n" + "=" * 78)
    print(f"Finished in {(time.time() - t_all) / 60:.1f} min")
    failed = [k for k, v in status.items() if not v["ok"]]
    print("All experiments succeeded." if not failed else f"FAILED: {', '.join(failed)}")


if __name__ == "__main__":
    main()
