#!/usr/bin/env python3
"""Batch-size study of the paper.

Runs every combination of problem (exp24_problems.py), method, batch size
and seed, and saves each run to results/data/exp24/<problem>/.

Methods
  q = 1            ei, qlogei, ts, random
  q = 2, 4, 8, 16  cl_min, cl_mean, cl_max, kb, lp, qlogei, ts
Random search does not depend on q and is run once per seed.

Every run starts from 2d Latin-hypercube points that depend only on the
seed and then spends 100 evaluations (30 seeds).  EI, CL, KB, LP and TS use
the GP of utils.py; its hyperparameters and output standardization are
refitted once per batch and kept fixed while the batch is built.  EI uses
xi = 0.01.  EI and the LP acquisition are maximized by L-BFGS-B from the best
5 of 1000 uniform points, with the objective divided by the best random
value.  q-LogEI uses BoTorch's SingleTaskGP on the unit cube and
optimize_acqf (joint optimization, 5 restarts, 256 raw samples).  Thompson
sampling draws one joint posterior sample per batch member on 2000
candidates (1000 uniform points and 1000 Gaussian perturbations of the five
best points) and takes each sample's minimizer, without replacement.  A step
that raises an exception or returns a non-finite point is replaced by
uniform random points and counted in n_fail.

Finished runs are skipped, so the script can be stopped and restarted.

Usage
  python exp24_batch_study.py --check        check the setup
  python exp24_batch_study.py --quick        short test run
  python exp24_batch_study.py                full study
  python exp24_batch_study.py --workers 12   number of worker processes
  python exp24_batch_study.py --status       progress of the full study
"""

import os
# Single-threaded numerical libraries and the GP configuration of the paper;
# set before numpy, torch and utils are imported.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["EXP_GP_LS"] = "turbo"          # lengthscales in [0.005, 2]
os.environ["EXP_GP_YNORM"] = "frozen"      # exact conditioning

import sys
import json
import time
import argparse
import traceback
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from exp24_problems import problem_names, get_problem, EXP24_VERSION   # noqa: E402

VERSION = EXP24_VERSION
BUDGET = 100
N_SEEDS = 30
QS = [2, 4, 8, 16]
N_RAW, N_STARTS = 1000, 5
TS_CANDIDATES = 2000
XI = 0.01
METHODS_Q1 = ["ei", "qlogei", "ts", "random"]
METHODS_BATCH = ["cl_min", "kb", "lp", "qlogei", "ts", "cl_mean", "cl_max"]
CONFIG = dict(version=VERSION, gp_lengthscale_bounds=[0.005, 2.0], gp_output_scaling="frozen",
              acq_opt=f"best {N_STARTS} of {N_RAW} random points + L-BFGS-B", xi=XI,
              ts_candidates=TS_CANDIDATES, qlogei="unit cube, 5 restarts, 256 raw samples")

RESULTS = os.environ.get("EXP_RESULTS_DIR", os.path.join(HERE, "results"))

# Objective values at three fixed points per problem, used by --check.
CHECK_VALUES = {
    'Hartmann-6': [-1.0188180556734787, -0.012414925145716325, -0.08483588837341112],
    'Shekel-4': [-0.603752963373568, -0.6548804783133695, -0.15407642939967278],
    'Ackley-8': [6.593599079287213, 7.331646065955912, 10.740236669188551],
    'Styblinski-Tang-8': [-232.0, -162.44759999999997, -24.61880466472306],
    'Rosenbrock-8': [409.5, 483322.33687499946, 293757.7290907434],
    'Levy-10': [24.065024676319258, 14.921909147537717, 97.57614554063751],
    'LCBench-167168': [56.65316390991211, 57.10117721557617, 72.06645584106445],
    'LCBench-189873': [90.57947063446045, 99.58626130223274, 99.41696101427078],
    'LCBench-189906': [46.564178466796875, 49.416473388671875, 62.36553955078125],
    'XGBoost-12': [7.557445764541626, 5.940181016921997, 39.163875579833984],
    'XGBoost-1501': [21.051305532455444, 23.60207438468933, 65.47031104564667],
    'XGBoost-16': [9.304803609848022, 8.884161710739136, 45.78213691711426],
    'XGBoost-40499': [10.400408506393433, 3.812229633331299, 23.49591851234436],
    'RandomForest-16': [6.886947154998779, 8.596307039260864, 12.57779598236084],
    'RandomForest-42': [16.354387998580933, 17.15715527534485, 37.74636387825012],
    'ElasticNet-375': [15.876317024230957, 84.7338154911995, 84.4687208533287],
    'ElasticNet-458': [0.7990241050720215, 63.09475898742676, 63.1551057100296],
    'DecisionTree-14': [29.929548501968384, 90.68376123905182, 31.75595998764038],
    'DecisionTree-40499': [17.743968963623047, 91.55488833785057, 18.32600235939026],
}


# ------------------------------------------------------------------ jobs
def all_jobs(problems, seeds, qs, methods):
    jobs = []
    for p in problems:
        for s in seeds:
            for m in METHODS_Q1:
                if m in methods:
                    jobs.append((p, m, 1, s))
            for q in qs:
                for m in METHODS_BATCH:
                    if m in methods:
                        jobs.append((p, m, q, s))
    return jobs


def job_path(data_dir, job):
    p, m, q, s = job
    return os.path.join(data_dir, p, f"{m}_q{q}_s{s}.json")


def is_done(path):
    """A run counts as done if its file is complete and carries the current
    version tag."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        return r.get("config", {}).get("version") == VERSION and "best" in r
    except Exception:                                           # noqa: BLE001
        return False


_DIM = {}


def _dim(p):
    if p not in _DIM:
        _DIM[p] = get_problem(p).dim
    return _DIM[p]


def est_cost(job, budget):
    """Rough cost in seconds of one run on one core.  Only used to start long
    jobs first and for the rough ETA in the progress log."""
    p, m, q, s = job
    if m == "random":
        return 0.05
    d = _dim(p)
    rounds = int(np.ceil(budget / q))
    fit = 0.27 * d / 6                              # GP hyperparameter fit
    acq = 0.05 + 0.05 * d                           # one acquisition maximization
    if m == "qlogei":
        return rounds * (0.5 + 0.5 * q) * (0.6 + 0.4 * d / 6)
    if m == "ts":
        return rounds * (fit + 0.2)
    if m == "lp":
        return rounds * (fit + 0.3 * d / 6) + budget * acq * 1.3
    return rounds * fit + budget * acq


# ------------------------------------------------------------------ acquisition
def ei_values(gp, Xc, f_best):
    from utils import expected_improvement
    mu, sd = gp.predict(Xc)
    return expected_improvement(mu, sd, f_best, xi=XI)


def lp_values(gp, Xc, f_best, batch, L, M, mu_b, sd_b):
    """EI times the Lipschitz penalizer of Gonzalez et al. (2016), minimization
    form, as in utils.optimize_acq_lp_lipschitz (vectorized)."""
    from scipy.stats import norm
    v = ei_values(gp, Xc, f_best)
    if len(batch):
        Bm = np.atleast_2d(np.array(batch))
        dist = np.linalg.norm(Xc[:, None, :] - Bm[None, :, :], axis=2)
        v = v * np.prod(norm.cdf((L * dist - (mu_b - M)[None, :]) / sd_b[None, :]), axis=1)
    return v


def maximize(fun, bounds, n_raw=N_RAW, n_starts=N_STARTS):
    """Maximize fun (vectorized: (n, d) -> (n,)) over the box: best n_starts
    of n_raw uniform points (np.random, seeded per run) as L-BFGS-B starts,
    objective scaled by the best random value."""
    from scipy.optimize import minimize
    lo, hi = bounds[:, 0], bounds[:, 1]
    R = lo + np.random.uniform(size=(n_raw, len(lo))) * (hi - lo)
    v = fun(R)
    v = np.where(np.isfinite(v), v, -np.inf)
    order = np.argsort(-v)
    best_x, best_v = R[order[0]].copy(), float(v[order[0]])
    scale = best_v if (np.isfinite(best_v) and best_v > 1e-300) else 1.0
    obj = lambda x: -float(fun(x.reshape(1, -1))[0]) / scale            # noqa: E731
    for i in order[:n_starts]:
        try:
            res = minimize(obj, R[i], bounds=list(bounds), method="L-BFGS-B")
            x = np.clip(res.x, bounds[:, 0], bounds[:, 1])
            val = float(fun(x.reshape(1, -1))[0])          # value at the clipped point
            if np.all(np.isfinite(x)) and np.isfinite(val) and val > best_v:
                best_x, best_v = x.copy(), val
        except Exception:                                       # noqa: BLE001
            continue
    return best_x


def ts_batch(gp, X, y, bounds, q, rng, n_cand=TS_CANDIDATES):
    """Parallel Thompson sampling on a candidate set."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    w = hi - lo
    d = len(lo)
    n_uni = n_cand // 2
    C_uni = lo + rng.uniform(size=(n_uni, d)) * w
    top = X[np.argsort(y)[:5]]
    idx = rng.randint(0, len(top), size=n_cand - n_uni)
    C_loc = np.clip(top[idx] + 0.1 * w * rng.randn(n_cand - n_uni, d), lo, hi)
    C = np.unique(np.vstack([C_uni, C_loc]), axis=0)             # clipping makes copies
    Cu, Xu = (C - lo) / w, (X - lo) / w
    dmin = np.min(np.linalg.norm(Cu[:, None, :] - Xu[None, :, :], axis=2), axis=1)
    C = C[dmin > 1e-9]                                           # never re-evaluate
    mu, cov = gp.model.predict(gp._s(C), return_cov=True)       # standardized scale
    cov = 0.5 * (cov + cov.T)
    L, jitter = None, 1e-8
    while L is None:
        try:
            L = np.linalg.cholesky(cov + jitter * np.eye(len(C)))
        except np.linalg.LinAlgError:
            jitter *= 10
            if jitter > 1e-2:
                raise
    F = mu[:, None] + L @ rng.randn(len(C), q)
    chosen = []
    for j in range(q):
        for i in np.argsort(F[:, j]):
            if i not in chosen:
                chosen.append(int(i))
                break
    return C[chosen]


# ------------------------------------------------------------------ methods
def run_sklearn(prob, method, q, seed, budget):
    """EI, CL (min/mean/max), KB, LP, TS and random search with the GP of utils.py."""
    from utils import GPSurrogate, estimate_lipschitz, latin_hypercube, pairwise_distances
    if method not in ("random", "ts", "ei", "lp", "cl_min", "cl_mean", "cl_max", "kb"):
        raise ValueError(f"unknown method {method}")
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    B, d = prob.bounds, prob.dim
    n_init = 2 * d
    X = latin_hypercube(n_init, d, B, seed=seed)
    y = prob(X)
    best, mind, n_done, n_fail, errors = [float(np.min(y))], [], 0, 0, []

    def rand(k):
        return B[:, 0] + rng.uniform(size=(k, d)) * (B[:, 1] - B[:, 0])

    while n_done < budget:
        aq = min(q, budget - n_done)
        if method == "random":
            batch = rand(aq)
        else:
            try:
                gp = GPSurrogate(bounds=B)
                gp.fit(X, y, optimize=True)
                f_best = float(np.min(y))
                if method == "ts":
                    batch = ts_batch(gp, X, y, B, aq, rng)
                elif method == "ei" or aq == 1:
                    batch = maximize(lambda Z: ei_values(gp, Z, f_best), B).reshape(1, -1)
                elif method == "lp":
                    L = estimate_lipschitz(gp, B, seed=1000 * seed + n_done)
                    bl = []
                    for j in range(aq):
                        if bl:
                            mu_b, sd_b = gp.predict(np.array(bl))
                            sd_b = np.maximum(sd_b, 1e-8)
                        else:
                            mu_b = sd_b = None
                        x = maximize(lambda Z: lp_values(gp, Z, f_best, bl, L, f_best, mu_b, sd_b), B)
                        bl.append(x.copy())
                    batch = np.array(bl)
                elif method in ("cl_min", "cl_mean", "cl_max", "kb"):
                    bl, Xa, ya = [], X.copy(), y.copy()
                    for j in range(aq):
                        gj = GPSurrogate(bounds=B)
                        gj.kernel_ = gp.kernel_                    # theta and y-scaling fixed
                        gj.fit(Xa, ya, optimize=False)
                        inc = float(np.min(ya))                    # incumbent includes the lies
                        x = maximize(lambda Z: ei_values(gj, Z, inc), B)
                        bl.append(x)
                        if method == "kb":
                            yl = float(gj.predict(x.reshape(1, -1))[0][0])
                        else:                                      # CL: observed values only
                            yl = float({"cl_min": np.min, "cl_mean": np.mean,
                                        "cl_max": np.max}[method](y))
                        Xa = np.vstack([Xa, x.reshape(1, -1)])
                        ya = np.append(ya, yl)
                    batch = np.array(bl)
                batch = np.atleast_2d(np.asarray(batch, dtype=float))
                if batch.shape != (aq, d) or not np.all(np.isfinite(batch)):
                    raise FloatingPointError(f"bad batch of shape {batch.shape}")
            except Exception as e:                                # noqa: BLE001
                n_fail += aq
                errors.append(repr(e)[:300])
                batch = rand(aq)
        X = np.vstack([X, batch])
        y = np.append(y, prob(batch))
        if len(batch) > 1:
            mind.append(float(np.min(pairwise_distances((batch - B[:, 0]) / (B[:, 1] - B[:, 0])))))
        for _ in range(len(batch)):
            n_done += 1
            best.append(float(np.min(y[:n_init + n_done])))
    return dict(best=best, min_dist_unit=mind, n_init=n_init, n_fail=n_fail,
                errors=errors[:5], incumbent=X[int(np.argmin(y))].tolist())


def run_qlogei(prob, q, seed, budget):
    """BoTorch q-LogEI with joint optimization of the batch, on the unit cube."""
    import torch
    torch.set_num_threads(1)
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import qLogExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from utils import latin_hypercube, pairwise_distances
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    B, d = prob.bounds, prob.dim
    lo, w = B[:, 0], B[:, 1] - B[:, 0]
    n_init = 2 * d
    X = latin_hypercube(n_init, d, B, seed=seed)
    y = prob(X)
    best, mind, n_done, n_fail, errors = [float(np.min(y))], [], 0, 0, []
    unit = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
    while n_done < budget:
        aq = min(q, budget - n_done)
        try:
            tX = torch.tensor((X - lo) / w, dtype=torch.double)
            tY = -torch.tensor(y, dtype=torch.double).unsqueeze(-1)   # BoTorch maximizes
            gp = SingleTaskGP(tX, tY)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            acqf = qLogExpectedImprovement(model=gp, best_f=tY.max().item())
            cand, _ = optimize_acqf(acqf, bounds=unit, q=aq, num_restarts=5, raw_samples=256)
            batch = lo + np.clip(cand.detach().numpy(), 0.0, 1.0) * w
            if batch.shape != (aq, d) or not np.all(np.isfinite(batch)):
                raise FloatingPointError(f"bad batch of shape {batch.shape}")
        except Exception as e:                                    # noqa: BLE001
            n_fail += aq
            errors.append(repr(e)[:300])
            batch = lo + rng.uniform(size=(aq, d)) * w
        X = np.vstack([X, batch])
        y = np.append(y, prob(batch))
        if len(batch) > 1:
            mind.append(float(np.min(pairwise_distances((batch - lo) / w))))
        for _ in range(len(batch)):
            n_done += 1
            best.append(float(np.min(y[:n_init + n_done])))
    return dict(best=best, min_dist_unit=mind, n_init=n_init, n_fail=n_fail,
                errors=errors[:5], incumbent=X[int(np.argmin(y))].tolist())


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    for attempt in range(20):            # Windows: antivirus / indexer may hold the file
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    os.replace(tmp, path)


def run_job(job, data_dir, budget):
    """Run one (problem, method, q, seed) and save it.  Never raises: errors
    are returned to the main process, which logs them and carries on."""
    try:
        import warnings
        warnings.filterwarnings("ignore")
        p, m, q, s = job
        path = job_path(data_dir, job)
        if is_done(path):
            return dict(job=job, skipped=True)
        prob = get_problem(p)
        t0 = time.time()
        res = run_qlogei(prob, q, s, budget) if m == "qlogei" else run_sklearn(prob, m, q, s, budget)
        res.update(problem=p, method=m, q=q, seed=s, budget=budget, final=res["best"][-1],
                   f_star=prob.f_star, dim=prob.dim, seconds=time.time() - t0, config=CONFIG)
        _save(path, res)
        return dict(job=job, final=res["final"], seconds=res["seconds"], n_fail=res["n_fail"])
    except Exception:                                            # noqa: BLE001
        return dict(job=job, error=traceback.format_exc()[-3000:])


# ------------------------------------------------------------------ helpers
def ensure_models():
    """Download or verify the YAHPO model files (see yahpo_data.py)."""
    import yahpo_data
    return yahpo_data.ensure()


def acquire_lock(path):
    """OS-level lock so that two copies cannot run at once.  The lock is
    released automatically when the process ends (also after a crash)."""
    f = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def _versions():
    from importlib import metadata
    out = {}
    for name, dist in (("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
                       ("matplotlib", "matplotlib"), ("onnxruntime", "onnxruntime"),
                       ("torch", "torch"), ("botorch", "botorch"), ("gpytorch", "gpytorch")):
        try:
            out[name] = metadata.version(dist)
        except Exception:                                       # noqa: BLE001
            out[name] = None
    return out


def check_setup():
    """Check the packages, evaluate every problem at three fixed points and
    compare with CHECK_VALUES, then run a few evaluations of every method."""
    ok = ensure_models()
    if not ok:
        print("SETUP PROBLEM - see FAIL lines")
        return False
    for mod in ("numpy", "scipy", "sklearn", "matplotlib", "onnxruntime", "torch",
                "botorch", "gpytorch"):
        try:
            m = __import__(mod)
            print(f"  ok   {mod:12s} {getattr(m, '__version__', '')}")
        except Exception as e:                                  # noqa: BLE001
            ok = False
            print(f"  FAIL {mod:12s} {e}")
    try:
        import utils                                            # noqa: F401
        print("  ok   utils.py")
    except Exception as e:                                      # noqa: BLE001
        ok = False
        print(f"  FAIL utils.py     {e}")
    expected = CHECK_VALUES
    for name in problem_names():
        p = get_problem(name)
        U = np.array([[0.3] * p.dim, [0.71] * p.dim, list(np.linspace(0.05, 0.95, p.dim))])
        got = p(p.bounds[:, 0] + U * (p.bounds[:, 1] - p.bounds[:, 0]))
        exp = np.array(expected[name])
        err = float(np.max(np.abs(got - exp) / np.maximum(1.0, np.abs(exp))))
        flag = "ok  " if err < 1e-4 else "FAIL"
        ok &= err < 1e-4
        print(f"  {flag} {name:22s} d={p.dim:2d}  relative difference {err:.1e}")
    p = get_problem("Hartmann-6")
    for m, q in (("ei", 1), ("cl_min", 2), ("kb", 2), ("lp", 2), ("ts", 2), ("qlogei", 2)):
        t0 = time.time()
        r = run_qlogei(p, q, 0, 4) if m == "qlogei" else run_sklearn(p, m, q, 0, 4)
        good = len(r["best"]) == 5 and r["n_fail"] == 0
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} 4 evaluations of {m:7s} q={q}: {time.time() - t0:5.1f} s"
              + ("" if good else f"  errors: {r['errors']}"))
    print("SETUP OK" if ok else "SETUP PROBLEM - see FAIL lines")
    return ok


def write_environment(path, workers):
    import platform
    env = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), python=sys.version,
               platform=platform.platform(), processor=platform.processor(),
               cpu_count=os.cpu_count(), workers=workers, versions=_versions(), config=CONFIG)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=1)


def run_analysis():
    import subprocess
    script = os.path.join(HERE, "analysis", "exp24_analysis.py")
    if not os.path.exists(script):
        script = os.path.join(HERE, "exp24_analysis.py")
    print(f"\nRunning the analysis: {script}")
    try:
        subprocess.run([sys.executable, "-X", "utf8", script, RESULTS], check=False)
    except Exception as e:                                      # noqa: BLE001
        print(f"Analysis failed ({e}); run it by hand: python {script}")


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 problems, 2 seeds, budget 16")
    ap.add_argument("--status", action="store_true", help="only report progress")
    ap.add_argument("--check", action="store_true", help="check the setup and exit")
    ap.add_argument("--no-analysis", action="store_true", help="do not run the analysis at the end")
    ap.add_argument("--problems", default="", help="comma-separated subset")
    ap.add_argument("--methods", default="", help="comma-separated subset")
    ap.add_argument("--qs", default="", help="comma-separated subset of 2,4,8,16")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check_setup() else 1)

    if not ensure_models():
        sys.exit("YAHPO model files are missing; see the messages above.")
    budget = BUDGET
    problems = problem_names()
    seeds = list(range(args.seeds))
    qs = QS
    methods = sorted(set(METHODS_Q1 + METHODS_BATCH))
    data_dir = os.path.join(RESULTS, "data", "exp24")
    if args.quick:
        budget, seeds, qs = 16, [0, 1], [2, 8]
        problems = ["Hartmann-6", "XGBoost-12"]
        data_dir = os.path.join(RESULTS, "data", "exp24_quick")
    if args.problems:
        problems = [p.strip() for p in args.problems.split(",")]
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    if args.qs:
        qs = [int(v) for v in args.qs.split(",")]
    for p in problems:
        get_problem(p)                       # fails early on a typo

    jobs = all_jobs(problems, seeds, qs, methods)
    todo = [j for j in jobs if not is_done(job_path(data_dir, j))]
    cost = {j: est_cost(j, budget) for j in jobs}
    tot, left = sum(cost.values()), sum(cost[j] for j in todo)
    print(f"Experiment 24 ({VERSION}): {len(problems)} problems, {len(seeds)} seeds, "
          f"q in {[1] + qs}, budget {budget}")
    print(f"  runs: {len(jobs)} in total, {len(jobs) - len(todo)} done, {len(todo)} to do "
          f"({100 * (1 - left / tot):.1f}% of the estimated work done)")
    if args.status:
        return
    if not todo:
        if not args.quick and not args.no_analysis:
            run_analysis()
        return
    print(f"  rough estimate for what is left: {left / 3600:.0f} core-hours, about "
          f"{left / 3600 / args.workers:.1f} h with {args.workers} workers on a fast CPU "
          f"(the log shows a measured ETA once runs finish)")

    log_dir = os.path.join(RESULTS, "logs")
    os.makedirs(log_dir, exist_ok=True)
    lock = acquire_lock(os.path.join(log_dir, "exp24_quick.lock" if args.quick else "exp24.lock"))
    if lock is None:
        print("Another copy of this experiment is already running (close that window first, "
              "or wait for it to finish).")
        sys.exit(1)
    log_path = os.path.join(log_dir, "exp24_quick.txt" if args.quick else "exp24_progress.txt")
    log = open(log_path, "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    import platform
    if not args.quick:
        write_environment(os.path.join(log_dir, "exp24_environment.json"), args.workers)
    say(f"# started {time.strftime('%Y-%m-%d %H:%M:%S')}  {VERSION}  workers={args.workers}  "
        f"python={platform.python_version()}  todo={len(todo)}")
    todo.sort(key=lambda j: -cost[j])                     # long jobs first
    t0 = time.time()
    done_cost, n_ok, n_err, n_fail_runs = 0.0, 0, 0, 0
    finished = set()

    def handle(r):
        nonlocal done_cost, n_ok, n_err, n_fail_runs
        j = tuple(r["job"])
        finished.add(j)
        done_cost += cost[j]
        if "error" in r:
            n_err += 1
            say(f"ERROR {j}\n{r['error']}")
            return
        n_ok += 1
        if r.get("skipped"):
            return
        if r.get("n_fail"):
            n_fail_runs += 1
        el = time.time() - t0
        eta = el / done_cost * (left - done_cost) if done_cost > 0 else float("nan")
        say(f"{n_ok + n_err:6d}/{len(todo)}  {j[0]:20s} {j[1]:7s} q={j[2]:2d} seed={j[3]:2d}  "
            f"final={r['final']:.5g}  {r['seconds']:6.1f}s"
            + (f"  [{r['n_fail']} fallback points]" if r.get("n_fail") else "")
            + f"  elapsed {el / 3600:5.2f} h  rough ETA {eta / 3600:5.2f} h")

    state = dict(pool=None, futs={})

    def kill_pool():
        """Stop a pool now: cancel queued runs and terminate the workers."""
        import threading
        from concurrent.futures import InvalidStateError
        default_hook = threading.excepthook

        def quiet(a):              # the executor's helper thread may complain about
            if not issubclass(a.exc_type, InvalidStateError):   # cancelled futures
                default_hook(a)
        threading.excepthook = quiet
        pool, futs = state["pool"], state["futs"]
        if pool is None:
            return
        for f in futs:
            f.cancel()
        procs = list((getattr(pool, "_processes", None) or {}).values())
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:                                       # noqa: BLE001
            pass
        for pr in procs:
            try:
                pr.terminate()
            except Exception:                                   # noqa: BLE001
                pass
        state["pool"], state["futs"] = None, {}

    try:
        if args.workers <= 1:
            for j in todo:
                handle(run_job(j, data_dir, budget))
        else:
            import random
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
            from concurrent.futures.process import BrokenProcessPool
            pending, stalls, attempt = list(todo), 0, 0
            while pending:
                # (no max_tasks_per_child: it can hang the pool in Python 3.11/3.12)
                state["pool"] = ProcessPoolExecutor(max_workers=args.workers,
                                                    mp_context=mp.get_context("spawn"))
                state["futs"] = {state["pool"].submit(run_job, j, data_dir, budget): j
                                 for j in pending}
                waiting, broken, last = set(state["futs"]), False, time.time()
                while waiting and not broken:
                    done, waiting = wait(waiting, timeout=600, return_when=FIRST_COMPLETED)
                    if not done and time.time() - last > 3600:
                        say("# note: no run has finished for an hour; still waiting")
                        last = time.time()
                    for f in done:
                        last = time.time()
                        try:
                            handle(f.result())
                        except BrokenProcessPool:
                            broken = True
                        except Exception:                           # noqa: BLE001
                            handle(dict(job=state["futs"][f], error=traceback.format_exc()[-3000:]))
                if not broken:
                    state["pool"].shutdown(wait=True)
                    state["pool"], state["futs"] = None, {}
                    break
                kill_pool()
                left_now = [j for j in pending if j not in finished
                            and not is_done(job_path(data_dir, j))]
                stalls = stalls + 1 if len(left_now) == len(pending) else 0
                attempt += 1
                if stalls >= 3:
                    say(f"# the worker pool died 3 times in a row without progress; stopping. "
                        f"{len(left_now)} runs are left. Run the same command again to retry them.")
                    break
                random.Random(attempt).shuffle(left_now)        # move a crashing run away
                pending = left_now
                say(f"# a worker process died; restarting the pool for the "
                    f"{len(pending)} remaining runs (restart {attempt})")
    except KeyboardInterrupt:
        kill_pool()
        say("# interrupted; finished runs are saved. Run the same command again to resume.")
        return
    say(f"# finished {time.strftime('%Y-%m-%d %H:%M:%S')}  ok={n_ok} errors={n_err} "
        f"runs-with-fallback-points={n_fail_runs}  wall {(time.time() - t0) / 3600:.2f} h")
    remaining = [j for j in jobs if not is_done(job_path(data_dir, j))]
    if remaining:
        say(f"# {len(remaining)} runs are not done. Run the same command again to retry them.")
    elif not args.quick and not args.no_analysis:
        run_analysis()


if __name__ == "__main__":
    main()
