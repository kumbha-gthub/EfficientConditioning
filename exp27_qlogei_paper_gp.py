#!/usr/bin/env python3
"""GP control for the q-LogEI comparison of the batch-size study (Section 7).

In Section 7, q-LogEI runs on BoTorch's default SingleTaskGP, while EI, CL-min
and KB run on the GP of Appendix B (scikit-learn, Matern-5/2 with ARD,
lengthscales in [0.005, 2], fixed jitter 1e-6, output standardization frozen
at the fit).  This script adds one arm that removes that difference:

  qlogei_pgp   q-LogEI computed on the GP of Appendix B.  At every batch the
               paper's GP is fitted exactly as for EI, CL-min and KB
               (utils.GPSurrogate, 5 restarts, random_state 0).  Its fitted
               hyperparameters and output standardization are copied into a
               GPyTorch model with the same kernel (ScaleKernel x Matern-5/2,
               ARD), zero mean on the standardized scale, and fixed noise 1e-6.
               The two models therefore have the same posterior; every batch
               checks this on 64 random points and stores the largest
               difference.  Acquisition and optimizer are those of the paper's
               q-LogEI arm: qLogExpectedImprovement with the best observation,
               optimize_acqf with 5 restarts and 256 raw samples, torch seed =
               run seed.

The other arms are run with the unchanged functions of exp24_batch_study.py:

  ei (q = 1), cl_min, kb      exp24_batch_study.run_sklearn
  qlogei                      exp24_batch_study.run_qlogei (BoTorch default GP)

Protocol: the 19 problems of exp24_problems.py, 2d Latin-hypercube points that
depend only on the seed, then 100 evaluations (for q = 16 the last batch has
four points), exactly as in exp24.  Results go to
results/data/exp27/<problem>/<method>_q<q>_s<seed>.json.

Usage
  python exp27_qlogei_paper_gp.py validate            equivalence and unit checks
  python exp27_qlogei_paper_gp.py run [workers]       all runs (resumable)
  python exp27_qlogei_paper_gp.py status              progress
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["EXP_GP_LS"] = "turbo"          # lengthscales in [0.005, 2]
os.environ["EXP_GP_YNORM"] = "frozen"      # exact conditioning

import sys
import json
import time
import traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp24_batch_study as E                                    # noqa: E402
from exp24_problems import problem_names, get_problem            # noqa: E402

VERSION = "exp27-v1"
BUDGET = E.BUDGET                     # 100
SEEDS = list(range(int(os.environ.get("EXP27_SEEDS", "10"))))
QS = [1, 4, 16]
METHODS = {1: ["ei", "qlogei", "qlogei_pgp"],
           4: ["cl_min", "kb", "qlogei", "qlogei_pgp"],
           16: ["cl_min", "kb", "qlogei", "qlogei_pgp"]}
N_CHECK = 64                          # points for the per-batch GP equivalence check
RESULTS = os.environ.get("EXP_RESULTS_DIR", os.path.join(HERE, "results"))
DATA_DIR = os.path.join(RESULTS, "data", "exp27")
CONFIG = dict(version=VERSION, exp24_version=E.VERSION, budget=BUDGET, xi=E.XI,
              gp="utils.GPSurrogate (Matern-5/2 ARD, ls [0.005, 2], alpha 1e-6, frozen y-scaling)",
              qlogei="qLogExpectedImprovement, optimize_acqf 5 restarts / 256 raw samples",
              qlogei_pgp="q-LogEI on the paper GP (hyperparameters copied into GPyTorch)")


# ------------------------------------------------------------------ paper GP in GPyTorch
def paper_gp_to_botorch(gp, X, y):
    """GPyTorch/BoTorch model with the posterior of the fitted paper GP `gp`
    (utils.GPSurrogate), on unit-cube inputs and negated standardized
    targets (BoTorch maximizes).  Returns (model, train_Y)."""
    import torch
    from botorch.models import SingleTaskGP
    from gpytorch.kernels import ScaleKernel, MaternKernel
    from gpytorch.means import ZeroMean
    k = gp.kernel_
    c = float(k.k1.constant_value)
    ls = np.atleast_1d(np.asarray(k.k2.length_scale, dtype=float))
    if float(k.k2.nu) != 2.5:
        raise ValueError("expected a Matern-5/2 kernel")
    m, sd = gp._ynorm
    d = X.shape[1]
    if ls.size == 1:
        ls = np.full(d, float(ls[0]))
    tX = torch.tensor(gp._s(X), dtype=torch.double)
    tY = torch.tensor(-(np.asarray(y, float) - m) / sd, dtype=torch.double).unsqueeze(-1)
    covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
    model = SingleTaskGP(tX, tY, train_Yvar=torch.full_like(tY, gp.noise),
                         covar_module=covar, mean_module=ZeroMean(),
                         outcome_transform=None)
    model = model.double()
    covar.base_kernel.lengthscale = torch.tensor(ls, dtype=torch.double).view(1, d)
    covar.outputscale = torch.tensor(c, dtype=torch.double)
    model.eval()
    return model, tY


def gp_difference(gp, model, Xc):
    """Largest absolute difference between the two posteriors at Xc
    (original units), relative to the output scale sd."""
    import torch
    m, sd = gp._ynorm
    mu_s, sd_s = gp.predict(Xc)
    with torch.no_grad():
        post = model.posterior(torch.tensor(gp._s(Xc), dtype=torch.double))
        mu_b = -post.mean.squeeze(-1).numpy() * sd + m
        sd_b = np.sqrt(np.maximum(post.variance.squeeze(-1).numpy(), 0.0)) * sd
    return float(max(np.max(np.abs(mu_s - mu_b)), np.max(np.abs(sd_s - sd_b))) / sd)


def run_qlogei_pgp(prob, q, seed, budget):
    """q-LogEI on the paper's GP; otherwise identical to exp24.run_qlogei."""
    import torch
    torch.set_num_threads(1)
    from botorch.acquisition import qLogExpectedImprovement
    from botorch.optim import optimize_acqf
    from utils import GPSurrogate, latin_hypercube, pairwise_distances
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    chk = np.random.RandomState(10_000 + seed)          # separate stream for the checks
    B, d = prob.bounds, prob.dim
    lo, w = B[:, 0], B[:, 1] - B[:, 0]
    n_init = 2 * d
    X = latin_hypercube(n_init, d, B, seed=seed)
    y = prob(X)
    best, mind, n_done, n_fail, errors, gpdiff = [float(np.min(y))], [], 0, 0, [], []
    unit = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
    while n_done < budget:
        aq = min(q, budget - n_done)
        try:
            gp = GPSurrogate(bounds=B)
            gp.fit(X, y, optimize=True)
            model, tY = paper_gp_to_botorch(gp, X, y)
            gpdiff.append(gp_difference(gp, model, lo + chk.uniform(size=(N_CHECK, d)) * w))
            acqf = qLogExpectedImprovement(model=model, best_f=tY.max().item())
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
                errors=errors[:5], incumbent=X[int(np.argmin(y))].tolist(),
                gp_max_rel_diff=float(max(gpdiff)) if gpdiff else None,
                gp_checks=len(gpdiff))


# ------------------------------------------------------------------ jobs
def all_jobs():
    return [(p, m, q, s) for p in problem_names() for s in SEEDS for q in QS for m in METHODS[q]]


def job_path(job):
    p, m, q, s = job
    return os.path.join(DATA_DIR, p, f"{m}_q{q}_s{s}.json")


def is_done(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        return r.get("config", {}).get("version") == VERSION and "best" in r
    except Exception:                                             # noqa: BLE001
        return False


def run_job(job):
    try:
        import warnings
        warnings.filterwarnings("ignore")
        p, m, q, s = job
        path = job_path(job)
        if is_done(path):
            return dict(job=job, skipped=True)
        prob = get_problem(p)
        t0 = time.time()
        if m == "qlogei_pgp":
            res = run_qlogei_pgp(prob, q, s, BUDGET)
        elif m == "qlogei":
            res = E.run_qlogei(prob, q, s, BUDGET)
        else:
            res = E.run_sklearn(prob, m, q, s, BUDGET)
        res.update(problem=p, method=m, q=q, seed=s, budget=BUDGET, final=res["best"][-1],
                   f_star=prob.f_star, dim=prob.dim, seconds=time.time() - t0, config=CONFIG)
        E._save(path, res)
        return dict(job=job, final=res["final"], seconds=res["seconds"], n_fail=res["n_fail"],
                    gp=res.get("gp_max_rel_diff"))
    except Exception:                                             # noqa: BLE001
        return dict(job=job, error=traceback.format_exc()[-3000:])


# ------------------------------------------------------------------ validate
def validate():
    """Unit checks: (1) the GPyTorch copy reproduces the paper GP's posterior;
    (2) the paper arms are the unchanged exp24 functions; (3) one short run
    of every arm."""
    import warnings
    warnings.filterwarnings("ignore")
    from utils import GPSurrogate, latin_hypercube
    worst = 0.0
    for p in problem_names():
        prob = get_problem(p)
        B, d = prob.bounds, prob.dim
        X = latin_hypercube(2 * d + 15, d, B, seed=3)
        y = prob(X)
        gp = GPSurrogate(bounds=B)
        gp.fit(X, y, optimize=True)
        model, _ = paper_gp_to_botorch(gp, X, y)
        Xc = B[:, 0] + np.random.RandomState(7).uniform(size=(500, d)) * (B[:, 1] - B[:, 0])
        diff = gp_difference(gp, model, np.vstack([Xc, X]))      # also at the data
        worst = max(worst, diff)
        print(f"  {p:22s} d={d:2d}  max |posterior difference| / sd = {diff:.2e}")
        assert diff < 1e-6, (p, diff)
    print(f"  GP equivalence ok on all 19 problems (worst {worst:.2e})")
    assert E.run_sklearn.__module__ == "exp24_batch_study" and E.run_qlogei.__module__ == "exp24_batch_study"
    prob = get_problem("Hartmann-6")
    for m, q in (("ei", 1), ("cl_min", 4), ("kb", 4), ("qlogei", 4), ("qlogei_pgp", 4)):
        t0 = time.time()
        if m == "qlogei_pgp":
            r = run_qlogei_pgp(prob, q, 0, 8)
        elif m == "qlogei":
            r = E.run_qlogei(prob, q, 0, 8)
        else:
            r = E.run_sklearn(prob, m, q, 0, 8)
        assert r["n_fail"] == 0 and len(r["best"]) == 9, (m, r["n_fail"], r["errors"])
        print(f"  {m:11s} q={q}: 8 evaluations ok in {time.time() - t0:.1f} s"
              + (f", GP diff {r['gp_max_rel_diff']:.1e}" if m == "qlogei_pgp" else ""))
    print("VALIDATE OK")


# ------------------------------------------------------------------ main
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "validate":
        return validate()
    jobs = all_jobs()
    todo = [j for j in jobs if not is_done(job_path(j))]
    if cmd == "status":
        print(f"{len(jobs) - len(todo)}/{len(jobs)} runs done")
        return
    if cmd != "run":
        sys.exit(__doc__)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    # seed by seed, so that an interrupted study leaves complete seeds;
    # within a seed, long jobs first (q-LogEI, high-dimensional problems)
    cost = {"qlogei": 3, "qlogei_pgp": 3, "ei": 2, "cl_min": 1, "kb": 1}
    todo.sort(key=lambda j: (j[3], -(cost[j[1]] * get_problem(j[0]).dim)))
    print(f"{len(todo)} runs to do with {workers} workers", flush=True)
    import multiprocessing as mp
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=20) as pool:
        for r in pool.imap_unordered(run_job, todo):
            n += 1
            p, m, q, s = r["job"]
            if "error" in r:
                print(f"[{n}/{len(todo)}] ERROR {p} {m} q={q} s={s}\n{r['error']}", flush=True)
            elif not r.get("skipped"):
                extra = f" gpdiff {r['gp']:.1e}" if r.get("gp") is not None else ""
                print(f"[{n}/{len(todo)}] {p} {m} q={q} s={s}: final {r['final']:.6g} "
                      f"({r['seconds']:.0f}s, fail {r['n_fail']}){extra}  "
                      f"elapsed {time.time() - t0:.0f}s", flush=True)
    print(f"all runs finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
