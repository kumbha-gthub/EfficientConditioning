#!/usr/bin/env python3
"""Optimizer budget of q-LogEI at q = 16.

q-LogEI with the budget of the batch-size study (optimize_acqf with 5
restarts and 256 raw samples) against q-LogEI with 20 restarts and 2048 raw
samples, with CL-min as a reference.  Hartmann-6D (qd = 96) and Levy-10D
(qd = 160), seeds 0-9, 2d Latin-hypercube points shared by the three arms of
a seed, then 100 evaluations in batches of 16 (the last batch has 4 points).

  q-LogEI  SingleTaskGP on the negated targets with inputs in the unit cube,
           fit_gpytorch_mll with the library defaults, qLogExpectedImprovement
           with the best negated observation, optimize_acqf; torch seed = seed.
  CL-min   the GP of Appendix B (ConstantKernel x Matern-5/2 with ARD,
           lengthscales in [0.005, 2] with initial value 0.5, alpha = 1e-6,
           5 restarts, random_state = 0, output standardization stored at the
           fit), EI with xi = 0.01, L-BFGS-B from the best 5 of 1000 uniform
           points with the objective divided by the best random value.

Metric: normalized regret (f_best - f_min) / (f_init - f_min), with f_init
the best value of the initial design.

Usage
  python exp25_qlogei_budget.py validate        checks of the components
  python exp25_qlogei_budget.py run [workers]   all 60 runs (default 7 workers)
  python exp25_qlogei_budget.py summarize       table and Wilcoxon tests
Finished runs are skipped, so an interrupted run can be restarted.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
from scipy import optimize, stats
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Some BLAS builds warn inside matmul; non-finite results are rejected below.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*encountered in matmul.*")

BUDGET = 100
Q = 16
XI = 0.01
ALPHA = 1e-6
SIGMA_FLOOR = 1e-10
SEEDS = list(range(10))
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "data", "exp25_qlogei_budget")

ARMS = {
    "qlogei_default": {"num_restarts": 5, "raw_samples": 256},
    "qlogei_scaled": {"num_restarts": 20, "raw_samples": 2048},
    "cl_min": {},
}


# ------------------------------------------------------------------ problems
def hartmann6(x):
    """Hartmann-6D on [0,1]^6, minimum -3.32237."""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])
    x = np.atleast_2d(x)
    inner = np.sum(A[None] * (x[:, None, :] - P[None]) ** 2, axis=-1)
    return -np.sum(alpha[None] * np.exp(-inner), axis=-1)


def levy(x):
    """Levy on [-10,10]^d, minimum 0 at (1,...,1)."""
    x = np.atleast_2d(x)
    w = 1.0 + (x - 1.0) / 4.0
    t1 = np.sin(np.pi * w[:, 0]) ** 2
    t3 = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
    wi = w[:, :-1]
    t2 = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(np.pi * wi + 1) ** 2), axis=1)
    return t1 + t2 + t3


PROBLEMS = {
    "hartmann6": {"fn": hartmann6, "dim": 6, "lb": 0.0, "ub": 1.0,
                  "fmin": -3.32237},
    "levy10": {"fn": levy, "dim": 10, "lb": -10.0, "ub": 10.0, "fmin": 0.0},
}


def make_objective(problem):
    p = PROBLEMS[problem]

    def f(u):
        """u in [0,1]^d -> objective value on the original domain."""
        u = np.atleast_2d(np.clip(u, 0.0, 1.0))
        x = p["lb"] + u * (p["ub"] - p["lb"])
        return p["fn"](x)

    return f, p["dim"], p["fmin"]


def batch_schedule():
    sizes = [Q] * (BUDGET // Q)
    if BUDGET % Q:
        sizes.append(BUDGET % Q)
    return sizes


# ------------------------------------------------------------------ CL-min
class PaperGP:
    def __init__(self, dim):
        self.dim = dim

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.mean = float(np.mean(y))
        self.std = float(np.std(y)) or 1.0
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=[0.5] * self.dim, length_scale_bounds=(0.005, 2.0),
            nu=2.5)
        self.gpr = GaussianProcessRegressor(
            kernel=kernel, alpha=ALPHA, n_restarts_optimizer=5,
            normalize_y=False, random_state=0)
        self.gpr.fit(np.asarray(X), (y - self.mean) / self.std)
        self.kernel_ = self.gpr.kernel_
        return self

    def condition(self, X_all, y_all):
        """Fitted kernel, no optimizer, stored output standardization."""
        y_all = np.asarray(y_all, dtype=float)
        gpr = GaussianProcessRegressor(
            kernel=self.kernel_, alpha=ALPHA, optimizer=None, normalize_y=False)
        gpr.fit(np.asarray(X_all), (y_all - self.mean) / self.std)
        clone = PaperGP(self.dim)
        clone.mean, clone.std, clone.gpr, clone.kernel_ = (
            self.mean, self.std, gpr, self.kernel_)
        return clone

    def predict(self, X):
        mu_s, sd_s = self.gpr.predict(np.atleast_2d(X), return_std=True)
        return mu_s * self.std + self.mean, sd_s * self.std


def expected_improvement(mu, sd, incumbent):
    active = sd < SIGMA_FLOOR
    s = np.maximum(sd, SIGMA_FLOOR)
    delta = incumbent - XI - mu
    z = delta / s
    ei = delta * stats.norm.cdf(z) + s * stats.norm.pdf(z)
    return np.where(active, 0.0, np.maximum(ei, 0.0))


def maximize_ei(model, incumbent, rng, dim):
    cands = rng.uniform(size=(1000, dim))
    mu, sd = model.predict(cands)
    vals = expected_improvement(mu, sd, incumbent)
    order = np.argsort(vals)[::-1]
    best_x, best_v = cands[order[0]], vals[order[0]]
    scale = best_v if best_v > 0 else 1.0

    def neg(x):
        m, s = model.predict(x.reshape(1, -1))
        return -expected_improvement(m, s, incumbent)[0] / scale

    for idx in order[:5]:
        res = optimize.minimize(neg, cands[idx], method="L-BFGS-B",
                                bounds=[(0.0, 1.0)] * dim)
        v = -res.fun * scale
        if np.isfinite(v) and np.all(np.isfinite(res.x)) and v > best_v:
            best_x, best_v = np.clip(res.x, 0.0, 1.0), v
    return best_x


def select_batch_cl_min(model, X_obs, y_obs, q, rng, dim):
    """CL-min with fixed hyperparameters; the incumbent includes the lies."""
    X_cond = list(np.asarray(X_obs))
    y_cond = list(np.asarray(y_obs, dtype=float))
    cl_lie = float(np.min(y_obs))
    batch = []
    current = model
    for j in range(q):
        incumbent = float(np.min(y_cond))
        xj = maximize_ei(current, incumbent, rng, dim)
        batch.append(xj)
        X_cond.append(xj)
        y_cond.append(cl_lie)
        if j < q - 1:
            current = model.condition(X_cond, y_cond)
    return np.array(batch)


# ------------------------------------------------------------------ q-LogEI
def select_batch_qlogei(X_obs, y_obs, q, num_restarts, raw_samples, dim):
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    X = torch.tensor(np.asarray(X_obs), dtype=torch.double)
    Y = -torch.tensor(np.asarray(y_obs, dtype=float),
                      dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(X, Y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    acq = qLogExpectedImprovement(model, best_f=Y.max())
    bounds = torch.stack([torch.zeros(dim, dtype=torch.double),
                          torch.ones(dim, dtype=torch.double)])
    cand, _ = optimize_acqf(acq, bounds=bounds, q=q,
                            num_restarts=num_restarts, raw_samples=raw_samples)
    return cand.detach().numpy()


# ------------------------------------------------------------------ one run
def run_one(problem, arm, seed):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    f, dim, fmin = make_objective(problem)
    t0 = time.time()
    init = qmc.LatinHypercube(d=dim, seed=seed).random(2 * dim)
    X_obs = [row for row in init]
    y_obs = list(f(init))
    finit = float(np.min(y_obs))
    rng = np.random.default_rng([seed, 7, dim])
    batch_times = []
    for bsize in batch_schedule():
        tb = time.time()
        if arm == "cl_min":
            model = PaperGP(dim).fit(np.array(X_obs), np.array(y_obs))
            batch = select_batch_cl_min(model, X_obs, y_obs, bsize, rng, dim)
        else:
            batch = select_batch_qlogei(X_obs, y_obs, bsize, dim=dim,
                                        **ARMS[arm])
        batch_times.append(round(time.time() - tb, 2))
        for x in batch:
            y_obs.append(float(f(x)[0]))
            X_obs.append(np.clip(x, 0.0, 1.0))
    fbest = float(np.min(y_obs))
    denom = finit - fmin
    regret = 0.0 if denom <= 0 else max(0.0, (fbest - fmin) / denom)
    return {
        "problem": problem, "arm": arm, "seed": seed,
        "finit": finit, "fbest": fbest, "regret": regret,
        "n_evals": len(y_obs) - 2 * dim,
        "batch_times_s": batch_times,
        "wall_time_s": round(time.time() - t0, 1),
    }


def run_path(problem, arm, seed):
    return os.path.join(RESULTS_DIR, f"{problem}_{arm}_seed{seed}.json")


def worker(task):
    problem, arm, seed = task
    path = run_path(problem, arm, seed)
    if os.path.exists(path):
        return f"skip {problem} {arm} seed={seed}"
    res = run_one(problem, arm, seed)
    with open(path, "w") as fh:
        json.dump(res, fh)
    return (f"done {problem} {arm} seed={seed}: regret {res['regret']:.3f} "
            f"({res['wall_time_s']:.0f}s)")


# ------------------------------------------------------------------ checks
def validate():
    xh = np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]])
    v = hartmann6(xh)[0]
    assert abs(v - (-3.32237)) < 1e-4, v
    print(f"1. Hartmann-6D at its minimizer: {v:.5f}  ok")
    v = levy(np.ones((1, 10)))[0]
    assert abs(v) < 1e-12, v
    print(f"2. Levy-10D at (1,...,1): {v:.1e}  ok")
    assert batch_schedule() == [16] * 6 + [4]
    print("3. batches 6 x 16 + 4  ok")
    a = qmc.LatinHypercube(d=6, seed=5).random(12)
    b = qmc.LatinHypercube(d=6, seed=5).random(12)
    assert np.array_equal(a, b)
    print("4. initial design reproducible  ok")
    f, dim, fmin = make_objective("levy10")
    u = np.full((1, 10), 0.55)
    assert abs(f(u)[0] - levy(-10 + u * 20)[0]) < 1e-12
    print("5. domain scaling  ok")
    f, dim, _ = make_objective("hartmann6")
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(12, dim))
    y = [float(f(x)[0]) for x in X]
    gp = PaperGP(dim).fit(X, np.array(y))
    batch = select_batch_cl_min(gp, X, y, 4, rng, dim)
    d = np.linalg.norm(batch[:, None] - batch[None], axis=-1)
    dmin = d[np.triu_indices(4, 1)].min()
    assert dmin > 1e-3, dmin
    print(f"6. CL-min batch without duplicates (min distance {dmin:.3f})  ok")
    cand = select_batch_qlogei(X, y, 4, num_restarts=2, raw_samples=64, dim=dim)
    assert cand.shape == (4, dim) and cand.min() >= 0 and cand.max() <= 1
    print("7. q-LogEI batch shape and bounds  ok")


# ------------------------------------------------------------------ summary
def summarize():
    rows = {}
    for problem in PROBLEMS:
        for arm in ARMS:
            res = [json.load(open(run_path(problem, arm, s)))
                   for s in SEEDS if os.path.exists(run_path(problem, arm, s))]
            if res:
                rows[(problem, arm)] = sorted(res, key=lambda r: r["seed"])
    print(f"{sum(len(v) for v in rows.values())} runs\n")
    print(f"{'problem':<11} {'arm':<16} {'n':>3} {'regret (mean +- sd)':>20} "
          f"{'s/batch':>8}")
    for (problem, arm), res in rows.items():
        r = np.array([x["regret"] for x in res])
        bt = np.concatenate([x["batch_times_s"] for x in res])
        print(f"{problem:<11} {arm:<16} {len(res):>3} "
              f"{r.mean():>10.3f} +- {r.std():<7.3f} {bt.mean():>8.1f}")
        print("    per seed: " + " ".join(f"{v:.3f}" for v in r))
    print("\nPaired Wilcoxon tests (two-sided) on the final regret:")
    for problem in PROBLEMS:
        for a, b in [("qlogei_default", "qlogei_scaled"),
                     ("qlogei_scaled", "cl_min"),
                     ("qlogei_default", "cl_min")]:
            if (problem, a) not in rows or (problem, b) not in rows:
                continue
            ra = np.array([x["regret"] for x in rows[(problem, a)]])
            rb = np.array([x["regret"] for x in rows[(problem, b)]])
            n = min(len(ra), len(rb))
            p = stats.wilcoxon(ra[:n], rb[:n]).pvalue
            print(f"  {problem}: {a} ({ra[:n].mean():.3f}) vs {b} "
                  f"({rb[:n].mean():.3f}): p = {p:.4f}, first lower on "
                  f"{int(np.sum(ra[:n] < rb[:n]))}/{n}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "validate":
        validate()
    elif cmd == "summarize":
        summarize()
    elif cmd == "run":
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        os.makedirs(RESULTS_DIR, exist_ok=True)
        tasks = [(p, a, s) for p in PROBLEMS for a in ARMS for s in SEEDS]
        pending = [t for t in tasks if not os.path.exists(run_path(*t))]
        print(f"{len(pending)} of {len(tasks)} runs to do")
        import multiprocessing as mp
        t0 = time.time()
        with mp.get_context("spawn").Pool(workers) as pool:
            for i, msg in enumerate(pool.imap_unordered(worker, pending), 1):
                print(f"[{i}/{len(pending)}] {msg} (elapsed {time.time() - t0:.0f}s)",
                      flush=True)
        print(f"finished in {time.time() - t0:.0f}s")
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
