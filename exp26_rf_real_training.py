#!/usr/bin/env python3
"""Batch BO on a tuning task with real training runs.

Every evaluation trains a scikit-learn random forest on mfeat-karhunen
(OpenML dataset 16, the dataset of RandomForest-16) and returns the
validation error in percent on a fixed stratified 70/30 split.  The four
hyperparameters and ranges are those of rbv2_ranger (Table 3):
  num.trees        [1, 2000], integer   -> n_estimators
  sample.fraction  [0.1, 1]             -> max_samples (bootstrap)
  mtry.power       [0, 1]               -> max_features = round(64 ** u)
  min.node.size    [1, 100], integer    -> min_samples_leaf
The forest uses random_state = 0, so the objective is deterministic.

The GP, EI and the batch rules follow Appendices B and B.1:
  GP     ConstantKernel(1.0, [1e-3, 1e3]) x Matern-5/2 with ARD, lengthscales
         in [0.005, 2] with initial value 0.5 on [0,1]^4, alpha = 1e-6,
         5 restarts, random_state = 0, output standardization stored at
         the fit; hyperparameters refitted once per batch.
  CL/KB  fitted kernel and stored standardization while the batch is
         built; CL lies use the observed values; the incumbent includes
         the lies.
  EI     xi = 0.01; sigma floored at 1e-10, EI = 0 at the floor; L-BFGS-B
         from the best 5 of 1000 uniform points, objective divided by the
         best random value.
  TS     1000 uniform points and 1000 Gaussian perturbations (sd 0.1) of the
         five best observed inputs, clipped; duplicates and evaluated inputs
         removed; joint samples on the standardized scale (Cholesky, jitter
         1e-8, increased tenfold if needed); no candidate chosen twice.
Runs: 2d = 8 Latin-hypercube points shared by all methods of a seed, then
100 evaluations (for q = 16 the last batch has 4 points); seeds 0-9.
The data are downloaded from OpenML by scikit-learn on the first run.

Usage
  python exp26_rf_real_training.py validate        checks of the components
  python exp26_rf_real_training.py run [workers]   all 70 runs (default 7 workers)
  python exp26_rf_real_training.py summarize       table and Wilcoxon tests
Finished runs are skipped, so an interrupted run can be restarted.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
from scipy import optimize, stats
from scipy.linalg import cholesky
from scipy.stats import qmc
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Some BLAS builds warn inside matmul; non-finite results are rejected below.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*encountered in matmul.*")

DIM = 4
N_INIT = 2 * DIM
BUDGET = 100
XI = 0.01
ALPHA = 1e-6
SIGMA_FLOOR = 1e-10
DUP_THRESHOLD = 1e-3
SEEDS = list(range(10))
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "data", "exp26_rf_real_training")

# (method, q); random search is stored with q = 0
CONFIGS = [
    ("random", 0),
    ("seq_ei", 1),
    ("cl_min", 4), ("kb", 4), ("ts", 4),
    ("cl_min", 16), ("kb", 16),
]

_DATA = {}


def load_data():
    """OpenML dataset 16 (mfeat-karhunen) and a fixed stratified split."""
    if "Xtr" in _DATA:
        return
    d = fetch_openml(data_id=16, as_frame=False, parser="auto")
    Xtr, Xva, ytr, yva = train_test_split(
        d.data, d.target, test_size=0.3, stratify=d.target, random_state=0)
    _DATA.update(Xtr=Xtr, Xva=Xva, ytr=ytr, yva=yva, n_features=d.data.shape[1])


def decode(u):
    """Map u in [0,1]^4 to the hyperparameters of the forest."""
    u = np.clip(np.asarray(u, dtype=float).ravel(), 0.0, 1.0)
    p = _DATA["n_features"]
    n_estimators = int(round(1 + u[0] * 1999))
    sample_fraction = 0.1 + 0.9 * u[1]
    mtry = int(np.clip(round(p ** u[2]), 1, p))
    min_leaf = int(round(1 + u[3] * 99))
    return {
        "n_estimators": n_estimators,
        "max_samples": None if sample_fraction >= 0.9999 else sample_fraction,
        "max_features": mtry,
        "min_samples_leaf": min_leaf,
    }


def evaluate(u, memo):
    """Train the forest for u and return the validation error in percent."""
    key = tuple(np.round(np.asarray(u, dtype=float).ravel(), 12))
    if key in memo:
        return memo[key]
    params = decode(u)
    rf = RandomForestClassifier(bootstrap=True, random_state=0, n_jobs=1, **params)
    rf.fit(_DATA["Xtr"], _DATA["ytr"])
    err = 100.0 * (1.0 - rf.score(_DATA["Xva"], _DATA["yva"]))
    assert np.isfinite(err), f"non-finite objective at {params}"
    memo[key] = err
    return err


# ------------------------------------------------------------------ GP
class PaperGP:
    """GP of Appendix B on inputs in [0,1]^4."""

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.mean = float(np.mean(y))
        self.std = float(np.std(y))
        if self.std < 1e-12:
            self.std = 1.0
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=[0.5] * DIM, length_scale_bounds=(0.005, 2.0), nu=2.5)
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
        clone = PaperGP()
        clone.mean, clone.std, clone.gpr, clone.kernel_ = (
            self.mean, self.std, gpr, self.kernel_)
        return clone

    def predict(self, X):
        mu_s, sd_s = self.gpr.predict(np.atleast_2d(X), return_std=True)
        return mu_s * self.std + self.mean, sd_s * self.std

    def predict_std_scale(self, X, return_cov=False):
        return self.gpr.predict(np.atleast_2d(X), return_cov=return_cov)


def expected_improvement(mu, sd, incumbent):
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    active = sd < SIGMA_FLOOR
    s = np.maximum(sd, SIGMA_FLOOR)
    delta = incumbent - XI - mu
    z = delta / s
    ei = delta * stats.norm.cdf(z) + s * stats.norm.pdf(z)
    ei = np.where(active, 0.0, np.maximum(ei, 0.0))
    return ei


def maximize_ei(model, incumbent, rng):
    cands = rng.uniform(size=(1000, DIM))
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
                                bounds=[(0.0, 1.0)] * DIM)
        v = -res.fun * scale
        if np.isfinite(v) and np.all(np.isfinite(res.x)) and v > best_v:
            best_x, best_v = np.clip(res.x, 0.0, 1.0), v
    assert np.all(np.isfinite(best_x)), "non-finite acquisition maximizer"
    return best_x, best_v


# ------------------------------------------------------------------ batches
def select_batch_cl_kb(model, X_obs, y_obs, q, lie_rule, rng):
    X_cond = list(np.asarray(X_obs))
    y_cond = list(np.asarray(y_obs, dtype=float))
    cl_lie = float(np.min(y_obs))
    batch = []
    current = model
    for j in range(q):
        incumbent = float(np.min(y_cond))
        xj, _ = maximize_ei(current, incumbent, rng)
        if lie_rule == "cl_min":
            lie = cl_lie
        else:
            lie = float(current.predict(xj.reshape(1, -1))[0][0])
        batch.append(xj)
        X_cond.append(xj)
        y_cond.append(lie)
        if j < q - 1:
            current = model.condition(X_cond, y_cond)
    return np.array(batch)


def select_batch_ts(model, X_obs, y_obs, q, rng):
    X_obs = np.asarray(X_obs)
    y_obs = np.asarray(y_obs, dtype=float)
    five_best = X_obs[np.argsort(y_obs)[:5]]
    perturbed = np.repeat(five_best, 200, axis=0) + rng.normal(
        0.0, 0.1, size=(1000, DIM))
    cands = np.vstack([rng.uniform(size=(1000, DIM)),
                       np.clip(perturbed, 0.0, 1.0)])
    cands = np.unique(cands, axis=0)
    obs_keys = {tuple(row) for row in np.round(X_obs, 15)}
    keep = [i for i, row in enumerate(np.round(cands, 15))
            if tuple(row) not in obs_keys]
    cands = cands[keep]
    mu_s, cov_s = model.predict_std_scale(cands, return_cov=True)
    jitter = 1e-8
    while True:
        try:
            L = cholesky(cov_s + jitter * np.eye(len(cands)), lower=True)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    batch, chosen = [], set()
    for _ in range(q):
        sample = mu_s + L @ rng.standard_normal(len(cands))
        for idx in np.argsort(sample):
            if idx not in chosen:
                chosen.add(idx)
                batch.append(cands[idx])
                break
    return np.array(batch)


def batch_schedule(q):
    if q <= 1:
        return [1] * BUDGET
    sizes = [q] * (BUDGET // q)
    if BUDGET % q:
        sizes.append(BUDGET % q)
    return sizes


def min_pairwise_distance(batch):
    if len(batch) < 2:
        return None
    d = np.linalg.norm(batch[:, None, :] - batch[None, :, :], axis=-1)
    return float(d[np.triu_indices(len(batch), k=1)].min())


# ------------------------------------------------------------------ one run
def run_one(method, q, seed):
    load_data()
    t0 = time.time()
    memo = {}
    init = qmc.LatinHypercube(d=DIM, seed=seed).random(N_INIT)
    X_obs = [row for row in init]
    y_obs = [evaluate(row, memo) for row in init]
    best_traj = [float(np.min(y_obs))]
    batch_dists = []
    rng = np.random.default_rng([seed, {"random": 0, "seq_ei": 1, "cl_min": 2,
                                        "kb": 3, "ts": 4}[method], q])
    if method == "random":
        for _ in range(BUDGET):
            x = rng.uniform(size=DIM)
            y_obs.append(evaluate(x, memo))
            X_obs.append(x)
            best_traj.append(float(np.min(y_obs)))
    else:
        for bsize in batch_schedule(q):
            model = PaperGP().fit(np.array(X_obs), np.array(y_obs))
            if method == "seq_ei":
                x, _ = maximize_ei(model, float(np.min(y_obs)), rng)
                batch = np.array([x])
            elif method in ("cl_min", "kb"):
                batch = select_batch_cl_kb(model, X_obs, y_obs, bsize, method, rng)
            elif method == "ts":
                batch = select_batch_ts(model, X_obs, y_obs, bsize, rng)
            d = min_pairwise_distance(batch)
            if d is not None:
                batch_dists.append(d)
            for x in batch:
                y_obs.append(evaluate(x, memo))
                X_obs.append(x)
                best_traj.append(float(np.min(y_obs)))
    return {
        "method": method, "q": q, "seed": seed,
        "final_best": float(np.min(y_obs)),
        "best_after_init": best_traj[0],
        "best_traj": best_traj,
        "batch_min_dists": batch_dists,
        "n_evals_total": len(y_obs),
        "wall_time_s": round(time.time() - t0, 1),
    }


def run_path(method, q, seed):
    return os.path.join(RESULTS_DIR, f"{method}_q{q}_seed{seed}.json")


def worker(task):
    method, q, seed = task
    path = run_path(method, q, seed)
    if os.path.exists(path):
        return f"skip {method} q={q} seed={seed}"
    res = run_one(method, q, seed)
    with open(path, "w") as f:
        json.dump(res, f)
    return (f"done {method} q={q} seed={seed}: best {res['final_best']:.2f}% "
            f"({res['wall_time_s']:.0f}s)")


# ------------------------------------------------------------------ checks
def validate():
    load_data()
    u = np.array([0.37, 0.81, 0.44, 0.12])
    e1, e2 = evaluate(u, {}), evaluate(u, {})
    assert e1 == e2, (e1, e2)
    print(f"1. objective deterministic ({e1:.3f}% twice)  ok")
    assert batch_schedule(4) == [4] * 25
    assert batch_schedule(16) == [16] * 6 + [4]
    print("2. batches 25 x 4 and 6 x 16 + 4  ok")
    a = qmc.LatinHypercube(d=DIM, seed=3).random(N_INIT)
    b = qmc.LatinHypercube(d=DIM, seed=3).random(N_INIT)
    assert np.array_equal(a, b)
    print("3. initial design reproducible  ok")
    rng = np.random.default_rng(0)
    memo = {}
    X = rng.uniform(size=(20, DIM))
    y = np.array([evaluate(x, memo) for x in X])
    gp = PaperGP().fit(X, y)
    xstar = rng.uniform(size=DIM)
    lie = float(np.min(y))
    cond = gp.condition(list(X) + [xstar], list(y) + [lie])
    mu, sd = cond.predict(xstar.reshape(1, -1))
    assert abs(mu[0] - lie) < 0.05 * gp.std and sd[0] < 0.05 * gp.std
    print(f"4. conditioning reproduces the lie (|mu - lie| = {abs(mu[0] - lie):.1e}, "
          f"sigma = {sd[0]:.1e})  ok")
    pts = rng.uniform(size=(50, DIM))
    mu_before, _ = gp.predict(pts)
    kb_lie = float(gp.predict(xstar.reshape(1, -1))[0][0])
    mu_after, _ = gp.condition(list(X) + [xstar], list(y) + [kb_lie]).predict(pts)
    dmax = float(np.max(np.abs(mu_after - mu_before)))
    assert dmax < 1e-5 * gp.std, dmax
    print(f"5. KB leaves the mean unchanged (max change {dmax:.1e})  ok")
    x, v = maximize_ei(gp, float(np.min(y)), np.random.default_rng(1))
    assert np.all(x >= 0) and np.all(x <= 1) and v >= 0
    print("6. EI maximizer within bounds  ok")
    for rule, seed in (("cl_min", 2), ("kb", 2)):
        batch = select_batch_cl_kb(gp, X, y, 4, rule, np.random.default_rng(seed))
        d = min_pairwise_distance(batch)
        assert d > DUP_THRESHOLD, d
        print(f"7. {rule} batch of 4 without duplicates (min distance {d:.3f})  ok")
    batch = select_batch_ts(gp, X, y, 4, np.random.default_rng(3))
    obs_keys = {tuple(row) for row in np.round(X, 15)}
    assert len(batch) == 4
    assert all(tuple(np.round(b, 15)) not in obs_keys for b in batch)
    print("8. TS batch of 4 avoids evaluated inputs  ok")


# ------------------------------------------------------------------ summary
def summarize():
    rows = {}
    for method, q in CONFIGS:
        res = [json.load(open(run_path(method, q, s)))
               for s in SEEDS if os.path.exists(run_path(method, q, s))]
        if res:
            rows[(method, q)] = sorted(res, key=lambda r: r["seed"])
    print(f"{sum(len(v) for v in rows.values())} runs\n")
    print(f"{'method':<8} {'q':>3} {'n':>3} {'error % (mean +- sd)':>22} "
          f"{'batches':>8} {'min dist':>9} {'< 1e-3':>7}")
    for (method, q), res in rows.items():
        finals = np.array([r["final_best"] for r in res])
        dists = [d for r in res for d in r["batch_min_dists"]]
        extra = (f"{len(dists):>8} {min(dists):>9.4f} "
                 f"{sum(d < DUP_THRESHOLD for d in dists):>7}") if dists else ""
        print(f"{method:<8} {q:>3} {len(res):>3} {finals.mean():>12.3f} +- "
              f"{finals.std():<6.3f} {extra}")

    def finals(key):
        return np.array([r["final_best"] for r in rows[key]])

    print("\nPaired Wilcoxon tests (two-sided) over seeds:")
    pairs = [(b, ("seq_ei", 1)) for b in CONFIGS[2:]]
    pairs += [(b, ("random", 0)) for b in CONFIGS[2:]]
    pairs += [(("cl_min", 4), ("kb", 4)), (("cl_min", 16), ("kb", 16))]
    for a, b in pairs:
        if a not in rows or b not in rows:
            continue
        fa, fb = finals(a), finals(b)
        n = min(len(fa), len(fb))
        ties = int(np.sum(fa[:n] == fb[:n]))
        if ties == n:
            print(f"  {a[0]} q={a[1]} vs {b[0]} q={b[1]}: identical")
            continue
        p = stats.wilcoxon(fa[:n], fb[:n]).pvalue
        print(f"  {a[0]} q={a[1]} vs {b[0]} q={b[1]}: mean {fa[:n].mean():.3f} vs "
              f"{fb[:n].mean():.3f}, p = {p:.4f} ({n} seeds, {ties} tied)")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "validate":
        validate()
    elif cmd == "summarize":
        summarize()
    elif cmd == "run":
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        os.makedirs(RESULTS_DIR, exist_ok=True)
        load_data()
        tasks = [(m, q, s) for (m, q) in CONFIGS for s in SEEDS]
        pending = [t for t in tasks if not os.path.exists(run_path(*t))]
        print(f"{len(pending)} of {len(tasks)} runs to do")
        import multiprocessing as mp
        t0 = time.time()
        with mp.get_context("spawn").Pool(workers, initializer=load_data) as pool:
            for i, msg in enumerate(pool.imap_unordered(worker, pending), 1):
                print(f"[{i}/{len(pending)}] {msg} (elapsed {time.time() - t0:.0f}s)",
                      flush=True)
        print(f"finished in {time.time() - t0:.0f}s")
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
