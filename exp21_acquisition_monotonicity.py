#!/usr/bin/env python3
"""Duplicates along BO trajectories for combinations of acquisition function,
lie and incumbent rule.

Hartmann-6D, 5 seeds, 30 iterations, q = 3.  The trajectory is advanced with
EI and CL-min batches.  At every iteration, from the same fitted GP and with
the same optimizer restarts, a batch is built for each combination of
  acquisition  EI, UCB (beta = 2), PI
  lie          CL-min, KB
  incumbent    'upd'    f+ = min(observations and lies so far)
               'fixed'  f+ = min(observations)
and the script records whether the batch contains a pair closer than 1e-3
and whether its first point is exploitative (mu(x1) < f+ - xi).

Use EXP_QUICK=1 for 2 seeds and 5 iterations.
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    hartmann6, GPSurrogate, optimize_acquisition, latin_hypercube,
    pairwise_distances, QUICK, save_json,
)

DIM = 6
BOUNDS = np.array([[0.0, 1.0]] * DIM)
SEEDS = [7, 21, 42, 73, 99]
N_ITERS = 30
Q = 3
N_RESTARTS = 5
XI = 0.01
DUP = 1e-3
if QUICK:
    SEEDS, N_ITERS = SEEDS[:2], 5

COMBOS = [("ei", "cl_min", "upd"), ("ei", "kb", "upd"), ("ei", "kb", "fixed"),
          ("ucb", "cl_min", "-"), ("ucb", "kb", "-"),
          ("pi", "cl_min", "upd"), ("pi", "kb", "upd"), ("pi", "kb", "fixed")]


def build_batch(gp, X, y, acq, lie, inc, rng_seed):
    """Greedy pseudo-observation batch with fixed hyper-parameters."""
    np.random.seed(rng_seed)            # identical restarts for every combo
    f_obs = float(np.min(y))
    f_best = f_obs
    X_aug, y_aug = X.copy(), y.copy()
    batch, regime = [], None
    for j in range(Q):
        gj = GPSurrogate()
        gj.kernel_ = gp.kernel_
        gj.fit(X_aug, y_aug, optimize=False)
        x_new, _ = optimize_acquisition(gj, BOUNDS, f_best,
                                        n_restarts=N_RESTARTS, acq_func=acq)
        if j == 0:
            mu1, _ = gj.predict(x_new.reshape(1, -1))
            regime = "exploit" if mu1[0] < f_obs - XI else "explore"
        batch.append(x_new)
        yl = f_obs if lie == "cl_min" else float(gj.predict(x_new.reshape(1, -1))[0][0])
        X_aug = np.vstack([X_aug, x_new.reshape(1, -1)])
        y_aug = np.append(y_aug, yl)
        if inc == "upd":
            f_best = min(f_best, yl)
    return np.array(batch), regime


def main():
    t0 = time.time()
    print("=" * 84)
    print("EXPERIMENT 21 -- duplicates by acquisition / lie / incumbent")
    print(f"  Hartmann-6D, seeds={SEEDS}, {N_ITERS} BO iterations, q={Q}")
    print("=" * 84)
    rec = {c: [] for c in COMBOS}
    for seed in SEEDS:
        np.random.seed(seed)
        X = latin_hypercube(2 * DIM, DIM, BOUNDS, seed=seed)
        y = hartmann6(X)
        for it in range(N_ITERS):
            gp = GPSurrogate(); gp.fit(X, y, optimize=True)
            rs = 100000 * seed + it
            for c in COMBOS:
                b, reg = build_batch(gp, X, y, *c, rng_seed=rs)
                rec[c].append(dict(seed=seed, it=it, regime=reg,
                                   min_dist=float(pairwise_distances(b).min())))
            # advance the trajectory with EI + CL-min
            b, _ = build_batch(gp, X, y, "ei", "cl_min", "upd", rng_seed=rs)
            X = np.vstack([X, b]); y = np.append(y, hartmann6(b))
        print(f"  seed {seed:>3} done  (best so far {np.min(y):.4f})")

    print(f"\n  {'acq':<4} {'lie':<7} {'f+':<6} {'dup rate':>9} "
          f"{'dup | exploit':>15} {'dup | explore':>15} {'5th pct min dist':>17}")
    print("  " + "-" * 78)
    summary = {}
    for c in COMBOS:
        r = rec[c]
        md = np.array([e["min_dist"] for e in r])
        dup = md < DUP
        ex = np.array([e["regime"] == "exploit" for e in r])
        d_ex = f"{dup[ex].mean():.1%} ({ex.sum()})" if ex.any() else "-"
        d_er = f"{dup[~ex].mean():.1%} ({(~ex).sum()})" if (~ex).any() else "-"
        print(f"  {c[0]:<4} {c[1]:<7} {c[2]:<6} {dup.mean():>9.1%} "
              f"{d_ex:>15} {d_er:>15} {np.percentile(md, 5):>17.2e}")
        summary["|".join(c)] = dict(dup_rate=float(dup.mean()),
                                    records=r)
    save_json("exp21_acquisition_monotonicity", summary)
    print("\n  (n) = number of batches whose first point was in that regime.")
    print("  Hypothesis under test: EI and UCB never duplicate; PI can duplicate")
    print("  under KB in the exploitation regime when f+ is held fixed.")
    print(f"\nTotal wall-clock: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
