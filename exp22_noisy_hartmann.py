#!/usr/bin/env python3
"""Batch BO with noisy observations on Hartmann-6D.

q = 3, 20 seeds, 2d Latin-hypercube initial points and 50 evaluations.  Each
observation gets Gaussian noise with standard deviation 0.1 times that of
the initial function values.  The GP noise variance is set to the true noise
level (in the standardized units of each fit), and the kernel
hyperparameters are refitted once per batch.  Methods: CL-min, KB, random
batches and sequential EI.  Reports the true function value at the point
with the best noisy observation, and Wilcoxon tests.

Use EXP_QUICK=1 for 2 seeds and 6 evaluations.
"""

import os, sys, time
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    hartmann6, GPSurrogate, optimize_acquisition, latin_hypercube,
    pairwise_distances, QUICK, save_json,
)

DIM = 6
BOUNDS = np.array([[0.0, 1.0]] * DIM)
N_SEEDS = 20
BUDGET = 50
Q = 3
N_RESTARTS = 10
NOISE_FRAC = 0.1
if QUICK:
    N_SEEDS, BUDGET = 2, 6


def run(strategy, seed):
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    noise_rng = np.random.RandomState(10_000 + seed)
    n_init = 2 * DIM
    X = latin_hypercube(n_init, DIM, BOUNDS, seed=seed)
    f_true = hartmann6(X)
    noise_sd = NOISE_FRAC * float(np.std(f_true))
    y = f_true + noise_rng.randn(len(f_true)) * noise_sd
    divs = []
    q = 1 if strategy == "sequential" else Q
    n_done = 0
    while n_done < BUDGET:
        aq = min(q, BUDGET - n_done)
        alpha = (noise_sd / max(float(np.std(y)), 1e-12)) ** 2
        gp = GPSurrogate(noise=alpha); gp.fit(X, y, optimize=True)
        if strategy == "random":
            batch = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(aq, DIM))
        else:
            batch, X_aug, y_aug = [], X.copy(), y.copy()
            for j in range(aq):
                gj = GPSurrogate(noise=alpha); gj.kernel_ = gp.kernel_
                gj.fit(X_aug, y_aug, optimize=False)
                x_new, _ = optimize_acquisition(gj, BOUNDS, float(np.min(y_aug)),
                                                n_restarts=N_RESTARTS)
                batch.append(x_new)
                yl = (float(np.min(y)) if strategy == "cl_min"
                      else float(gj.predict(x_new.reshape(1, -1))[0][0]))
                X_aug = np.vstack([X_aug, x_new.reshape(1, -1)])
                y_aug = np.append(y_aug, yl)
            batch = np.array(batch)
        ft = hartmann6(batch)
        X = np.vstack([X, batch]); f_true = np.append(f_true, ft)
        y = np.append(y, ft + noise_rng.randn(len(ft)) * noise_sd)
        if len(batch) > 1:
            divs.append(float(np.mean(pairwise_distances(batch))))
        n_done += len(batch)
    inc = int(np.argmin(y))
    return dict(best_noisy=float(np.min(y)), true_at_incumbent=float(f_true[inc]),
                best_true=float(np.min(f_true)), div=float(np.mean(divs)) if divs else None,
                noise_sd=noise_sd)


def main():
    t0 = time.time()
    print("=" * 78)
    print("EXPERIMENT 22 -- Noisy Hartmann-6D")
    print(f"  q={Q}  seeds={N_SEEDS}  budget={BUDGET}  noise = {NOISE_FRAC} x std(f_init)")
    print("=" * 78)
    strategies = ["cl_min", "kb", "random", "sequential"]
    res = {}
    for s in strategies:
        print(f"  Running {s:<11}...", end="", flush=True)
        res[s] = [run(s, seed) for seed in range(N_SEEDS)]
        print(" done")
    print(f"\n  {'Strategy':<12} {'true f at incumbent':>22} {'best noisy y':>18} "
          f"{'best true f seen':>18} {'diversity':>10}")
    for s in strategies:
        a = np.array([r["true_at_incumbent"] for r in res[s]])
        b = np.array([r["best_noisy"] for r in res[s]])
        c = np.array([r["best_true"] for r in res[s]])
        dv = [r["div"] for r in res[s] if r["div"] is not None]
        dvs = f"{np.mean(dv):.3f}" if dv else "-"
        print(f"  {s:<12} {a.mean():>12.4f} +- {a.std():<6.4f} {b.mean():>10.4f} +- "
              f"{b.std():<5.3f} {c.mean():>10.4f} +- {c.std():<5.3f} {dvs:>10}")
    print("\n  Wilcoxon (true f at incumbent, paired seeds):")
    for a, b in [("cl_min", "random"), ("kb", "random"), ("cl_min", "kb"),
                 ("cl_min", "sequential"), ("kb", "sequential")]:
        fa = np.array([r["true_at_incumbent"] for r in res[a]])
        fb = np.array([r["true_at_incumbent"] for r in res[b]])
        try:
            p = wilcoxon(fa, fb).pvalue
        except ValueError:
            p = float("nan")
        print(f"    {a:<11} vs {b:<11} p = {p:.4f}")
    save_json("exp22_noisy_hartmann", res)
    print(f"\nTotal wall-clock: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
