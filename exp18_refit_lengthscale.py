#!/usr/bin/env python3
"""SDD with all GP hyperparameters refitted after every pseudo-observation.

Compares batches built with fixed hyperparameters (the default) with batches
for which all hyperparameters are re-estimated by marginal likelihood after
every pseudo-observation, and reports the mean relative change of the
fitted lengthscales per step.  Hartmann-6D and Ackley-8D, 5 seeds, KB and
CL-min.  Use --quick for Hartmann-6D, 2 seeds, KB only.
"""

import os, sys, time, argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import sdd_starts, dup_threshold
from utils import (
    hartmann6, ackley, GPSurrogate, optimize_acquisition,
    latin_hypercube, pairwise_distances, get_gp_lengthscale,
)

SEEDS = [7, 21, 42, 73, 99]
Q = 3
DUP = 1e-3


def sdd_batch(X_init, y_init, bounds, d, strategy, refit):
    """SDD batch of q.  With refit=True, all hyperparameters are
    re-optimized after every pseudo-observation; otherwise they stay fixed.
    Returns (batch, list of fitted lengthscale norms per step)."""
    gp = GPSurrogate(bounds=bounds)   # inputs scaled to the unit cube
    gp.fit(X_init.copy(), y_init.copy(), optimize=True)   # initial ML-II fit
    ell_hist = [float(np.mean(get_gp_lengthscale(gp)))]

    f_best = float(np.min(y_init))
    starts = sdd_starts(bounds)
    batch = []
    Xa, ya = X_init.copy(), y_init.copy()

    for _ in range(Q):
        bx, be = None, -np.inf
        for s in starts:
            x, ei = optimize_acquisition(gp, bounds, f_best,
                                         fixed_start=np.array(s, float))
            if ei > be:
                be, bx = ei, x.copy()
        batch.append(bx.copy())

        if strategy == "kb":
            y_lie = float(gp.predict(bx.reshape(1, -1))[0][0])
        else:                                             # cl-min
            y_lie = float(np.min(y_init))

        Xa = np.vstack([Xa, bx.reshape(1, -1)])
        ya = np.append(ya, y_lie)
        if refit:
            gp = GPSurrogate(bounds=bounds)
            gp.fit(Xa, ya, optimize=True)                 # re-estimate the hyperparameters
            ell_hist.append(float(np.mean(get_gp_lengthscale(gp))))
        else:
            gp.condition_on(bx, y_lie)                    # fixed hypers
        f_best = min(f_best, y_lie)

    return np.array(batch), ell_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = SEEDS[:2] if args.quick else SEEDS
    strategies = ["kb"] if args.quick else ["kb", "cl"]
    setups = ([dict(name="Hartmann-6D", func=hartmann6, d=6,
                    bounds=np.array([[0.0, 1.0]] * 6), n_init=30)]
              if args.quick else
              [dict(name="Hartmann-6D", func=hartmann6, d=6,
                    bounds=np.array([[0.0, 1.0]] * 6), n_init=30),
               dict(name="Ackley-8D", func=ackley, d=8,
                    bounds=np.array([[-5.0, 5.0]] * 8), n_init=30)])

    print("=" * 78)
    print("EXPERIMENT 18 - Hyperparameters refitted after every pseudo-observation")
    print(f"  seeds={seeds}, q={Q}")
    print("=" * 78)

    for setup in setups:
        print(f"\n{'-'*70}\n  {setup['name']}\n{'-'*70}")
        dup = dup_threshold(setup["bounds"])
        for strat in strategies:
            fixed_mins, refit_mins, drifts = [], [], []
            for seed in seeds:
                X = latin_hypercube(setup["n_init"], setup["d"],
                                    setup["bounds"], seed=seed)
                y = setup["func"](X)

                bf, _ = sdd_batch(X, y, setup["bounds"], setup["d"],
                                  strat, refit=False)
                fixed_mins.append(float(np.min(pairwise_distances(bf))))

                br, ell_hist = sdd_batch(X, y, setup["bounds"], setup["d"],
                                         strat, refit=True)
                refit_mins.append(float(np.min(pairwise_distances(br))))

                # mean relative lengthscale drift per conditioning step
                eh = np.array(ell_hist)
                rel = np.abs(np.diff(eh)) / np.maximum(eh[:-1], 1e-12)
                drifts.append(float(np.mean(rel)))

            fd = "YES" if np.mean(fixed_mins) > dup else "NO"
            rd = "YES" if np.mean(refit_mins) > dup else "NO"
            print(f"  {strat.upper():<7} "
                  f"fixed-hyper SDD min={np.mean(fixed_mins):.4f} ({fd})  |  "
                  f"refit-every-step SDD min={np.mean(refit_mins):.4f} ({rd})  |  "
                  f"mean ell drift/step={100*np.mean(drifts):.1f}%")
            print(f"  {'':<7} per-seed fixed: "
                  + "  ".join(f"{v:.4f}" for v in fixed_mins)
                  + f"   [{sum(v > dup for v in fixed_mins)}/{len(seeds)} diverse]")
            print(f"  {'':<7} per-seed refit: "
                  + "  ".join(f"{v:.4f}" for v in refit_mins)
                  + f"   [{sum(v > dup for v in refit_mins)}/{len(seeds)} diverse]")
            print(f"  {'':<7} per-seed drift/step: "
                  + "  ".join(f"{100*v:.1f}%" for v in drifts))

    print("\n" + "=" * 78)
    print("How to read: 'fixed-hyper' conditions with the hyper-parameters of")
    print("the initial fit (as every other script does); 'refit' re-fits all")
    print("hyper-parameters, including the output standardization, after every")
    print("pseudo-observation; 'drift/step' is the mean relative change of the")
    print("lengthscales per step.")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal wall-clock: {(time.time()-t0)/60:.1f} min")
