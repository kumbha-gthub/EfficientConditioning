#!/usr/bin/env python3
"""Time per batch step of each surrogate.

Builds a batch of q = 3 points from n = 50 training points in d = 6 and times
each step: GP conditioning with fixed hyperparameters, MQ-RBF re-solve,
NN-ensemble retraining and random-forest rebuilding.  Reports the mean and
standard deviation over 20 repetitions.
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    hartmann6, GPSurrogate, NNEnsembleSurrogate, RFSurrogate,
    MQRBFSurrogate, latin_hypercube,
)

from utils import QUICK

DIM = 6
BOUNDS = np.array([[0.0, 1.0]] * DIM)
N_TRAIN = 50
Q = 3
REPS = 20
SEED = 42


def time_conditioning(surrogate_cls, X_train, y_train, name, retrain=False):
    """Time Q sequential condition_on (or retrain) calls."""
    times = []
    for rep in range(REPS):
        surr = surrogate_cls()
        surr.fit(X_train.copy(), y_train.copy())

        x_fake = np.random.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(Q, DIM))
        y_fake = np.random.randn(Q)

        t0 = time.perf_counter()
        for j in range(Q):
            if retrain:
                X_aug = np.vstack([surr.X_train, x_fake[j:j+1]])
                y_aug = np.append(surr.y_train, y_fake[j])
                surr.fit(X_aug, y_aug, retrain=True)
            else:
                surr.condition_on(x_fake[j], y_fake[j])
        elapsed = time.perf_counter() - t0
        times.append(elapsed / Q)

    return np.mean(times), np.std(times)


def main():
    np.random.seed(SEED)
    X_train = latin_hypercube(N_TRAIN, DIM, BOUNDS, seed=SEED)
    y_train = hartmann6(X_train)

    print("=" * 65)
    print("EXPERIMENT 5 - Wall-Clock Timing (per batch step)")
    print(f"  n_train={N_TRAIN}, d={DIM}, q={Q}, reps={REPS}")
    print("=" * 65)

    results = []

    # GP conditioning (fixed hyperparams)
    mean_t, std_t = time_conditioning(GPSurrogate, X_train, y_train, "GP")
    results.append(("GP (condition)", mean_t, std_t))

    # MQ-RBF re-solve
    mean_t, std_t = time_conditioning(MQRBFSurrogate, X_train, y_train, "MQ-RBF")
    results.append(("MQ-RBF (re-solve)", mean_t, std_t))

    # NN Ensemble retrain
    mean_t, std_t = time_conditioning(
        NNEnsembleSurrogate, X_train, y_train, "NN Ensemble", retrain=True)
    results.append(("NN Ensemble (retrain)", mean_t, std_t))

    # RF rebuild
    mean_t, std_t = time_conditioning(
        RFSurrogate, X_train, y_train, "RF", retrain=True)
    results.append(("RF (rebuild)", mean_t, std_t))

    print(f"\n{'Surrogate':<25} {'Mean (s)':>12} {'Std (s)':>12} {'Speedup':>10}")
    print("-" * 62)
    nn_time = results[2][1]
    for name, m, s in results:
        speedup = nn_time / m if m > 0 else float('inf')
        print(f"{name:<25} {m:>12.5f} {s:>12.5f} {speedup:>9.1f}x")

    print("\n" + "=" * 65)
    print("LATEX TABLE:")
    print("=" * 65)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Wall-clock time per batch step ($n{=}50$, $d{=}6$, $q{=}3$).}")
    print(r"\label{tab:timing}")
    print(r"\small")
    print(r"\begin{tabular}{lrr}")
    print(r"\toprule")
    print(r"\textbf{Surrogate} & \textbf{Time/step (s)} & \textbf{Speedup} \\")
    print(r"\midrule")
    for name, m, s in results:
        speedup = nn_time / m
        print(f"  {name} & ${m:.4f} \\pm {s:.4f}$ & ${speedup:.0f}\\times$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if QUICK:   # smoke-test mode (EXP_QUICK=1)
    REPS = 3


if __name__ == "__main__":
    main()
