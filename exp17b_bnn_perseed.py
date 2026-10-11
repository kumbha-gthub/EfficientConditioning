#!/usr/bin/env python3
"""Last-layer BNN in the SDD on 20 seeds, under CL-min and KB.

Hartmann-6D and Ackley-8D, n_init = 30, q = 3, 20 fixed SDD starts.  The MLP
feature map is seeded (random_state=0) and the acquisition is maximized from
the fixed starts only, so the script is deterministic.
"""

import os, sys, time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp17_lastlayer_bnn import LastLayerBNN
from utils import sdd_starts, dup_threshold
from utils import (
    hartmann6, ackley, optimize_acquisition, latin_hypercube,
    pairwise_distances,
)

SEEDS20 = [7, 21, 42, 73, 99, 1, 2, 3, 5, 11,
           13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
Q = 3
DUP = 1e-3

SETUPS = [
    dict(name="Hartmann-6D", func=hartmann6, d=6,
         bounds=np.array([[0.0, 1.0]] * 6), n_init=30),
    dict(name="Ackley-8D", func=ackley, d=8,
         bounds=np.array([[-5.0, 5.0]] * 8), n_init=30),
]


def sdd_batch_strategy(surr, y_train, q, bounds, d, strategy):
    """SDD batch as in exp17, with the lie as an argument."""
    starts = sdd_starts(bounds)
    f_best = float(np.min(y_train))
    batch = []
    for _ in range(q):
        bx, be = None, -np.inf
        for s in starts:
            x, ei = optimize_acquisition(surr, bounds, f_best,
                                         fixed_start=np.array(s, float))
            if ei > be:
                be, bx = ei, x.copy()
        batch.append(bx.copy())
        if strategy == "kb":
            y_lie = float(surr.predict(bx.reshape(1, -1))[0][0])
        else:                                             # cl-min
            y_lie = float(np.min(y_train))
        surr.condition_on(bx, y_lie)
        f_best = min(f_best, y_lie)
    return np.array(batch)


def main():
    print("=" * 78)
    print("EXPERIMENT 17b - Last-layer BNN, 20 seeds")
    print(f"  seeds={SEEDS20}")
    print(f"  q={Q}, strategies=CL-min and KB, duplicate threshold 1e-3 x domain side")
    print("=" * 78)

    for setup in SETUPS:
        print(f"\n{'-' * 70}\n  {setup['name']}\n{'-' * 70}")
        dup = dup_threshold(setup["bounds"])
        for strategy in ["cl", "kb"]:
            mins = []
            for seed in SEEDS20:
                X = latin_hypercube(setup["n_init"], setup["d"],
                                    setup["bounds"], seed=seed)
                y = setup["func"](X)
                m = LastLayerBNN()
                m.fit(X.copy(), y.copy())
                b = sdd_batch_strategy(m, y, Q, setup["bounds"],
                                       setup["d"], strategy)
                mins.append(float(np.min(pairwise_distances(b))))
            mins = np.array(mins)
            n_div = int(np.sum(mins > dup))
            print(f"  {strategy.upper():<7} diverse in {n_div}/20 seeds"
                  f"  | mean min={mins.mean():.4f}"
                  f"  | diverse-seed mins: "
                  f"{[float(round(v, 3)) for v in sorted(mins[mins > dup])]}")
            print("          per-seed mins: "
                  + "  ".join(f"{v:.4f}" for v in mins))

    print("\n" + "=" * 78)
    print("How to read: 'diverse in k/20' counts seeds whose minimum pairwise")
    print("distance exceeds 1e-3 x domain side.  A model with (C1) but without")
    print("local (C2) can leave the acquisition ordering unchanged, which shows")
    print("up as exact duplicates -- most visibly under KB, whose lie does not")
    print("move the mean.")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal wall-clock: {(time.time()-t0)/60:.1f} min")
