#!/usr/bin/env python3
"""Refitted non-responsive models, and the SDD with UCB.

Hartmann-6D, q = 3, n_init = 30, seed 42, the 20 fixed SDD starts and the
CL-min lie.

Part A: the NN ensemble and the random forest with and without refitting
        after every pseudo-observation, next to GP conditioning and the
        MQ-RBF re-solve.
Part B: the SDD of the GP, MQ-RBF, NN ensemble and random forest with EI
        and with UCB.
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import sdd_starts
from utils import (
    hartmann6, GPSurrogate, NNEnsembleSurrogate, RFSurrogate,
    MQRBFSurrogate, optimize_acquisition, latin_hypercube,
    pairwise_distances, FIG_DIR,
)

DIM = 6
BOUNDS = np.array([[0.0, 1.0]] * DIM)
N_INIT = 30
Q = 3
SEED = 42
FIXED_STARTS = sdd_starts(BOUNDS)   # 20 fixed Latin-hypercube starts


def run_batch_generic(surrogate, y_train, q, fixed_starts, acq_func="ei",
                      retrain_each_step=False):
    """Run batch selection with optional full retraining at each step."""
    f_best = float(np.min(y_train))
    batch = []
    starts = [np.array(s, dtype=float) for s in fixed_starts]

    for j in range(q):
        best_x, best_acq = None, -np.inf
        for s in starts:
            x, acq_val = optimize_acquisition(
                surrogate, BOUNDS, f_best, fixed_start=s, acq_func=acq_func)
            if acq_val > best_acq:
                best_acq = acq_val
                best_x = x.copy()

        batch.append(best_x.copy())
        y_lie = float(np.min(y_train))  # CL-min

        if retrain_each_step:
            X_new = np.vstack([surrogate.X_train, best_x.reshape(1, -1)])
            y_new = np.append(surrogate.y_train, y_lie)
            surrogate.fit(X_new, y_new, retrain=True)
        else:
            surrogate.condition_on(best_x, y_lie)

        f_best = min(f_best, y_lie)

    return np.array(batch)


# ================================================================
#  PART A - Retrained NN/RF vs GP conditioning
# ================================================================

def part_a(X_init, y_init):
    print("\n" + "=" * 72)
    print("PART A - Retrained NN/RF vs GP Efficient Conditioning")
    print("=" * 72)

    configs = [
        ("GP (condition)",          GPSurrogate,        False),
        ("MQ-RBF (re-solve)",       MQRBFSurrogate,     False),
        ("NN (no retrain)",         NNEnsembleSurrogate, False),
        ("NN (retrained)",          NNEnsembleSurrogate, True),
        ("RF (no rebuild)",         RFSurrogate,         False),
        ("RF (rebuilt)",            RFSurrogate,         True),
    ]

    results = {}
    timings = {}
    for name, Cls, retrain in configs:
        print(f"\n  {name} ...", end="", flush=True)
        surr = Cls()
        surr.fit(X_init.copy(), y_init.copy())

        t0 = time.perf_counter()
        batch = run_batch_generic(surr, y_init, Q, FIXED_STARTS,
                                  retrain_each_step=retrain)
        elapsed = time.perf_counter() - t0

        dists = pairwise_distances(batch)
        results[name] = dict(
            batch=batch, dists=dists,
            min_d=float(np.min(dists)),
            mean_d=float(np.mean(dists)),
        )
        timings[name] = elapsed
        diverse = "YES" if np.min(dists) > 1e-3 else "NO"
        print(f" min_dist={np.min(dists):.6f}  diverse={diverse}  "
              f"time={elapsed:.3f}s")

    # summary table
    print(f"\n{'=' * 72}")
    print(f"{'Surrogate':<25} {'Min Dist':>10} {'Mean Dist':>10} "
          f"{'Diverse':>8} {'Time (s)':>10}")
    print("-" * 72)
    for name in results:
        r = results[name]
        tag = "YES" if r["min_d"] > 1e-3 else "NO"
        print(f"{name:<25} {r['min_d']:>10.6f} {r['mean_d']:>10.6f} "
              f"{tag:>8} {timings[name]:>10.3f}")

    # figure
    pal = {
        "GP (condition)": "#2196F3",
        "MQ-RBF (re-solve)": "#FF9800",
        "NN (no retrain)": "#FF5722",
        "NN (retrained)": "#E91E63",
        "RF (no rebuild)": "#4CAF50",
        "RF (rebuilt)": "#8BC34A",
    }

    fig, ax = plt.subplots(figsize=(11, 5))
    names = list(results.keys())
    x_pos = np.arange(len(names))
    w = 0.30
    ax.bar(x_pos - w/2,
           [results[n]["min_d"] for n in names], w,
           label="Min pairwise dist",
           color=[pal[n] for n in names], alpha=0.6,
           edgecolor="k", linewidth=0.5)
    ax.bar(x_pos + w/2,
           [results[n]["mean_d"] for n in names], w,
           label="Mean pairwise dist",
           color=[pal[n] for n in names], alpha=1.0,
           edgecolor="k", linewidth=0.5)
    ax.axhline(1e-3, color="red", ls="--", lw=1, alpha=0.6,
               label="Diversity threshold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Euclidean Distance")
    ax.set_title("Retrained Parametric Models vs Efficient Conditioning\n"
                 f"Hartmann-6D, q={Q}, CL-min, deterministic starts")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "exp7_retrained.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved -> {out}")
    return results, timings


# ================================================================
#  PART B - SDD with EI and with UCB
# ================================================================

def part_b(X_init, y_init):
    print("\n\n" + "=" * 72)
    print("PART B - Acquisition Function Agnosticism (EI vs UCB)")
    print("=" * 72)

    surrogates = [
        ("GP",            GPSurrogate),
        ("MQ-RBF",        MQRBFSurrogate),
        ("NN Ensemble",   NNEnsembleSurrogate),
        ("Random Forest", RFSurrogate),
    ]
    acq_funcs = ["ei", "ucb"]

    rows = []
    for name, Cls in surrogates:
        row = {"name": name}
        for af in acq_funcs:
            surr = Cls()
            surr.fit(X_init.copy(), y_init.copy())
            batch = run_batch_generic(surr, y_init, Q, FIXED_STARTS,
                                      acq_func=af)
            dists = pairwise_distances(batch)
            row[f"{af}_min"] = float(np.min(dists))
            row[f"{af}_mean"] = float(np.mean(dists))
        rows.append(row)

    # table
    print(f"\n{'Surrogate':<17} | {'EI':^25} | {'UCB':^25} |")
    print(f"{'':17} | {'min dist':>12} {'mean dist':>12} | "
          f"{'min dist':>12} {'mean dist':>12} |")
    print("-" * 75)
    for r in rows:
        ei_div = "Y" if r["ei_min"] > 1e-3 else "N"
        ucb_div = "Y" if r["ucb_min"] > 1e-3 else "N"
        print(f"{r['name']:<17} | {r['ei_min']:>12.6f} {r['ei_mean']:>12.6f} | "
              f"{r['ucb_min']:>12.6f} {r['ucb_mean']:>12.6f} |  "
              f"EI:{ei_div} UCB:{ucb_div}")

    # figure
    pal = {"GP": "#2196F3", "MQ-RBF": "#FF9800",
           "NN Ensemble": "#FF5722", "Random Forest": "#4CAF50"}
    names = [r["name"] for r in rows]
    x_pos = np.arange(len(names))
    w = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos - 1.5*w, [r["ei_min"] for r in rows], w,
           label="EI - min dist", color=[pal[n] for n in names],
           alpha=0.55, edgecolor="k", linewidth=0.5)
    ax.bar(x_pos - 0.5*w, [r["ei_mean"] for r in rows], w,
           label="EI - mean dist", color=[pal[n] for n in names],
           alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.bar(x_pos + 0.5*w, [r["ucb_min"] for r in rows], w,
           label="UCB - min dist", color=[pal[n] for n in names],
           alpha=0.55, edgecolor="k", linewidth=0.5, hatch="//")
    ax.bar(x_pos + 1.5*w, [r["ucb_mean"] for r in rows], w,
           label="UCB - mean dist", color=[pal[n] for n in names],
           alpha=0.85, edgecolor="k", linewidth=0.5, hatch="//")

    ax.axhline(1e-3, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Euclidean Distance")
    ax.set_title("Acquisition Function Agnosticism - SDD with EI vs UCB\n"
                 f"Hartmann-6D, q={Q}, CL-min, deterministic starts")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "exp7_acq_agnostic.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved -> {out}")
    return rows


# ================================================================

def main():
    np.random.seed(SEED)
    X_init = latin_hypercube(N_INIT, DIM, BOUNDS, seed=SEED)
    y_init = hartmann6(X_init)

    results_a, timings_a = part_a(X_init, y_init)
    part_b(X_init, y_init)

    print("\n" + "=" * 72)
    print("HOW TO READ")
    print("=" * 72)
    print("Part A: 'no retrain' / 'no rebuild' = pseudo-observation appended")
    print("  without refitting; 'retrained' / 'rebuilt' =")
    print("  full refit after every pseudo-observation. Compare the Diverse and")
    print("  Time columns.")
    print("Part B: the same SDD with UCB instead of EI.")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nDone in {time.time()-t0:.1f}s")
