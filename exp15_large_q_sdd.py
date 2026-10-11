#!/usr/bin/env python3
"""SDD at large batch sizes on Levy-10D.

CL-min batches of q = 10, 20 and 50 for the GP and MQ-RBF, and of q = 10 and
20 for the NN ensemble and the random forest (n_init = 30, seed 42, the 20
fixed SDD starts).  The two non-responsive models return the same point at
every step, so they are not run at q = 50.  Reports the minimum and mean
pairwise distance and the minimum distance of every member to the earlier
ones.

Use --quick for the GP and MQ-RBF at q = 10 and 20.
"""

import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import sdd_starts, dup_threshold, SDD_STARTS
from functools import partial
from utils import (
    levy, GPSurrogate, NNEnsembleSurrogate, RFSurrogate, MQRBFSurrogate,
    optimize_acquisition, latin_hypercube, pairwise_distances, FIG_DIR,
)

# -- configuration -----------------------------------------------------
DIM      = 10
BOUNDS   = np.array([[-10.0, 10.0]] * DIM)
N_INIT   = 30
SEED     = 42
FIXED_STARTS = sdd_starts(BOUNDS)        # 20 fixed Latin-hypercube starts
DUP_THRESHOLD = dup_threshold(BOUNDS)    # 1e-3 x side length

Q_CONDITIONING = [10, 20, 50]     # GP, MQ-RBF
Q_DEGENERATE   = [10, 20]         # NN, RF

# ---------------------------------------------------------------------


def sdd_batch(surrogate, y_train, q):
    """CL-min batch of q with the fixed SDD starts."""
    f_best = float(np.min(y_train))
    batch, per_member_min = [], []
    for j in range(q):
        best_x, best_ei = None, -np.inf
        for s in FIXED_STARTS:
            x, ei = optimize_acquisition(surrogate, BOUNDS, f_best,
                                         fixed_start=np.array(s, float))
            if ei > best_ei:
                best_ei, best_x = ei, x.copy()
        batch.append(best_x.copy())
        if len(batch) > 1:
            d_to_prev = [np.linalg.norm(best_x - b) for b in batch[:-1]]
            per_member_min.append(float(np.min(d_to_prev)))
        y_lie = float(np.min(y_train))                  # CL-min
        surrogate.condition_on(best_x, y_lie)
        f_best = min(f_best, y_lie)
    return np.array(batch), per_member_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="GP/MQ-RBF only, q in {10,20}")
    args = ap.parse_args()

    if args.quick:
        plan = [("GP", partial(GPSurrogate, bounds=BOUNDS), [10, 20]),
                ("MQ-RBF", MQRBFSurrogate, [10, 20])]
    else:
        plan = [("GP", partial(GPSurrogate, bounds=BOUNDS), Q_CONDITIONING),
                ("MQ-RBF", MQRBFSurrogate, Q_CONDITIONING),
                ("NN Ensemble", NNEnsembleSurrogate, Q_DEGENERATE),
                ("Random Forest", RFSurrogate, Q_DEGENERATE)]

    print("=" * 78)
    print("EXPERIMENT 15 - SDD at large batch sizes (Levy-10D)")
    print(f"  n_init={N_INIT}, seed={SEED}, {len(FIXED_STARTS)} fixed starts "
          f"({SDD_STARTS}), CL-min, unit-cube GP, power-function MQ-RBF")
    print(f"  duplicate threshold = {DUP_THRESHOLD:g} (1e-3 x domain side)")
    print("=" * 78)

    np.random.seed(SEED)
    X_init = latin_hypercube(N_INIT, DIM, BOUNDS, seed=SEED)
    y_init = levy(X_init)

    rows = []                     # (name, q, min_d, mean_d, n_pairs, time)
    per_member = {}               # (name, q) -> list

    for name, Cls, q_list in plan:
        for q in q_list:
            print(f"\n  {name:<14} q={q:<3} ...", end="", flush=True)
            surr = Cls()
            surr.fit(X_init.copy(), y_init.copy())
            t0 = time.perf_counter()
            batch, pm = sdd_batch(surr, y_init, q)
            elapsed = time.perf_counter() - t0
            d = pairwise_distances(batch)
            min_d, mean_d = float(np.min(d)), float(np.mean(d))
            rows.append((name, q, min_d, mean_d, len(d), elapsed))
            per_member[(name, q)] = pm
            tag = "DIVERSE" if min_d > DUP_THRESHOLD else "DEGENERATE"
            print(f" min={min_d:.4f}  mean={mean_d:.4f}  "
                  f"({len(d)} pairs)  {tag}  [{elapsed:.1f}s]")

    # -- summary ------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY - SDD at large q (Levy-10D)")
    print("=" * 78)
    hdr = (f"{'Surrogate':<16} {'q':>4} {'Pairs':>6} {'Min Dist':>10} "
           f"{'Mean Dist':>10} {'Diverse?':>9} {'Time(s)':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, q, mn, me, npairs, tt in rows:
        tag = "YES" if mn > DUP_THRESHOLD else "NO"
        print(f"{name:<16} {q:>4} {npairs:>6} {mn:>10.4f} {me:>10.4f} "
              f"{tag:>9} {tt:>9.1f}")


    # -- figure: per-member min distance ------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"GP": "#2196F3", "MQ-RBF": "#FF9800"}
    styles = {10: ":", 20: "--", 50: "-"}
    for (name, q), pm in per_member.items():
        if name not in colors or not pm:
            continue
        ax.plot(range(2, q + 1), pm, styles.get(q, "-"),
                color=colors[name], lw=2, marker="o", markersize=3,
                label=f"{name}  q={q}")
    ax.axhline(DUP_THRESHOLD, color="red", ls="--", lw=1, alpha=0.6,
               label="Duplicate threshold (1e-3)")
    ax.set_yscale("log")
    ax.set_xlabel("Batch member index", fontsize=11)
    ax.set_ylabel("Min distance to all previous members (log)", fontsize=11)
    ax.set_title("Per-member diversity at large batch sizes - Levy-10D "
                 "(SDD, CL-min)", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25, which="both")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "exp15_large_q_sdd.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved -> {out}")

    # -- LaTeX --------------------------------------------------
    print("\n" + "=" * 78)
    print("LATEX")
    print("=" * 78)
    print(r"\begin{tabular}{lrrrc}")
    print(r"\toprule")
    print(r"\textbf{Surrogate} & $q$ & \textbf{Min Dist} & "
          r"\textbf{Mean Dist} & \textbf{Diverse?} \\")
    print(r"\midrule")
    for name, q, mn, me, npairs, tt in rows:
        tag = "Yes" if mn > DUP_THRESHOLD else "No"
        print(f"  {name} & {q} & {mn:.3f} & {me:.3f} & {tag} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    gp50 = [r for r in rows if r[0] == "GP" and r[1] == 50]
    if gp50:
        ok = gp50[0][2] > DUP_THRESHOLD
        print(f"\nGP at q=50: minimum over all {gp50[0][4]} pairwise distances = "
              f"{gp50[0][2]:.3f} ({'above' if ok else 'BELOW'} the duplicate "
              f"threshold {DUP_THRESHOLD:g}).")
    print("Note: the batches are built greedily from the same state, so the")
    print("q=10 batch is the first 10 members of the q=20 batch, and so on;")
    print("the minimum distance can therefore only stay equal or shrink with q.")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal wall-clock: {(time.time()-t0)/60:.1f} min")