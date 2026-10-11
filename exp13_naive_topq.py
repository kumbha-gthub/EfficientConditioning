#!/usr/bin/env python3
"""Naive top-q selection compared with CL-min.

Part A (Hartmann-6D, 5 seeds, 50 iterations of sequential BO with EI,
n_init = 12): at every iteration, from the same fitted GP, build
  (i)  the naive batch: L-BFGS-B on EI from 20 random starts, keeping the
       3 results with the largest EI;
  (ii) the CL-min batch of 3 (fixed hyperparameters);
and record the minimum pairwise distance of each.  A batch with a pair
closer than 1e-3 counts as containing a duplicate.

Part B (Levy-2D on [-10, 10]^2, 5 seeds, n_init = 10): the 3 best of
100,000 Latin-hypercube candidates by EI, against the CL-min batch.

Use --quick for 2 seeds and 20 iterations.
"""

import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    hartmann6, levy, GPSurrogate, optimize_acquisition, latin_hypercube,
    pairwise_distances, expected_improvement, FIG_DIR,
)

# -- configuration ----------------------------------------------------
Q              = 3
SEEDS          = [7, 21, 42, 73, 99]
BUDGET_A       = 50                      # sequential BO evals, Part A
N_RESTARTS_TOPQ = 20                     # naive strategy restarts
N_RESTARTS_BO  = 10                      # BO acquisition optimiser restarts
DUP_THRESHOLD  = 1e-3                    # same threshold as the SDD
N_CANDIDATES_B = 100_000                 # Part B dense set

DIM_A    = 6
BOUNDS_A = np.array([[0.0, 1.0]] * DIM_A)
DIM_B    = 2
BOUNDS_B = np.array([[-10.0, 10.0]] * DIM_B)

# ---------------------------------------------------------------------


def naive_topq(gp, f_best, q, rng, bounds):
    """Top-q converged points (by EI value) from pre-drawn random restarts."""
    pts, vals = [], []
    for _ in range(N_RESTARTS_TOPQ):
        s = rng.uniform(bounds[:, 0], bounds[:, 1])
        x, ei = optimize_acquisition(gp, bounds, f_best, fixed_start=s)
        if x is not None:
            pts.append(x)
            vals.append(ei)
    order = np.argsort(vals)[::-1][:q]
    return np.array([pts[i] for i in order])


def clmin_batch_from(X, y, kernel, q, bounds):
    """CL-min batch of q from the current data, fixed hyperparameters."""
    gp = GPSurrogate(bounds=bounds)
    gp.kernel_ = kernel
    gp.fit(X.copy(), y.copy(), optimize=False)
    X_aug, y_aug = X.copy(), y.copy()
    batch = []
    for j in range(q):
        gp_j = GPSurrogate(bounds=bounds)
        gp_j.kernel_ = kernel
        gp_j.fit(X_aug, y_aug, optimize=False)
        fj = float(np.min(y_aug))
        x_new, _ = optimize_acquisition(gp_j, bounds, fj,
                                        n_restarts=N_RESTARTS_BO)
        batch.append(x_new)
        yl = float(np.min(y))                       # CL-min
        X_aug = np.vstack([X_aug, x_new.reshape(1, -1)])
        y_aug = np.append(y_aug, yl)
    return np.array(batch)


# ================================================================
#  PART A - trajectory study on Hartmann-6D
# ================================================================

def part_a(seeds, budget):
    print("\n" + "=" * 78)
    print(f"PART A - Naive top-{Q} vs CL-min along sequential BO trajectories")
    print(f"  Hartmann-6D, seeds={seeds}, budget={budget}, "
          f"{N_RESTARTS_TOPQ} restarts for naive top-q")
    print("=" * 78)

    naive_traj, cl_traj = [], []          # per-seed lists of min-dists

    for seed in seeds:
        rng = np.random.RandomState(seed + 2000)
        # pin the global-RNG restarts used by optimize_acquisition
        # (CL-min batches + trajectory advance), for exact reproducibility
        np.random.seed(seed)
        X = latin_hypercube(2 * DIM_A, DIM_A, BOUNDS_A, seed=seed)
        y = hartmann6(X)

        nd, cd = [], []
        for it in range(budget):
            gp = GPSurrogate(bounds=BOUNDS_A)
            gp.fit(X, y, optimize=True)
            f_best = float(np.min(y))

            # (i) naive top-q from the current model
            b_naive = naive_topq(gp, f_best, Q, rng, BOUNDS_A)
            nd.append(float(np.min(pairwise_distances(b_naive))))

            # (ii) CL-min batch from the same model state
            b_cl = clmin_batch_from(X, y, gp.kernel_, Q, BOUNDS_A)
            cd.append(float(np.min(pairwise_distances(b_cl))))

            # advance the trajectory by one sequential EI step
            x_new, _ = optimize_acquisition(gp, BOUNDS_A, f_best,
                                            n_restarts=N_RESTARTS_BO)
            X = np.vstack([X, x_new.reshape(1, -1)])
            y = np.append(y, hartmann6(x_new.reshape(1, -1)))

        naive_traj.append(nd)
        cl_traj.append(cd)

        n_dup_naive = sum(1 for d in nd if d < DUP_THRESHOLD)
        n_dup_cl = sum(1 for d in cd if d < DUP_THRESHOLD)
        print(f"  seed {seed:>3}: duplicate iterations "
              f"(min dist < {DUP_THRESHOLD:g}) - "
              f"naive: {n_dup_naive}/{budget}   CL-min: {n_dup_cl}/{budget}   "
              f"| median min-dist - naive: {np.median(nd):.4f}  "
              f"CL-min: {np.median(cd):.4f}")

    naive_all = np.array(naive_traj)      # (n_seeds, budget)
    cl_all = np.array(cl_traj)

    dup_naive = float(np.mean(naive_all < DUP_THRESHOLD))
    dup_cl = float(np.mean(cl_all < DUP_THRESHOLD))

    print("\n" + "-" * 78)
    print(f"  AGGREGATE over {len(seeds)} seeds x {budget} iterations "
          f"= {naive_all.size} batch constructions per strategy")
    print("-" * 78)
    print(f"  {'':30} {'Naive top-q':>15} {'CL-min (ours)':>15}")
    print(f"  {'Duplicate rate (<1e-3)':<30} {100*dup_naive:>14.1f}% "
          f"{100*dup_cl:>14.1f}%")
    print(f"  {'Min pairwise dist, median':<30} "
          f"{np.median(naive_all):>15.4f} {np.median(cl_all):>15.4f}")
    print(f"  {'Min pairwise dist, 5th pct':<30} "
          f"{np.percentile(naive_all, 5):>15.6f} "
          f"{np.percentile(cl_all, 5):>15.6f}")
    print(f"  {'Smallest observed min dist':<30} "
          f"{naive_all.min():>15.2e} {cl_all.min():>15.2e}")

    # -- figure -------------------------------------------------
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.2, 2.6), layout="constrained")
    iters = np.arange(1, naive_all.shape[1] + 1)
    for i in range(len(seeds)):
        ax.plot(iters, np.maximum(naive_all[i], 1e-7), color="#E53935",
                alpha=0.30, lw=0.8)
        ax.plot(iters, np.maximum(cl_all[i], 1e-7), color="#1E88E5",
                alpha=0.30, lw=0.8)
    ax.plot(iters, np.maximum(np.median(naive_all, axis=0), 1e-7),
            color="#E53935", lw=1.8, label="Naive top-$q$ (median)")
    ax.plot(iters, np.maximum(np.median(cl_all, axis=0), 1e-7),
            color="#1E88E5", lw=1.8, label="CL-min (median)")
    ax.axhline(DUP_THRESHOLD, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("Min. pairwise distance")
    ax.grid(True, alpha=0.25, which="both")
    fig.legend(loc="outside upper center", ncol=2, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"exp13_naive_topq.{ext}"), dpi=300)
    plt.close(fig)
    print(f"\n  Figure saved -> {os.path.join(FIG_DIR, 'exp13_naive_topq.pdf')}")

    return naive_all, cl_all, dup_naive, dup_cl


# ================================================================
#  PART B - dense-candidate study in 2D
# ================================================================

def part_b(seeds):
    print("\n" + "=" * 78)
    print(f"PART B - Dense-candidate top-{Q} in 2D "
          f"({N_CANDIDATES_B:,} candidates, resolution ~"
          f"{(1/N_CANDIDATES_B)**(1/DIM_B):.4f} of domain width)")
    print("  Levy-2D, n_init=10")
    print("=" * 78)

    width = float(BOUNDS_B[0, 1] - BOUNDS_B[0, 0])
    rows = []
    for seed in seeds:
        # pin the global-RNG restarts used by the CL-min arm
        np.random.seed(seed)
        X = latin_hypercube(10, DIM_B, BOUNDS_B, seed=seed)
        y = levy(X)
        gp = GPSurrogate(bounds=BOUNDS_B)
        gp.fit(X, y, optimize=True)
        f_best = float(np.min(y))

        cand = latin_hypercube(N_CANDIDATES_B, DIM_B, BOUNDS_B,
                               seed=seed + 1)
        mus, sgs = [], []
        for i in range(0, len(cand), 20_000):
            mu, sg = gp.predict(cand[i:i + 20_000])
            mus.append(mu)
            sgs.append(sg)
        ei = expected_improvement(np.concatenate(mus),
                                  np.concatenate(sgs), f_best)
        idx = np.argsort(ei)[::-1][:Q]
        d_naive = pairwise_distances(cand[idx])

        b_cl = clmin_batch_from(X, y, gp.kernel_, Q, BOUNDS_B)
        d_cl = pairwise_distances(b_cl)

        rows.append((seed, np.min(d_naive), np.min(d_cl)))
        print(f"  seed {seed:>3}: naive top-{Q} min dist = "
              f"{np.min(d_naive):>8.4f} ({np.min(d_naive)/width:.5f} of "
              f"domain width)   |   CL-min = {np.min(d_cl):>8.4f} "
              f"({np.min(d_cl)/width:.4f})")

    naive_mins = np.array([r[1] for r in rows])
    cl_mins = np.array([r[2] for r in rows])
    print(f"\n  mean over seeds - naive: {naive_mins.mean():.4f} "
          f"({naive_mins.mean()/width:.5f} of width)   "
          f"CL-min: {cl_mins.mean():.4f} ({cl_mins.mean()/width:.4f})")
    print(f"  separation ratio (CL-min / naive): "
          f"{cl_mins.mean()/max(naive_mins.mean(), 1e-12):.0f}x")
    return rows


# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="2 seeds, 20 evals - smoke test")
    ap.add_argument("--part-b-only", action="store_true",
                    help="run only Part B (Levy-2D)")
    args = ap.parse_args()
    if args.part_b_only:
        part_b(SEEDS[:2] if args.quick else SEEDS)
        return

    seeds = SEEDS[:2] if args.quick else SEEDS
    budget = 20 if args.quick else BUDGET_A

    print("=" * 78)
    print("EXPERIMENT 13 - Naive top-q selection vs CL-min")
    print("=" * 78)

    naive_all, cl_all, dup_naive, dup_cl = part_a(seeds, budget)
    part_b(seeds)

    # ---------------- LaTeX ----------------
    print("\n" + "=" * 78)
    print("LATEX")
    print("=" * 78)
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r" & \textbf{Naive top-$q$} & \textbf{CL-min} \\")
    print(r"\midrule")
    print(f"  Duplicate rate (min dist $<10^{{-3}}$) & "
          f"{100*dup_naive:.1f}\\% & {100*dup_cl:.1f}\\% \\\\")
    print(f"  Median min pairwise distance & "
          f"{np.median(naive_all):.3f} & {np.median(cl_all):.3f} \\\\")
    print(f"  5th-percentile min distance & "
          f"{np.percentile(naive_all, 5):.1e} & "
          f"{np.percentile(cl_all, 5):.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\nSummary:")
    print(f"  * Along {naive_all.size} batch constructions, naive top-{Q} "
          f"contained near-exact duplicates in {100*dup_naive:.1f}% of "
          f"iterations (min dist down to {naive_all.min():.1e});")
    print(f"    CL-min: {100*dup_cl:.1f}%.")
    print(f"  * In 2D, where a {N_CANDIDATES_B//1000}k-candidate set is "
          f"genuinely dense, the top-{Q} by EI collapse to grid resolution "
          f"around the argmax.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal wall-clock: {(time.time()-t0)/60:.1f} min")