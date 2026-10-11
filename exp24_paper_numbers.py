#!/usr/bin/env python3
"""
Numbers of the batch-size study reported in the paper, recomputed from the
summary file written by exp24_analysis.py.

Usage:  python analysis/exp24_paper_numbers.py [path/to/exp24_summary.csv]

Input: results/data/exp24_summary.csv (one row per problem, method and q, with
the mean normalized regret over the 30 seeds in column mean_nreg).

Prints
  * the mean normalized regret over the 19 problems, and separately over the
    synthetic and the tuning problems;
  * the across-problem Wilcoxon tests on the 19 per-problem means (two-sided,
    scipy defaults), with Holm's adjustment over the ten pairs of the five
    main methods at each q;
  * the comparisons of the lies and of every batch method with sequential EI;
  * the change of each method from its own q = 1 version to q = 16;
  * the per-problem statements (best method per problem, problems where
    random search beats a method, weakly separating problems).

The ranks, the bootstrap intervals and the figures need the per-run files
and come from exp24_analysis.py.
"""

import os
import sys
import csv
import numpy as np
from scipy.stats import wilcoxon

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_summary():
    if len(sys.argv) > 1:
        cands = [sys.argv[1]]
    else:
        cands = [os.path.join(HERE, "results", "data", "exp24_summary.csv"),
                 os.path.join(HERE, "..", "results", "data", "exp24_summary.csv")]
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    sys.exit("Could not find exp24_summary.csv. Looked in:\n  " +
             "\n  ".join(os.path.abspath(c) for c in cands))


rows = list(csv.DictReader(open(_find_summary(), encoding="utf-8")))
M = {(r["problem"], r["method"], int(r["q"])): float(r["mean_nreg"]) for r in rows}
problems = []
for r in rows:
    if r["problem"] not in problems:
        problems.append(r["problem"])
SYN = [p for p in problems if not p.startswith(("LCBench", "XGBoost", "RandomForest",
                                                 "ElasticNet", "DecisionTree"))]
HPO = [p for p in problems if p not in SYN]
QS = [2, 4, 8, 16]
MAIN = ["cl_min", "kb", "lp", "qlogei", "ts"]
ALL = ["cl_min", "kb", "cl_mean", "cl_max", "lp", "qlogei", "ts"]
Q1 = {"cl_min": "ei", "kb": "ei", "cl_mean": "ei", "cl_max": "ei", "lp": "ei",
      "qlogei": "qlogei", "ts": "ts"}
LABEL = {"cl_min": "CL-min", "kb": "KB", "cl_mean": "CL-mean", "cl_max": "CL-max",
         "lp": "Lipschitz LP", "qlogei": "q-LogEI", "ts": "Thompson sampling",
         "ei": "Sequential EI", "random": "Random search"}
n_runs = sum(int(r["n"]) for r in rows)
print(f"{len(problems)} problems ({len(SYN)} synthetic, {len(HPO)} tuning), "
      f"{len(rows)} rows, {n_runs} runs")


def val(m, q):
    """(method, q) as shown in the tables: q = 1 is the method's own sequential version."""
    return (Q1[m], 1) if q == 1 else (m, q)


def mean(m, q, ps=problems):
    return float(np.mean([M[(p,) + val(m, q)] for p in ps]))


def wil(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.all(a == b):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def holm(ps):
    ps = np.asarray(ps, float)
    adj, run = np.empty(len(ps)), 0.0
    for i, k in enumerate(np.argsort(ps)):
        run = max(run, (len(ps) - i) * ps[k])
        adj[k] = min(1.0, run)
    return adj


def per_problem(m, q):
    return [M[(p,) + val(m, q)] for p in problems]


# ------------------------------------------------------------------ tables
print("\n== Mean normalized regret after 100 evaluations (19 problems)")
for m in ALL:
    print(f"   {LABEL[m]:18s} " + "  ".join(f"{mean(m, q):.3f}" for q in [1] + QS))
print(f"   {LABEL['random']:18s} {np.mean([M[(p, 'random', 1)] for p in problems]):.3f}")

print("\n== Mean normalized regret, synthetic (6) | tuning (13)")
for m in ALL:
    print(f"   {LABEL[m]:18s} " + "  ".join(f"{mean(m, q, SYN):.3f}" for q in [1] + QS) +
          "  |  " + "  ".join(f"{mean(m, q, HPO):.3f}" for q in [1] + QS))
print(f"   {LABEL['random']:18s} {np.mean([M[(p, 'random', 1)] for p in SYN]):.3f}  |  "
      f"{np.mean([M[(p, 'random', 1)] for p in HPO]):.3f}")

# ------------------------------------------------------------------ tests
pairs = [(a, b) for i, a in enumerate(MAIN) for b in MAIN[i + 1:]]
print("\n== Five main methods: Wilcoxon on the 19 per-problem means; Holm over the ten pairs per q")
for q in QS:
    ps = [wil(per_problem(a, q), per_problem(b, q)) for a, b in pairs]
    for (a, b), p, h in zip(pairs, ps, holm(ps)):
        lower = sum(x < y for x, y in zip(per_problem(a, q), per_problem(b, q)))
        print(f"   q={q:2d} {LABEL[a]:13s} vs {LABEL[b]:18s} p={p:.4g}  Holm={h:.4g}  "
              f"means {mean(a, q):.4f} vs {mean(b, q):.4f}  first lower on {lower}/19")

print("\n== Lies (uncorrected)")
for q in QS:
    for a, b in [("cl_min", "kb"), ("cl_min", "cl_mean"), ("cl_min", "cl_max"),
                 ("kb", "cl_mean"), ("kb", "cl_max"), ("cl_mean", "cl_max")]:
        p = wil(per_problem(a, q), per_problem(b, q))
        print(f"   q={q:2d} {LABEL[a]:8s} vs {LABEL[b]:8s} p={p:.4g}  "
              f"means {mean(a, q):.4f} vs {mean(b, q):.4f}")

print("\n== Every method against sequential EI (uncorrected)")
ei = [M[(p, "ei", 1)] for p in problems]
for q in QS:
    for m in ALL:
        x = per_problem(m, q)
        print(f"   q={q:2d} {LABEL[m]:18s} p={wil(x, ei):.4g}  higher than EI on "
              f"{sum(a > b for a, b in zip(x, ei))}/19")
for m in ["qlogei", "ts"]:
    print(f"   q= 1 {LABEL[m]:18s} p={wil(per_problem(m, 1), ei):.4g}")

print("\n== Change from the method's own q = 1 version to q = 16 (mean over problems)")
for m in MAIN:
    print(f"   {LABEL[m]:18s} {mean(m, 16) - mean(m, 1):+.4f}")

# ------------------------------------------------------------------ per problem
print("\n== Lowest and highest mean regret per problem among the five main methods "
      "(q = 1: EI, q-LogEI, TS)")
for q in [1] + QS:
    meths = ["ei", "qlogei", "ts"] if q == 1 else MAIN
    best, worst = {}, {}
    for p in problems:
        v = {m: M[(p, m, q)] for m in meths}
        best.setdefault(min(v, key=v.get), []).append(p)
        worst.setdefault(max(v, key=v.get), []).append(p)
    print(f"   q={q:2d} lowest:  " + "; ".join(f"{LABEL[m]} {len(ps)} {ps}" for m, ps in best.items()))
    print("        highest: " + "; ".join(f"{LABEL[m]} {len(ps)}" for m, ps in worst.items()))

print("\n== Problems on which random search has a lower mean regret than a method")
for p in problems:
    rnd = M[(p, "random", 1)]
    worse = [f"{LABEL[m]} q={q}" for q in [1] + QS
             for m in (["ei", "qlogei", "ts"] if q == 1 else MAIN) if M[(p, m, q)] > rnd]
    if worse:
        print(f"   {p}: " + ", ".join(worse))

print("\n== Problems on which EI, CL-min, KB and TS stay below 0.1 at every q")
for p in problems:
    v = [M[(p, "ei", 1)], M[(p, "ts", 1)]] + [M[(p, m, q)] for q in QS for m in ["cl_min", "kb", "ts"]]
    if max(v) < 0.1:
        print(f"   {p}  (largest {max(v):.3f})")
