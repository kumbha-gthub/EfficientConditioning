#!/usr/bin/env python3
"""Recompute the numbers of Section 6 and Appendices C and D from the
results of run_all.py.

Usage:  python analysis/make_tables.py [path/to/results]

Experiments that save per-seed results (results/data/*.json) are recomputed
from those files: the noise comparison (exp22), the SDD tables of exp19,
exp20 and exp23, and the duplicate counts of exp21.  The other experiments
(exp5, exp7, exp13, exp15, exp16, exp17, exp17b, exp18) write their results
only to the logs; for these the script prints the relevant log lines.

Tests are two-sided Wilcoxon signed-rank tests on paired seeds (SciPy
defaults).  The batch-size study has its own analysis scripts
(analysis/exp24_analysis.py and analysis/exp24_paper_numbers.py).
"""
import json, os, re, sys
import numpy as np
from scipy.stats import wilcoxon

# Print safely on any Windows console code page.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def _find_results():
    """Return the results folder: the command-line argument if given, otherwise
    a folder named 'results' next to this script or one level above it."""
    if len(sys.argv) > 1:
        cands = [sys.argv[1]]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        cands = [os.path.join(here, "results"), os.path.join(here, "..", "results")]
    for c in cands:
        if os.path.isdir(os.path.join(c, "data")):
            return os.path.abspath(c)
    sys.exit("Could not find the results folder. Looked in:\n  " + "\n  ".join(os.path.abspath(c) for c in cands)
             + "\nRun the script with the path of the results folder as argument.")


R = _find_results()
D, L = os.path.join(R, "data"), os.path.join(R, "logs")


def jl(name):
    with open(os.path.join(D, name), encoding="utf-8") as f:
        return json.load(f)


def log(name):
    with open(os.path.join(L, name), encoding="utf-8") as f:
        return f.read()


def w(a, b):
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


def ms(x, nd=3):
    return f"{np.mean(x):.{nd}f} \\pm {np.std(x):.{nd}f}"


out = []
P = out.append

# ------------------------------------------------------------------ noise (exp22)
d = jl("exp22_noisy_hartmann.json")
P("== Noisy Hartmann (exp22): true f at incumbent")
T = {k: np.array([r["true_at_incumbent"] for r in v]) for k, v in d.items()}
for k, v in T.items():
    dv = [r["div"] for r in d[k] if r["div"] is not None]
    P(f"   {k:10s} {ms(v)}  diversity {np.mean(dv) if dv else float('nan'):.3f}  noise_sd(mean) {np.mean([r['noise_sd'] for r in d[k]]):.4f}")
for a, b in [("cl_min", "random"), ("kb", "random"), ("cl_min", "kb"), ("cl_min", "sequential"), ("kb", "sequential")]:
    P(f"   {a} vs {b}: p={w(T[a],T[b]):.2e} first lower {int((T[a]<T[b]).sum())}/{len(T[a])}")

names = {"126025": "adult", "167200": "HIGGS", "168331": "volkert", "189908": "Fashion-MNIST"}

# ------------------------------------------------------------------ SDD tables from JSON
P("== SDD exp19 part B (5 seeds, q=3)")
d = jl("exp19_mqrbf_power_function.json")
for k, v in d["B"].items():
    thr = 1e-3 if k.startswith("Hartmann") else 1e-2
    x = np.array(v, float)
    P(f"   {k:45s} diverse {int((x>thr).sum())}/5  mean min {x.mean():.3f}  mins {np.round(x,4).tolist()}")
P("== exp19 part A (Hartmann, 5 seeds)")
for k, v in d["A"].items():
    P(f"   {k:18s} corr {np.mean([r['corr'] for r in v]):+.3f} ratio_data {np.mean([r['ratio_data'] for r in v]):.3f} profile {np.round(np.mean([r['ratio_r'] for r in v],0),3).tolist()}")
P("== exp19 part C (Levy-10D, seed 42)")
for k, v in d["C"].items():
    P(f"   {k:32s} min {v['min']:.3f} mean {v['mean']:.3f}")
P("== exp19 part D (per-step conditioning time, s)")
for k, v in d["D"].items():
    P(f"   {k:18s} {v}")

d = jl("exp20_sdd_start_sensitivity.json")
P("== exp20 (SDD protocol sensitivity)")
for k, v in d.items():
    if "lengthscale" in k:
        P(f"   {k}: " + "; ".join(f"{m}: median l/width {r['median_rel']:.3f}, all-at-init {r['seeds_all_at_init']}/5, dims at upper {100*r['frac_dims_at_upper']:.0f}%" for m, r in v.items()))
    else:
        side = 1.0 if k.startswith("Hartmann") else (10.0 if k.startswith("Ackley") else 20.0)
        x = np.array(v["mins"], float)
        P(f"   {k:45s} diverse {int((x>1e-3*side).sum())}/5 mean min {x.mean():.3f} unmoved {v['unmoved']}/{v['n_members']} EI=0 {v['ei_zero']}")

P("== LCBench SDD (exp23 part A)")
for t, nm in names.items():
    d = jl(f"exp23_lcbench_hpo_A_{t}.json")["A"]
    for k, v in d.items():
        x = np.array(v, float)
        P(f"   {nm:14s} {k:40s} diverse {int((x>1e-3).sum())}/5 mean min {x.mean():.3f}")

# ------------------------------------------------------------------ exp21
d = jl("exp21_acquisition_monotonicity.json")
P("== exp21 (Hartmann, 5 seeds x 30 iterations)")
for k, v in d.items():
    rec = v["records"]; md = np.array([r["min_dist"] for r in rec])
    ex = [r for r in rec if r["regime"] == "exploit"]; ep = [r for r in rec if r["regime"] != "exploit"]
    dex = sum(r["min_dist"] < 1e-3 for r in ex); dep = sum(r["min_dist"] < 1e-3 for r in ep)
    P(f"   {k:16s} dup {int((md<1e-3).sum())}/{len(md)} ({100*v['dup_rate']:.1f}%)  exploit {dex}/{len(ex)}  explore {dep}/{len(ep)}  p5 {np.percentile(md,5):.2e}  median {np.median(md):.3f}")

# ------------------------------------------------------------------ log-only experiments (echo key lines)
P("== Log-only experiments (verbatim key lines)")
for name, pats in [("exp5_full.txt", [r"^\S.*\(condition\)|\(re-solve\)|\(retrain\)|\(rebuild\)"]),
                   ("exp13_full.txt", [r"Duplicate rate|Min pairwise dist|Smallest observed|mean over seeds|separation ratio"]),
                   ("exp15_full.txt", [r"^(GP|MQ-RBF|NN Ensemble|Random Forest)\s+\d+\s+\d+"]),
                   ("exp16_full.txt", [r"l=|In-regime"]),
                   ("exp17_full.txt", [r"SDD min=|sigma_after|per-seed SDD"]),
                   ("exp17b_full.txt", [r"diverse in"]),
                   ("exp18_full.txt", [r"fixed-hyper"]),
                   ("exp7_full.txt", [r"^(GP|MQ-RBF|NN|RF) \(.*\)\s+\d"])]:
    txt = log(name)
    for line in txt.splitlines():
        if any(re.search(p, line.strip()) for p in pats):
            P(f"   [{name}] {line.strip()}")

print("\n".join(out))
