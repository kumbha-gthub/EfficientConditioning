#!/usr/bin/env python3
"""
Analysis of the batch-size study (exp24_batch_study.py).

Usage:  python analysis/exp24_analysis.py [path/to/results]

Reads results/data/exp24/<problem>/<method>_q<q>_s<seed>.json and writes
  results/data/exp24_reference.json   reference minima of the tuning problems
  results/data/exp24_summary.csv      per problem, method, q: mean, sem, median
  results/logs/exp24_numbers.txt      every number quoted in the paper
  results/figures/exp24_summary, exp24_lies, exp24_per_problem (.pdf and .png)

Definitions
  Normalized regret of a run (problem p, seed s):
      r = (f_best - f_ref) / (f_init(p, s) - f_ref),   clipped to [0, 1]
  f_best      best value after the evaluations considered;
  f_init(p,s) best value of the initial design of seed s (the initial design
              is identical for every method of a seed, so r stays paired);
  f_ref       known minimum (synthetic) or, for tuning problems, the lowest
              value found by any run or by a reference search of 200,000
              uniform random configurations (seed 12345).
  r = 1: no improvement on the initial design;  r = 0: best known value.

  Equal evaluations: r after the full budget (100 evaluations); as a check
  also after 96 evaluations, where every batch size has complete batches.
  Equal rounds:      r after R = 6 rounds, i.e. after min(6 q, 100) evaluations.
  Ranks:             within each (problem, seed, q) the five batch methods are
                     ranked by their final value (1 = best, ties share ranks).

Statistics (scipy defaults, two-sided, paired over seeds):
  * per problem and q: Wilcoxon signed-rank test over the 30 seeds, Holm
    correction over the 19 problems of a comparison; the direction of a
    significant result is the sign of the signed-rank statistic;
  * across problems: Wilcoxon signed-rank test on the 19 per-problem mean
    normalized regrets (not corrected over comparisons);
  * 95% confidence intervals: bootstrap over seeds within each problem
    (2,000 resamples, seed 0); the set of problems is held fixed.
Only (problem, seed) pairs for which every compared run exists are used, and
only runs written by the current code version (config.version).
"""

import os
import sys
import json
import glob
import numpy as np
from scipy.stats import wilcoxon, rankdata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))     # script in analysis/
sys.path.insert(0, HERE)                          # script next to exp24_problems.py

from exp24_problems import problem_names, get_problem, EXP24_VERSION   # noqa: E402

VERSION = EXP24_VERSION


def _find_results():
    if len(sys.argv) > 1:
        cands = [sys.argv[1]]
    else:
        cands = [os.path.join(HERE, "results"), os.path.join(HERE, "..", "results")]
    for c in cands:
        if os.path.isdir(os.path.join(c, "data", "exp24")):
            return os.path.abspath(c)
    sys.exit("Could not find results/data/exp24. Looked in:\n  " +
             "\n  ".join(os.path.abspath(c) for c in cands))


R = _find_results()
DATA, FIG, LOG = (os.path.join(R, s) for s in ("data", "figures", "logs"))
os.makedirs(FIG, exist_ok=True)
os.makedirs(LOG, exist_ok=True)

BATCH = ["cl_min", "kb", "lp", "qlogei", "ts"]            # main comparison
LIES = ["cl_min", "kb", "cl_mean", "cl_max"]              # lie comparison
Q1_OF = {"cl_min": "ei", "kb": "ei", "lp": "ei", "qlogei": "qlogei", "ts": "ts",
         "cl_mean": "ei", "cl_max": "ei"}
LABEL = {"cl_min": "CL-min", "kb": "KB", "lp": "Lipschitz LP", "qlogei": "q-LogEI",
         "ts": "Thompson sampling", "ei": "Sequential EI", "random": "Random search",
         "cl_mean": "CL-mean", "cl_max": "CL-max"}
# Colour-blind safe colours, plus a distinct marker per method.  CL-mean
# shares the colour of LP; the two never appear in the same figure.
COLOR = {"cl_min": "#2a78d6", "kb": "#eb6834", "lp": "#1baf7a", "qlogei": "#eda100",
         "ts": "#e87ba4", "cl_mean": "#1baf7a", "cl_max": "#4a3aa7"}
MARKER = {"cl_min": "o", "kb": "s", "lp": "^", "qlogei": "D", "ts": "v",
          "cl_mean": "P", "cl_max": "X"}
INK, MUTED = "#0b0b0b", "#52514e"
ROUNDS = 6
out = []
P = out.append


# ------------------------------------------------------------------ load
runs, skipped = {}, 0
for path in glob.glob(os.path.join(DATA, "exp24", "*", "*.json")):
    try:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("config", {}).get("version") != VERSION:
            skipped += 1
            continue
        runs[(r["problem"], r["method"], int(r["q"]), int(r["seed"]))] = r
    except Exception:                                           # noqa: BLE001
        skipped += 1
if not runs:
    sys.exit("No exp24 runs of the current version found.")
problems = [p for p in problem_names() if p in {k[0] for k in runs}]
seeds_all = sorted({k[3] for k in runs})
qs = sorted({k[2] for k in runs if k[1] in BATCH and k[2] > 1})
budget = max(r["budget"] for r in runs.values())
ALLB = BATCH + [m for m in LIES if m not in BATCH]
CONFIGS = [("ei", 1), ("qlogei", 1), ("ts", 1), ("random", 1)] + [(m, q) for q in qs for m in ALLB]


# ------------------------------------------------------------------ reference values
ref_path = os.path.join(DATA, "exp24_reference.json")
ref = {}
if os.path.exists(ref_path):
    with open(ref_path, encoding="utf-8") as f:
        ref = json.load(f)
    if ref.get("_version") != VERSION:
        ref = {}
for p in problems:
    prob = get_problem(p)
    if prob.f_star is not None:
        ref[p] = dict(f_ref=float(prob.f_star), source="known minimum")
        continue
    if "random_search_min" not in ref.get(p, {}):
        rs = np.random.default_rng(12345)
        mins = [float(np.min(prob(rs.uniform(size=(10000, prob.dim))))) for _ in range(20)]
        ref.setdefault(p, {})["random_search_min"] = min(mins)
    run_min = min(r["final"] for k, r in runs.items() if k[0] == p)
    ref[p]["best_run_min"] = float(run_min)
    ref[p]["f_ref"] = float(min(run_min, ref[p]["random_search_min"]))
    ref[p]["source"] = "min(best run, 200k random search)"
ref["_version"] = VERSION
with open(ref_path, "w", encoding="utf-8") as f:
    json.dump(ref, f, indent=1)


def nreg(r, n_eval=None):
    """Normalized regret of run r after n_eval evaluations (default: budget)."""
    fr = ref[r["problem"]]["f_ref"]
    v = r["best"][-1] if n_eval is None else r["best"][min(n_eval, len(r["best"]) - 1)]
    den = r["best"][0] - fr
    if den <= 0:
        return 0.0
    return float(np.clip((v - fr) / den, 0.0, 1.0))


# ------------------------------------------------------------------ helpers
def complete_seeds(p, keys):
    return [s for s in seeds_all if all((p, m, q, s) in runs for (m, q) in keys)]


def wil(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if len(a) < 2 or np.all(d == 0):
        return 1.0
    try:
        p = float(wilcoxon(a, b).pvalue)
        return p if np.isfinite(p) else 1.0
    except ValueError:
        return 1.0


def direction(a, b):
    """-1 if a tends to be lower (better) than b, +1 if higher, 0 if no
    difference: sign of the signed-rank statistic over non-zero differences."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    return float(np.sign(np.sum(np.sign(d) * rankdata(np.abs(d)))))


def holm(ps):
    ps = np.asarray([p if np.isfinite(p) else 1.0 for p in ps], float)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(np.argsort(ps)):
        run = max(run, (m - i) * ps[k])
        adj[k] = min(1.0, run)
    return adj


rng_boot = np.random.default_rng(0)


def boot(mat):
    """mat: list over problems of arrays over seeds.  Mean of per-problem
    means, with a 95% CI from resampling seeds within each problem."""
    est = float(np.mean([np.mean(v) for v in mat]))
    bs = [np.mean([np.mean(v[rng_boot.integers(0, len(v), len(v))]) for v in mat])
          for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi)


def table(n_eval_fn):
    """(method, q) -> (mean, lo, hi) of normalized regret over all problems."""
    res = {}
    for (m, q) in CONFIGS:
        mat = []
        for p in problems:
            ss = complete_seeds(p, [(m, q)])
            if ss:
                mat.append(np.array([nreg(runs[(p, m, q, s)], n_eval_fn(q)) for s in ss]))
        if len(mat) == len(problems):
            res[(m, q)] = boot(mat)
    return res


# ------------------------------------------------------------------ 1. overview
P(f"== Experiment 24 ({VERSION}): {len(problems)} problems, {len(seeds_all)} seeds, "
  f"q in {[1] + qs}, budget {budget}")
P(f"   runs loaded: {len(runs)}; files skipped (other version or unreadable): {skipped}")
nf = [(k, r["n_fail"]) for k, r in runs.items() if r.get("n_fail")]
P(f"   runs with fallback random points: {len(nf)}; fallback points in total: "
  f"{sum(n for _, n in nf)}")
for (m, q) in CONFIGS:
    k = [kk for kk, _ in nf if kk[1] == m and kk[2] == q]
    if k:
        P(f"      {LABEL[m]} q={q}: {len(k)} runs")
missing = [(p, m, q, s) for p in problems for (m, q) in CONFIGS for s in seeds_all
           if (p, m, q, s) not in runs]
P(f"   missing runs: {len(missing)}")

P("\n== Reference values f_ref and the median best initial value")
for p in problems:
    b0 = np.median([r["best"][0] for k, r in runs.items() if k[0] == p])
    P(f"   {p:22s} f_ref {ref[p]['f_ref']:.6g} ({ref[p]['source']}); median initial best {b0:.6g}")

rows = []
for p in problems:
    for (m, q) in CONFIGS:
        rr = [runs[(p, m, q, s)] for s in seeds_all if (p, m, q, s) in runs]
        if not rr:
            continue
        v = np.array([r["final"] for r in rr])
        nr = np.array([nreg(r) for r in rr])
        rows.append((p, m, q, len(v), v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0,
                     np.median(v), nr.mean(), np.median(nr)))
with open(os.path.join(DATA, "exp24_summary.csv"), "w", encoding="utf-8") as f:
    f.write("problem,method,q,n,mean,sem,median,mean_nreg,median_nreg\n")
    for r in rows:
        f.write(",".join([r[0], r[1], str(r[2]), str(r[3])] + [f"{x:.6g}" for x in r[4:]]) + "\n")

# ------------------------------------------------------------------ 2. equal evaluations
agg = table(lambda q: budget)
P(f"\n== Mean normalized regret after {budget} evaluations (mean over problems; 95% CI)")
for k, (e, lo, hi) in agg.items():
    P(f"   q={k[1]:2d} {LABEL[k[0]]:18s} {e:.4f}  [{lo:.4f}, {hi:.4f}]")
agg96 = table(lambda q: 96)
P("\n== Same after 96 evaluations (complete batches for every q)")
for k, (e, lo, hi) in agg96.items():
    P(f"   q={k[1]:2d} {LABEL[k[0]]:18s} {e:.4f}  [{lo:.4f}, {hi:.4f}]")

P("\n== Relative to the method's own q = 1 version (EI for CL-min, KB, LP), after "
  f"{budget} evaluations: mean over problems of (mean regret at q) - (mean regret at q = 1)")
for q in qs:
    for m in ALLB:
        diffs = []
        for p in problems:
            ss = complete_seeds(p, [(m, q), (Q1_OF[m], 1)])
            if ss:
                diffs.append(np.mean([nreg(runs[(p, m, q, s)]) for s in ss]) -
                             np.mean([nreg(runs[(p, Q1_OF[m], 1, s)]) for s in ss]))
        if len(diffs) == len(problems):
            P(f"   q={q:2d} {LABEL[m]:18s} {np.mean(diffs):+.4f}  "
              f"(worse than q=1 on {sum(d > 0 for d in diffs)} of {len(diffs)} problems)")

# ------------------------------------------------------------------ 3. ranks
P("\n== Average rank among the five batch methods (1 = best; 95% CI)")
rank_res = {}
for q in qs:
    keys = [(m, q) for m in BATCH]
    per = {m: [] for m in BATCH}
    used = 0
    for p in problems:
        ss = complete_seeds(p, keys)
        if not ss:
            continue
        used += 1
        Rk = np.array([rankdata([runs[(p, m, q, s)]["final"] for m in BATCH]) for s in ss])
        for i, m in enumerate(BATCH):
            per[m].append(Rk[:, i])
    for m in BATCH:
        if per[m]:
            rank_res[(m, q)] = boot(per[m])
            e, lo, hi = rank_res[(m, q)]
            P(f"   q={q:2d} {LABEL[m]:18s} {e:.2f}  [{lo:.2f}, {hi:.2f}]")
    P(f"   (q={q}: {used} of {len(problems)} problems complete)")


# ------------------------------------------------------------------ 3b. lies
P("\n== The four lies (CL-min, KB, CL-mean, CL-max): average rank among them (95% CI)")
lie_rank = {}
for q in qs:
    keys = [(m, q) for m in LIES]
    per = {m: [] for m in LIES}
    used = 0
    for p in problems:
        ss = complete_seeds(p, keys)
        if not ss:
            continue
        used += 1
        Rk = np.array([rankdata([runs[(p, m, q, s)]["final"] for m in LIES]) for s in ss])
        for i, m in enumerate(LIES):
            per[m].append(Rk[:, i])
    for m in LIES:
        if per[m]:
            lie_rank[(m, q)] = boot(per[m])
            e, lo, hi = lie_rank[(m, q)]
            P(f"   q={q:2d} {LABEL[m]:18s} {e:.2f}  [{lo:.2f}, {hi:.2f}]")
    P(f"   (q={q}: {used} of {len(problems)} problems complete)")


# ------------------------------------------------------------------ 4. pairwise tests
def compare(a, qa, b, qb, title):
    ps, dirs, ma, mb = [], [], [], []
    for p in problems:
        ss = complete_seeds(p, [(a, qa), (b, qb)])
        if len(ss) < 2:
            continue
        va = np.array([runs[(p, a, qa, s)]["final"] for s in ss])
        vb = np.array([runs[(p, b, qb, s)]["final"] for s in ss])
        ps.append(wil(va, vb))
        dirs.append(direction(va, vb))
        ma.append(np.mean([nreg(runs[(p, a, qa, s)]) for s in ss]))
        mb.append(np.mean([nreg(runs[(p, b, qb, s)]) for s in ss]))
    if not ps:
        return
    adj = holm(ps)
    better = int(np.sum((adj < 0.05) & (np.array(dirs) < 0)))
    worse = int(np.sum((adj < 0.05) & (np.array(dirs) > 0)))
    P(f"   {title:52s} better on {better:2d}, worse on {worse:2d} of {len(ps)} | across problems "
      f"p={wil(ma, mb):.3g} (mean regret {np.mean(ma):.4f} vs {np.mean(mb):.4f})")


P(f"\n== Pairwise comparisons after {budget} evaluations")
P("   per problem: Wilcoxon over seeds, Holm over problems; 'better/worse' = first method "
  "significantly better/worse (adjusted p < 0.05)")
pairs = [("cl_min", "kb"), ("cl_min", "lp"), ("cl_min", "qlogei"), ("cl_min", "ts"),
         ("kb", "lp"), ("kb", "qlogei"), ("kb", "ts"), ("lp", "qlogei"), ("lp", "ts"),
         ("qlogei", "ts")]
for q in qs:
    for a, b in pairs:
        compare(a, q, b, q, f"q={q:2d} {LABEL[a]} vs {LABEL[b]}")
P(f"\n== Lie comparisons after {budget} evaluations")
for q in qs:
    for a, b in [("cl_min", "cl_mean"), ("cl_min", "cl_max"), ("kb", "cl_mean"), ("kb", "cl_max"),
                 ("cl_mean", "cl_max")]:
        compare(a, q, b, q, f"q={q:2d} {LABEL[a]} vs {LABEL[b]}")
P(f"\n== Batch methods against sequential EI at an equal number of evaluations ({budget})")
for q in qs:
    for m in ALLB:
        compare(m, q, "ei", 1, f"q={q:2d} {LABEL[m]} vs sequential EI")
P("\n== q = 1 methods against each other")
compare("ei", 1, "qlogei", 1, "Sequential EI vs q-LogEI (q=1)")
compare("ei", 1, "ts", 1, "Sequential EI vs Thompson sampling (q=1)")
for m in ALLB + ["ei"]:
    for q in ([1] if m == "ei" else qs):
        compare(m, q, "random", 1, f"q={q:2d} {LABEL[m]} vs random search")

# ------------------------------------------------------------------ 5. equal rounds
agg_rounds = table(lambda q: ROUNDS * q)
P(f"\n== Mean normalized regret after {ROUNDS} rounds (= min({ROUNDS} q, {budget}) evaluations)")
for k, (e, lo, hi) in agg_rounds.items():
    if k[0] != "random":
        P(f"   q={k[1]:2d} {LABEL[k[0]]:18s} {e:.4f}  [{lo:.4f}, {hi:.4f}]")

# ------------------------------------------------------------------ 6. diversity and time
P("\n== Batch diversity: median minimum pairwise distance in a batch (unit cube); share of "
  "batches with a pair closer than 1e-3")
for q in qs:
    for m in ALLB:
        md = np.array([d for k, r in runs.items() if k[1] == m and k[2] == q for d in r["min_dist_unit"]])
        if len(md):
            P(f"   q={q:2d} {LABEL[m]:18s} median {np.median(md):.4f}  near-duplicates {np.mean(md < 1e-3):.4f}")
P("\n== Median wall-clock seconds per run")
for (m, q) in CONFIGS:
    sec = [r["seconds"] for k, r in runs.items() if k[1] == m and k[2] == q]
    if sec:
        P(f"   q={q:2d} {LABEL[m]:18s} {np.median(sec):8.1f} s  (max {np.max(sec):.1f})")

# ------------------------------------------------------------------ figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Drawn at the 6.0 in text width of the JMLR style, with 9 pt text.
WIDTH = 6.0
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "pdf.fonttype": 42, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def series(ax, tab, with_q1=False, methods=BATCH):
    xq = ([1] if with_q1 else []) + qs
    for i, m in enumerate(methods):
        pts = [(q, tab[(Q1_OF[m], 1) if q == 1 else (m, q)]) for q in xq
               if ((Q1_OF[m], 1) if q == 1 else (m, q)) in tab]
        if not pts:
            continue
        sh = (i - (len(methods) - 1) / 2) * 0.035
        x = [q * 2 ** sh for q, _ in pts]
        e = np.array([t[0] for _, t in pts])
        lo = np.array([t[1] for _, t in pts])
        hi = np.array([t[2] for _, t in pts])
        ax.errorbar(x, e, yerr=[e - lo, hi - e], color=COLOR[m], marker=MARKER[m], ms=4, lw=1.3,
                    capsize=0, elinewidth=0.9, label=LABEL[m])
    ax.set_xscale("log", base=2)
    ax.set_xticks(xq)
    ax.set_xticklabels([str(q) for q in xq])
    ax.minorticks_off()


if not (agg and rank_res):
    print("Note: exp24_summary not written (some main method is missing for some problem).")
else:
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.6), layout="constrained")
    series(axes[0], agg, with_q1=True)
    if ("random", 1) in agg:
        axes[0].axhline(agg[("random", 1)][0], color=MUTED, ls=":", lw=1.0, label="Random search")
    axes[0].set_ylabel("Mean normalized regret")
    axes[0].set_title(f"Equal evaluations ({budget})")
    series(axes[1], agg_rounds, with_q1=True)
    axes[1].set_ylabel("Mean normalized regret")
    axes[1].set_title(f"Equal rounds ({ROUNDS})")
    series(axes[2], rank_res)
    axes[2].set_ylabel("Rank (1 = best)")
    axes[2].set_title("Average rank")
    for ax in axes:
        ax.set_xlabel("Batch size $q$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3)
    save(fig, "exp24_summary")

if not (lie_rank and all((m, q) in agg for m in LIES for q in qs)):
    print("Note: exp24_lies not written (some lie is missing for some problem).")
else:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), layout="constrained")
    series(axes[0], agg, with_q1=True, methods=LIES)
    axes[0].set_ylabel("Mean normalized regret")
    axes[0].set_title(f"Equal evaluations ({budget})")
    axes[0].set_xlabel("Batch size $q$")
    series(axes[1], lie_rank, methods=LIES)
    axes[1].set_ylabel("Average rank (1 = best)")
    axes[1].set_title("Rank among the four lies")
    axes[1].set_xlabel("Batch size $q$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4)
    save(fig, "exp24_lies")

n, cols = len(problems), 4
rows_ = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows_, cols, figsize=(WIDTH, 1.3 * rows_ + 0.4), squeeze=False,
                         layout="constrained")
for i, p in enumerate(problems):
    ax = axes[i // cols][i % cols]
    tab = {}
    for (m, q) in CONFIGS:
        ss = complete_seeds(p, [(m, q)])
        if ss:
            v = np.array([nreg(runs[(p, m, q, s)]) for s in ss])
            h = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            tab[(m, q)] = (float(v.mean()), max(0.0, float(v.mean() - h)), min(1.0, float(v.mean() + h)))
    series(ax, tab, with_q1=True)
    if ("random", 1) in tab:
        ax.axhline(tab[("random", 1)][0], color=MUTED, ls=":", lw=1.0,
                   label="Random search" if i == 0 else None)
    ax.set_title(p)
    ax.set_xlabel("Batch size $q$" if i + cols >= n else "")
fig.supylabel("Normalized regret", fontsize=9, color=INK)
hl = {}
for ax in axes.flat:
    for h_, l_ in zip(*ax.get_legend_handles_labels()):
        hl.setdefault(l_, h_)
for j in range(n, rows_ * cols):
    axes[j // cols][j % cols].axis("off")
fig.legend(list(hl.values()), list(hl.keys()), loc="outside lower center", ncol=3)
save(fig, "exp24_per_problem")

with open(os.path.join(LOG, "exp24_numbers.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
print(f"\nWrote {os.path.join(LOG, 'exp24_numbers.txt')}, exp24_summary.csv, "
      "exp24_reference.json and the exp24 figures.")
