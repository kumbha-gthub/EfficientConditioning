#!/usr/bin/env python3
"""Analysis of the GP control for q-LogEI (exp27_qlogei_paper_gp.py).

Usage:  python exp27_analysis.py            (after `exp27_qlogei_paper_gp.py run`)

Reads  results/data/exp27/<problem>/<method>_q<q>_s<seed>.json
Writes results/exp27_summary.csv          per problem, arm, q: mean, sd, n
       results/exp27_reference.json       reference values of the tuning problems
       results/EXP27_NUMBERS.md           every number and test, ready to quote
       results/figures/exp27_summary.png  mean regret and loss from batching vs q
       results/figures/exp27_per_problem.png

Definitions are those of the batch-size study (analysis/exp24_analysis.py):
  normalized regret r = (f_best - f_ref) / (f_init - f_ref), clipped to [0, 1];
  f_init = best value of the initial design (shared by all arms of a seed);
  f_ref  = known minimum (synthetic) or, for tuning problems, the lowest value
           found by any exp27 run or by 200,000 uniform random configurations
           (seed 12345, the same reference search as exp24).
Statistics: two-sided Wilcoxon signed-rank tests (scipy defaults) on the 19
per-problem mean regrets; Holm's method within each family stated below;
95% percentile-bootstrap intervals over seeds within problems (2,000
resamples, seed 0).  Only (problem, seed) pairs with every arm present enter.
"""

import os
import sys
import csv
import json
import glob
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp24_problems import problem_names, get_problem             # noqa: E402

VERSION = "exp27-v1"
RES = os.environ.get("EXP_RESULTS_DIR", os.path.join(HERE, "results"))
DATA = os.path.join(RES, "data", "exp27")
FIG = os.path.join(RES, "figures")
ARMS = {1: ["ei", "qlogei", "qlogei_pgp"],
        4: ["cl_min", "kb", "qlogei", "qlogei_pgp"],
        16: ["cl_min", "kb", "qlogei", "qlogei_pgp"]}
Q1_OF = {"cl_min": "ei", "kb": "ei", "ei": "ei", "qlogei": "qlogei", "qlogei_pgp": "qlogei_pgp"}
LABEL = {"ei": "Sequential EI", "cl_min": "CL-min", "kb": "KB",
         "qlogei": "q-LogEI, BoTorch GP", "qlogei_pgp": "q-LogEI, paper GP"}
SYNTH = ["Hartmann-6", "Shekel-4", "Ackley-8", "Styblinski-Tang-8", "Rosenbrock-8", "Levy-10"]


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def wil(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def fmt_p(p):
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


def main():
    runs = {}
    for path in glob.glob(os.path.join(DATA, "*", "*.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("config", {}).get("version") == VERSION:
            runs[(r["problem"], r["method"], int(r["q"]), int(r["seed"]))] = r
    if not runs:
        sys.exit(f"no exp27 results in {DATA}")
    problems = [p for p in problem_names() if any(k[0] == p for k in runs)]
    configs = [(m, q) for q in ARMS for m in ARMS[q]]
    seeds = {p: sorted(s for s in {k[3] for k in runs if k[0] == p}
                       if all((p, m, q, s) in runs for m, q in configs)) for p in problems}
    missing = [(p, m, q, s) for p in problems for s in {k[3] for k in runs if k[0] == p}
               for m, q in configs if (p, m, q, s) not in runs]

    # reference values
    ref_path = os.path.join(RES, "exp27_reference.json")
    ref = json.load(open(ref_path)) if os.path.exists(ref_path) else {}
    for p in problems:
        prob = get_problem(p)
        if prob.f_star is not None:
            ref[p] = dict(f_ref=float(prob.f_star), source="known minimum")
            continue
        if "random_search_min" not in ref.get(p, {}):
            rs = np.random.default_rng(12345)
            ref.setdefault(p, {})["random_search_min"] = float(min(
                np.min(prob(rs.uniform(size=(10000, prob.dim)))) for _ in range(20)))
        ref[p]["best_run_min"] = float(min(r["final"] for k, r in runs.items() if k[0] == p))
        ref[p]["f_ref"] = min(ref[p]["best_run_min"], ref[p]["random_search_min"])
        ref[p]["source"] = "min(best exp27 run, 200k random search)"
    os.makedirs(RES, exist_ok=True)
    json.dump(ref, open(ref_path, "w"), indent=1)

    def regret(p, m, q, s):
        r = runs[(p, m, q, s)]
        f0, fr, fb = r["best"][0], ref[p]["f_ref"], r["final"]
        return 0.0 if f0 - fr <= 0 else float(np.clip((fb - fr) / (f0 - fr), 0.0, 1.0))

    R = {(p, m, q): np.array([regret(p, m, q, s) for s in seeds[p]]) for p in problems for m, q in configs}
    M = {k: float(v.mean()) for k, v in R.items()}

    def avg(m, q, group=None):
        ps = [p for p in problems if group is None or (p in SYNTH) == (group == "synthetic")]
        return float(np.mean([M[(p, m, q)] for p in ps]))

    rng = np.random.default_rng(0)

    def boot(m, q):
        vals = []
        for _ in range(2000):
            vals.append(np.mean([R[(p, m, q)][rng.integers(0, len(seeds[p]), len(seeds[p]))].mean()
                                 for p in problems]))
        return np.percentile(vals, [2.5, 97.5])

    def vec(m, q):
        return np.array([M[(p, m, q)] for p in problems])

    def loss(m, q):
        return vec(m, q) - vec(Q1_OF[m], 1)

    L = []
    P = L.append
    n_seeds = sorted({len(v) for v in seeds.values()})
    P("# exp27 numbers: GP control for q-LogEI\n")
    P(f"Problems: {len(problems)}; complete seeds per problem: {n_seeds}; "
      f"runs loaded: {len(runs)}; missing runs: {len(missing)}")
    fails = sum(r["n_fail"] for r in runs.values())
    pgp = [r for r in runs.values() if r["method"] == "qlogei_pgp"]
    gpd = max(r["gp_max_rel_diff"] for r in pgp if r.get("gp_max_rel_diff") is not None)
    nchk = sum(r.get("gp_checks", 0) for r in pgp)
    P(f"Failed acquisition steps (replaced by random points): {fails}")
    P(f"GP equivalence of the q-LogEI (paper GP) arm: {nchk} batch checks, "
      f"largest posterior difference {gpd:.1e} (relative to the output scale)\n")

    P("## Mean normalized regret after 100 evaluations (19 problems; synthetic 6 / tuning 13)\n")
    P("| Arm | q | all 19 | 95% bootstrap | synthetic | tuning |")
    P("|---|---|---|---|---|---|")
    rows = []
    for q in ARMS:
        for m in ARMS[q]:
            lo, hi = boot(m, q)
            P(f"| {LABEL[m]} | {q} | {avg(m, q):.3f} | [{lo:.3f}, {hi:.3f}] | "
              f"{avg(m, q, 'synthetic'):.3f} | {avg(m, q, 'tuning'):.3f} |")
            rows.append((m, q))

    def block(title, pairs, holm_family=True):
        P(f"\n## {title}\n")
        P("| Comparison | q | mean A | mean B | p | Holm | lower in |")
        P("|---|---|---|---|---|---|---|")
        ps = [wil(a, b) for (_, _, a, b) in pairs]
        adj = holm(ps) if holm_family else ps
        out = []
        for (name, q, a, b), p, h in zip(pairs, ps, adj):
            lower = "A" if np.mean(a) < np.mean(b) else "B"
            P(f"| {name} | {q} | {np.mean(a):.3f} | {np.mean(b):.3f} | {fmt_p(p)} | {fmt_p(h)} | {lower} |")
            out.append((name, q, float(np.mean(a)), float(np.mean(b)), p, float(h)))
        return out

    main_pairs = [(f"{LABEL[m]} (A) vs q-LogEI, paper GP (B)", q, vec(m, q), vec("qlogei_pgp", q))
                  for q in (4, 16) for m in ("cl_min", "kb")]
    rep_pairs = [(f"{LABEL[m]} (A) vs q-LogEI, BoTorch GP (B)", q, vec(m, q), vec("qlogei", q))
                 for q in (4, 16) for m in ("cl_min", "kb")]
    gp_pairs = [("q-LogEI paper GP (A) vs BoTorch GP (B)", q, vec("qlogei_pgp", q), vec("qlogei", q))
                for q in (1, 4, 16)]
    q1_pairs = [("q-LogEI paper GP (A) vs sequential EI (B)", 1, vec("qlogei_pgp", 1), vec("ei", 1)),
                ("q-LogEI BoTorch GP (A) vs sequential EI (B)", 1, vec("qlogei", 1), vec("ei", 1))]
    loss_pairs = [(f"loss {LABEL[m]} (A) vs loss {LABEL[o]} (B)", q, loss(m, q), loss(o, q))
                  for q in (4, 16) for m in ("cl_min", "kb") for o in ("qlogei_pgp", "qlogei")]
    res_main = block("Main test: same GP for all arms (Holm over these 4)", main_pairs)
    res_rep = block("Replication of Section 7: q-LogEI on BoTorch's GP (Holm over these 4)", rep_pairs)
    block("GP effect within q-LogEI (Holm over these 3)", gp_pairs)
    block("q = 1: same GP, LogEI vs EI (Holm over these 2)", q1_pairs)
    block("Loss from batching, regret(q) - regret(q = 1) of the same arm (Holm over these 8)", loss_pairs)

    P("\n## Loss from batching (mean over 19 problems)\n")
    P("| Arm | q = 4 | q = 16 |")
    P("|---|---|---|")
    for m in ("cl_min", "kb", "qlogei", "qlogei_pgp"):
        P(f"| {LABEL[m]} | {np.mean(loss(m, 4)):+.3f} | {np.mean(loss(m, 16)):+.3f} |")

    # verdict
    sig = {(n, q): h < 0.05 and a < b for (n, q, a, b, p, h) in res_main}
    s4 = all(v for (n, q), v in sig.items() if q == 4)
    s16 = all(v for (n, q), v in sig.items() if q == 16)
    rep = all(h < 0.05 and a < b for (n, q, a, b, p, h) in res_rep)
    P("\n## Verdict (automatic, from the Holm-adjusted main test)\n")
    if s4 and s16:
        P("SCENARIO A: with the same GP, CL-min and KB still had significantly lower regret than "
          "q-LogEI at q = 4 and q = 16. The GP difference does not explain the Section 7 result.")
    elif s16:
        P("SCENARIO B: with the same GP, CL-min and KB had significantly lower regret than q-LogEI "
          "at q = 16 but not (both) at q = 4.")
    else:
        P("SCENARIO C: with the same GP, CL-min and KB did not have significantly lower regret than "
          "q-LogEI at q = 16 (Holm). Part of the Section 7 difference is due to the GP.")
    P(f"Replication with BoTorch's GP (both q, both lies significant): {'yes' if rep else 'no'}")
    if missing:
        P(f"\nWARNING: {len(missing)} runs missing, e.g. {missing[:5]}")
    os.makedirs(RES, exist_ok=True)
    open(os.path.join(RES, "EXP27_NUMBERS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))

    # LaTeX table for Appendix D.4 (mean normalized regret, all 19 problems)
    def cell(m, q):
        return f"{avg(m, q):.3f}" if m in ARMS[q] else "--"
    tex = [r"\begin{table}[ht]",
           r"\caption{Mean normalized regret after 100 evaluations, averaged over the 19 problems, in the "
           r"control with $q$-LogEI on our GP (seeds 0--9). At $q = 1$, CL-min and KB are sequential EI. "
           r"With ten seeds and reference values from this execution, the values differ from those of "
           r"Table~\ref{tab:batch}.}",
           r"\label{tab:qlogei_gp}", r"\centering", r"\small",
           r"\begin{tabular}{lccc}", r"\toprule", r"Method & $q = 1$ & $q = 4$ & $q = 16$ \\", r"\midrule",
           f"CL-min & {cell('ei', 1)} & {cell('cl_min', 4)} & {cell('cl_min', 16)} \\\\",
           f"KB & {cell('ei', 1)} & {cell('kb', 4)} & {cell('kb', 16)} \\\\",
           f"$q$-LogEI, BoTorch's GP & {cell('qlogei', 1)} & {cell('qlogei', 4)} & {cell('qlogei', 16)} \\\\",
           f"$q$-LogEI, GP of Appendix~\\ref{{app:impl}} & {cell('qlogei_pgp', 1)} & {cell('qlogei_pgp', 4)} "
           f"& {cell('qlogei_pgp', 16)} \\\\",
           r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(RES, "exp27_table.tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")

    with open(os.path.join(RES, "exp27_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem", "method", "q", "n_seeds", "mean_regret", "sd_regret"])
        for p in problems:
            for m, q in configs:
                v = R[(p, m, q)]
                w.writerow([p, m, q, len(v), f"{v.mean():.6f}", f"{v.std(ddof=1) if len(v) > 1 else 0:.6f}"])

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    col = {"cl_min": "#2a78d6", "kb": "#eb6834", "qlogei": "#eda100", "qlogei_pgp": "#7a4fbf", "ei": "#52514e"}
    mk = {"cl_min": "o", "kb": "s", "qlogei": "D", "qlogei_pgp": "P", "ei": "x"}
    qs = [1, 4, 16]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in ("cl_min", "kb", "qlogei", "qlogei_pgp"):
        mm = [Q1_OF[m] if q == 1 else m for q in qs]
        y = [avg(a, q) for a, q in zip(mm, qs)]
        ci = np.array([boot(a, q) for a, q in zip(mm, qs)])
        ax[0].errorbar(qs, y, yerr=[np.array(y) - ci[:, 0], ci[:, 1] - np.array(y)], marker=mk[m],
                       color=col[m], label=LABEL[m], capsize=3)
        ax[1].plot([4, 16], [np.mean(loss(m, 4)), np.mean(loss(m, 16))], marker=mk[m], color=col[m],
                   label=LABEL[m])
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xticks(qs if a is ax[0] else [4, 16])
        a.set_xticklabels([str(q) for q in (qs if a is ax[0] else [4, 16])])
        a.set_xlabel("Batch size q")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Mean normalized regret (19 problems)")
    ax[0].set_title("After 100 evaluations (q = 1: sequential version)")
    ax[1].set_ylabel("regret(q) - regret(q = 1), same arm")
    ax[1].set_title("Loss from batching")
    ax[1].axhline(0, color="#999", lw=0.8)
    ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp27_summary.png"), dpi=200)

    fig, axs = plt.subplots(4, 5, figsize=(15, 11))
    for a, p in zip(axs.flat, problems):
        for m in ("cl_min", "kb", "qlogei", "qlogei_pgp"):
            mm = [Q1_OF[m] if q == 1 else m for q in qs]
            y = [M[(p, x, q)] for x, q in zip(mm, qs)]
            se = [R[(p, x, q)].std(ddof=1) / np.sqrt(len(R[(p, x, q)])) if len(R[(p, x, q)]) > 1 else 0
                  for x, q in zip(mm, qs)]
            a.errorbar(qs, y, yerr=1.96 * np.array(se), marker=mk[m], color=col[m], ms=4, capsize=2,
                       label=LABEL[m])
        a.set_xscale("log", base=2)
        a.set_xticks(qs)
        a.set_xticklabels([str(q) for q in qs])
        a.set_title(p, fontsize=9)
        a.grid(alpha=0.3)
    for a in list(axs.flat)[len(problems):]:
        a.axis("off")
    axs.flat[0].legend(fontsize=7, frameon=False)
    fig.supxlabel("Batch size q")
    fig.supylabel("Normalized regret after 100 evaluations (mean, ±1.96 SE)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp27_per_problem.png"), dpi=150)
    print(f"\nwrote {os.path.join(RES, 'EXP27_NUMBERS.md')} and figures in {FIG}")


if __name__ == "__main__":
    main()
