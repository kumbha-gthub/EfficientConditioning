#!/usr/bin/env python3
"""Sensitivity of the SDD to the optimizer starts and to the GP configuration.

Starts:  'diag3'  0.2, 0.5 and 0.8 along the diagonal of the domain
         'lhs20'  20 fixed Latin-hypercube points
Models:  GP A  raw inputs, lengthscales in [0.01, 100] (initial value 1),
               outputs re-standardized after every pseudo-observation
         GP B  as A, with inputs scaled to [0,1]^d
         GP C  the GP of all other experiments: inputs in [0,1]^d,
               lengthscales in [0.005, 2], output standardization fixed
               at the hyperparameter fit
         MQ-RBF (power); NN ensemble and random forest with 'lhs20'
Lies:    CL-min and KB

Reports the minimum pairwise distance, the number of batch members returned
unmoved at a starting point, the number of steps in which EI underflowed to
zero at every start, and diagnostics of the fitted lengthscales.
Hartmann-6D and Ackley-8D with q = 3, Levy-10D with q = 3 and 10; seeds 7,
21, 42, 73, 99; n_init = 30.

Use EXP_QUICK=1 for 2 seeds on Hartmann-6D and Levy-10D.
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    hartmann6, ackley, levy, GPSurrogate, MQRBFPowerSurrogate,
    NNEnsembleSurrogate, RFSurrogate, optimize_acquisition, latin_hypercube,
    pairwise_distances, get_gp_lengthscale, QUICK, save_json, dup_threshold,
)

SEEDS = [7, 21, 42, 73, 99]
DUP = 1e-3
N_INIT = 30
SETUPS = [
    dict(name="Hartmann-6D", func=hartmann6, d=6,
         bounds=np.array([[0.0, 1.0]] * 6), qs=[3]),
    dict(name="Ackley-8D", func=ackley, d=8,
         bounds=np.array([[-5.0, 5.0]] * 8), qs=[3]),
    dict(name="Levy-10D", func=levy, d=10,
         bounds=np.array([[-10.0, 10.0]] * 10), qs=[3, 10]),
]
if QUICK:
    SEEDS = SEEDS[:2]
    SETUPS = [SETUPS[0], SETUPS[2]]


GP_VARIANTS = {
    "GP A":    lambda B: GPSurrogate(ls="legacy", ynorm="refit"),
    "GP B":    lambda B: GPSurrogate(bounds=B, ls="legacy", ynorm="refit"),
    "GP C": lambda B: GPSurrogate(bounds=B),
}


def make_starts(kind, bounds):
    lo, hi = bounds[:, 0], bounds[:, 1]
    if kind == "diag3":
        return [lo + f * (hi - lo) for f in (0.2, 0.5, 0.8)]
    return list(latin_hypercube(20, bounds.shape[0], bounds, seed=12345))


def sdd(surr, y, q, bounds, starts, lie):
    f_best = float(np.min(y))
    batch, unmoved, ei_zero = [], 0, 0
    for _ in range(q):
        bx, be, bs = None, -np.inf, None
        for s in starts:
            x, v = optimize_acquisition(surr, bounds, f_best,
                                        fixed_start=np.array(s, float))
            if v > be:
                be, bx, bs = v, x.copy(), s
        if be <= 0.0:
            ei_zero += 1
        if np.linalg.norm(bx - bs) < 1e-6:
            unmoved += 1
        batch.append(bx.copy())
        yl = (float(np.min(y)) if lie == "cl_min"
              else float(surr.predict(bx.reshape(1, -1))[0][0]))
        surr.condition_on(bx, yl)
        f_best = min(f_best, yl)
    return np.array(batch), unmoved, ei_zero


def ls_report(setup, seeds):
    """Fitted lengthscales in units of the domain width, per GP variant."""
    B = setup["bounds"]; w = B[:, 1] - B[:, 0]
    rows = {}
    for label, ctor in GP_VARIANTS.items():
        rel, at_init, at_upper = [], 0, []
        for seed in seeds:
            X = latin_hypercube(N_INIT, setup["d"], B, seed=seed)
            y = setup["func"](X)
            m = ctor(B); m.fit(X, y)
            lo_b, hi_b, init = m._ls
            ls_fit = np.atleast_1d(m.kernel_.k2.length_scale)   # in model units
            rel += list(get_gp_lengthscale(m) / w)
            at_init += int(np.all(np.isclose(ls_fit, init)))
            at_upper.append(float(np.mean(ls_fit >= hi_b * (1 - 1e-3))))
        rows[label] = dict(median_rel=float(np.median(rel)),
                           seeds_all_at_init=at_init,
                           frac_dims_at_upper=float(np.mean(at_upper)))
        print(f"    {label:<9} median lengthscale / domain width = "
              f"{np.median(rel):8.3f}   seeds with ALL l at init: "
              f"{at_init}/{len(seeds)}   dims at the upper bound: "
              f"{100 * np.mean(at_upper):.0f}%")
    return rows


def main():
    t0 = time.time()
    print("=" * 84)
    print("EXPERIMENT 20 -- SDD sensitivity to optimizer starts and GP configuration")
    print(f"  seeds={SEEDS}  n_init={N_INIT}")
    print("=" * 84)
    out = {}
    for st in SETUPS:
        B, d = st["bounds"], st["d"]
        print(f"\n{'-' * 84}\n  {st['name']}\n{'-' * 84}")
        out[st["name"] + "|lengthscale"] = ls_report(st, SEEDS)
        configs = [("GP A", "diag3"), ("GP A", "lhs20"), ("GP B", "lhs20"),
                   ("GP C", "diag3"), ("GP C", "lhs20"),
                   ("MQ-RBF power fn", "diag3"), ("MQ-RBF power fn", "lhs20")]
        dup = dup_threshold(B)
        if not QUICK:
            configs += [("NN Ensemble", "lhs20"), ("Random Forest", "lhs20")]
        for q in st["qs"]:
            print(f"\n    q = {q}")
            print(f"    {'model':<16} {'starts':<6} {'lie':<7} "
                  f"{'per-seed min dist':<46} {'div':>4} {'unmoved':>8} {'EI=0':>5}")
            for mname, skind in configs:
                if q > 3 and mname in ("NN Ensemble", "Random Forest"):
                    continue
                lies = ["cl_min"] if mname in ("NN Ensemble", "Random Forest") \
                    else ["cl_min", "kb"]
                starts = make_starts(skind, B)
                for lie in lies:
                    mins, unm, ez = [], 0, 0
                    for seed in SEEDS:
                        X = latin_hypercube(N_INIT, d, B, seed=seed)
                        y = st["func"](X)
                        if mname in GP_VARIANTS:
                            m = GP_VARIANTS[mname](B)
                        elif mname == "MQ-RBF power fn":
                            m = MQRBFPowerSurrogate()
                        elif mname == "NN Ensemble":
                            m = NNEnsembleSurrogate()
                        else:
                            m = RFSurrogate()
                        m.fit(X, y)
                        b, u, z = sdd(m, y, q, B, starts, lie)
                        mins.append(float(pairwise_distances(b).min()))
                        unm += u; ez += z
                    nd = sum(v > dup for v in mins)
                    key = f"{st['name']}|q{q}|{mname}|{skind}|{lie}"
                    out[key] = dict(mins=mins, unmoved=unm, ei_zero=ez,
                                    n_members=q * len(SEEDS))
                    shown = "  ".join(f"{v:.4f}" for v in mins)
                    print(f"    {mname:<16} {skind:<6} {lie:<7} {shown:<46} "
                          f"{nd}/{len(mins):<2} {unm:>4}/{q * len(SEEDS):<3} {ez:>5}")
    save_json("exp20_sdd_start_sensitivity", out)
    print("\nColumns: div = seeds with min dist > 1e-3 x domain side; unmoved = batch members")
    print("returned exactly at a start point (optimizer did not move);")
    print("EI=0 = conditioning steps where the best EI over all starts underflowed.")
    print(f"\nTotal wall-clock: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
