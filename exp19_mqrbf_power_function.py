#!/usr/bin/env python3
"""MQ-RBF with a heuristic uncertainty against MQ-RBF with its power function.

Both models use the multiquadric interpolant; they differ in the
uncertainty (utils.MQRBFHeuristicSurrogate and utils.MQRBFPowerSurrogate).

Part A  uncertainty profile on Hartmann-6D (n_init = 30, 5 seeds): sigma at
        the data, and sigma_after / sigma_before at distance r from a
        conditioned point.
Part B  SDD (q = 3, 20 fixed starts) with CL-min and KB on Hartmann-6D and
        Ackley-8D, 5 seeds: the GP and both MQ-RBF models with EI, MQ-RBF
        (power) also with UCB, and the NN ensemble and the random forest
        under KB.
Part C  SDD at q up to 50 on Levy-10D (n_init = 30, seed 42).
Part D  time per conditioning step (n = 50, d = 6, q = 3).

Use EXP_QUICK=1 for 2 seeds, Hartmann-6D only and q <= 20.
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    hartmann6, ackley, levy, GPSurrogate, MQRBFHeuristicSurrogate, MQRBFPowerSurrogate,
    sdd_starts, dup_threshold,
    NNEnsembleSurrogate, RFSurrogate, optimize_acquisition, latin_hypercube,
    pairwise_distances, QUICK, save_json,
)
from scipy.spatial.distance import cdist

SEEDS = [7, 21, 42, 73, 99]
Q = 3
DUP = 1e-3
if QUICK:
    SEEDS = SEEDS[:2]

SETUPS = [
    dict(name="Hartmann-6D", func=hartmann6, d=6,
         bounds=np.array([[0.0, 1.0]] * 6), n_init=30),
    dict(name="Ackley-8D", func=ackley, d=8,
         bounds=np.array([[-5.0, 5.0]] * 8), n_init=30),
]
if QUICK:
    SETUPS = SETUPS[:1]

MODELS = {
    "GP":               GPSurrogate,
    "MQ-RBF heuristic": MQRBFHeuristicSurrogate,
    "MQ-RBF power fn":  MQRBFPowerSurrogate,
}


def make(name, bounds):
    """The GP scales its inputs to the unit cube; the MQ-RBF models are
    isotropic with a data-driven spread and need no rescaling."""
    return GPSurrogate(bounds=bounds) if name == "GP" else MODELS[name]()


def diag_starts(bounds, fracs=(0.2, 0.5, 0.8)):
    lo, hi = bounds[:, 0], bounds[:, 1]
    return [lo + f * (hi - lo) for f in fracs]


def sdd_batch(surr, y_train, q, bounds, starts, lie="cl_min", acq="ei"):
    f_best = float(np.min(y_train))
    batch = []
    for _ in range(q):
        bx, be = None, -np.inf
        for s in starts:
            x, v = optimize_acquisition(surr, bounds, f_best,
                                        fixed_start=np.array(s, float),
                                        acq_func=acq)
            if v > be:
                be, bx = v, x.copy()
        batch.append(bx.copy())
        if lie == "cl_min":
            yl = float(np.min(y_train))
        else:
            yl = float(surr.predict(bx.reshape(1, -1))[0][0])
        surr.condition_on(bx, yl)
        f_best = min(f_best, yl)
    return np.array(batch)


# ------------------------------------------------------------------
def part_a():
    print("\n" + "=" * 78)
    print("PART A -- Uncertainty-proxy diagnostics (Hartmann-6D, n_init=30)")
    print("=" * 78)
    d, B = 6, np.array([[0.0, 1.0]] * 6)
    radii = [0.0, 0.05, 0.1, 0.2, 0.4]
    out = {}
    for name, cls in MODELS.items():
        rows = []
        for seed in SEEDS:
            X = latin_hypercube(30, d, B, seed=seed)
            y = hartmann6(X)
            m = make(name, B); m.fit(X, y)
            rng = np.random.RandomState(seed)
            P = rng.rand(3000, d)
            _, sP = m.predict(P)
            _, sX = m.predict(X)
            corr = float(np.corrcoef(cdist(P, X).min(1), sP)[0, 1])
            ratio_data = float(np.mean(sX) / np.median(sP))
            xs = P[int(np.argmax(sP))]
            m2 = make(name, B); m2.fit(X, y)
            m2.condition_on(xs, float(np.min(y)))
            dirs = rng.randn(8, d)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            rr = []
            for r in radii:
                pts = np.clip(xs + r * dirs, 0, 1)
                rr.append(float(np.mean(m2.predict(pts)[1] / m.predict(pts)[1])))
            rows.append(dict(seed=seed, corr=corr, ratio_data=ratio_data,
                             ratio_r=rr))
        out[name] = rows
        corr = np.mean([r["corr"] for r in rows])
        rd = np.mean([r["ratio_data"] for r in rows])
        rr = np.mean([r["ratio_r"] for r in rows], axis=0)
        print(f"\n  {name}")
        print(f"    corr(sigma, distance to nearest data)  = {corr:+.3f}   "
              f"(an uncertainty should be > 0)")
        print(f"    mean sigma at data / median sigma       = {rd:.3f}   "
              f"(an uncertainty should be << 1)")
        print("    sigma_after / sigma_before after conditioning at x*, "
              "by distance r from x*:")
        print("      " + "  ".join(f"r={r:<4}: {v:6.3f}" for r, v in zip(radii, rr)))
        print("      (C2 needs ~0 at r=0, rising towards 1 with r)")
    return out


# ------------------------------------------------------------------
def part_b():
    print("\n" + "=" * 78)
    print(f"PART B -- SDD (20 fixed starts, q={Q}), seeds={SEEDS}")
    print("=" * 78)
    configs = [("GP", "cl_min", "ei"), ("GP", "kb", "ei"),
               ("MQ-RBF heuristic", "cl_min", "ei"),
               ("MQ-RBF heuristic", "kb", "ei"),
               ("MQ-RBF power fn", "cl_min", "ei"),
               ("MQ-RBF power fn", "kb", "ei"),
               ("MQ-RBF power fn", "cl_min", "ucb"),
               ("MQ-RBF power fn", "kb", "ucb")]
    extra = [("NN Ensemble", NNEnsembleSurrogate, "kb"),
             ("Random Forest", RFSurrogate, "kb")]
    out = {}
    for st in SETUPS:
        print(f"\n  {st['name']}")
        print(f"    {'Surrogate':<18} {'lie':<7} {'acq':<4} "
              f"{'per-seed min dist':<44} {'diverse':>8}")
        starts = sdd_starts(st["bounds"])
        dup = dup_threshold(st["bounds"])
        for name, lie, acq in configs:
            mins = []
            for seed in SEEDS:
                X = latin_hypercube(st["n_init"], st["d"], st["bounds"], seed=seed)
                y = st["func"](X)
                m = make(name, st["bounds"]); m.fit(X, y)
                b = sdd_batch(m, y, Q, st["bounds"], starts, lie, acq)
                mins.append(float(pairwise_distances(b).min()))
            nd = sum(v > dup for v in mins)
            out[f"{st['name']}|{name}|{lie}|{acq}"] = mins
            print(f"    {name:<18} {lie:<7} {acq:<4} "
                  f"{'  '.join(f'{v:.4f}' for v in mins):<44} {nd}/{len(mins)}")
        if not QUICK:
            for name, cls, lie in extra:
                mins = []
                for seed in SEEDS:
                    X = latin_hypercube(st["n_init"], st["d"], st["bounds"], seed=seed)
                    y = st["func"](X)
                    m = cls(); m.fit(X, y)
                    b = sdd_batch(m, y, Q, st["bounds"], starts, lie, "ei")
                    mins.append(float(pairwise_distances(b).min()))
                nd = sum(v > dup for v in mins)
                out[f"{st['name']}|{name}|{lie}|ei"] = mins
                print(f"    {name:<18} {lie:<7} {'ei':<4} "
                      f"{'  '.join(f'{v:.4f}' for v in mins):<44} {nd}/{len(mins)}")
    return out


# ------------------------------------------------------------------
def part_c():
    print("\n" + "=" * 78)
    print("PART C -- Large-q SDD, Levy-10D (n_init=30, seed 42, 20 fixed starts,")
    print("          threshold 0.02)")
    print("=" * 78)
    d, B = 10, np.array([[-10.0, 10.0]] * 10)
    X = latin_hypercube(30, d, B, seed=42)
    y = levy(X)
    starts = sdd_starts(B)
    dup = dup_threshold(B)
    runs = [("MQ-RBF power fn", "cl_min", 10), ("MQ-RBF power fn", "cl_min", 20),
            ("MQ-RBF power fn", "cl_min", 50), ("MQ-RBF power fn", "kb", 10),
            ("MQ-RBF power fn", "kb", 20), ("MQ-RBF heuristic", "kb", 10),
            ("GP", "kb", 10), ("GP", "kb", 20)]
    if QUICK:
        runs = [r for r in runs if r[2] <= 20]
    out = {}
    print(f"    {'Surrogate':<18} {'lie':<7} {'q':>3} {'min dist':>10} "
          f"{'mean dist':>10} {'diverse':>8} {'time(s)':>8}")
    for name, lie, q in runs:
        t0 = time.time()
        m = make(name, B); m.fit(X, y)
        b = sdd_batch(m, y, q, B, starts, lie, "ei")
        pdv = pairwise_distances(b)
        dt = time.time() - t0
        out[f"{name}|{lie}|q{q}"] = dict(min=float(pdv.min()), mean=float(pdv.mean()))
        print(f"    {name:<18} {lie:<7} {q:>3} {pdv.min():>10.4f} {pdv.mean():>10.4f} "
              f"{'YES' if pdv.min() > dup else 'NO':>8} {dt:>8.1f}")
    return out


# ------------------------------------------------------------------
def part_d():
    print("\n" + "=" * 78)
    print("PART D -- Conditioning-step time (n=50, d=6, q=3, 20 reps)")
    print("=" * 78)
    d, B = 6, np.array([[0.0, 1.0]] * 6)
    X = latin_hypercube(50, d, B, seed=42)
    y = hartmann6(X)
    reps = 5 if QUICK else 20
    rng = np.random.RandomState(42)
    out = {}
    for name, cls in MODELS.items():
        ts = []
        for _ in range(reps):
            m = make(name, B); m.fit(X, y)
            xf = rng.uniform(0, 1, size=(Q, d)); yf = rng.randn(Q)
            t0 = time.perf_counter()
            for j in range(Q):
                m.condition_on(xf[j], yf[j])
            ts.append((time.perf_counter() - t0) / Q)
        out[name] = [float(np.mean(ts)), float(np.std(ts))]
        print(f"    {name:<18} {np.mean(ts):.6f} +- {np.std(ts):.6f} s / step")
    return out


def main():
    t0 = time.time()
    print("=" * 78)
    print("EXPERIMENT 19 -- MQ-RBF: heuristic uncertainty vs power function")
    print("=" * 78)
    res = dict(A=part_a(), B=part_b(), C=part_c(), D=part_d())
    save_json("exp19_mqrbf_power_function", res)
    print(f"\nTotal wall-clock: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
