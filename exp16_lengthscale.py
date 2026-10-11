#!/usr/bin/env python3
"""Suppression radius and batch separation for a GP with a fixed kernel.

GP with a fixed isotropic SE kernel (sigma_f^2 = 1, sigma_n^2 = 1e-6) on
Ackley rescaled to [0,1]^2 (lengthscales 0.03, 0.05, 0.08, 0.12, 0.2;
n_init = 10) and Hartmann-6D (lengthscales 0.1, 0.2, 0.3; n_init = 12),
5 seeds each.

Part A: condition (KB) at the point of largest variance and record, along 24
        random rays, the radius at which the fractional variance reduction
        falls below tau = 0.5; compare with l * sqrt(ln(c / tau)).
Part B: SDD batches of q = 3 under KB and CL-min; minimum pairwise distance.

Use --quick for the two-dimensional setup with 3 seeds.
"""

import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import sdd_starts
from utils import (
    hartmann6, ackley, optimize_acquisition, latin_hypercube,
    pairwise_distances, FIG_DIR,
)

# -- configuration ----------------------------------------------------
SEEDS      = [7, 21, 42, 73, 99]
Q          = 3
TAU        = 0.5
NOISE      = 1e-6
SIGMA_F2   = 1.0                      # signal variance, fixed
SETUPS = [
    dict(name="Ackley-2D",   d=2, ells=[0.03, 0.05, 0.08, 0.12, 0.2],
         func=lambda X: ackley(10.0 * X - 5.0), n_init=10),
    dict(name="Hartmann-6D", d=6, ells=[0.1, 0.2, 0.3],
         func=hartmann6, n_init=12),
]
# ---------------------------------------------------------------------


class FixedGP:
    """GP with a fixed SE kernel (sigma_f^2, l and sigma_n^2 fixed).

    Works internally in standardized-y space; EI's argmax is invariant
    to joint (mu, sigma) rescaling, so batch selection is unaffected,
    and Part A uses only variance ratios, where the scale cancels.
    """

    def __init__(self, lengthscale):
        self.kernel = (ConstantKernel(SIGMA_F2, constant_value_bounds="fixed")
                       * RBF(length_scale=lengthscale,
                             length_scale_bounds="fixed"))
        self.X_train = None
        self.y_std_train = None

    def fit(self, X, y_std):
        self.X_train = X.copy()
        self.y_std_train = y_std.copy()
        self.model = GaussianProcessRegressor(
            kernel=self.kernel, alpha=NOISE, optimizer=None,
            normalize_y=False)
        self.model.fit(X, y_std)

    def predict(self, X):
        X = np.atleast_2d(X)
        mu, sigma = self.model.predict(X, return_std=True)
        return mu, sigma

    def condition_on(self, x_new, y_new_std):
        self.X_train = np.vstack([self.X_train, np.atleast_2d(x_new)])
        self.y_std_train = np.append(self.y_std_train, y_new_std)
        self.fit(self.X_train, self.y_std_train)


def standardize(y):
    return (y - np.mean(y)) / max(np.std(y), 1e-12)


# ================================================================
#  PART A - direct suppression-radius measurement
# ================================================================

def measure_radius(setup, ell, seed, n_dirs=24, n_radii=600):
    d = setup["d"]
    bounds = np.array([[0.0, 1.0]] * d)
    X = latin_hypercube(setup["n_init"], d, bounds, seed=seed)
    y = standardize(setup["func"](X))

    gp = FixedGP(ell)
    gp.fit(X, y)

    # x* = candidate with the largest variance
    cand = latin_hypercube(4000, d, bounds, seed=seed + 500)
    _, sg = gp.predict(cand)
    x_star = cand[int(np.argmax(sg))]
    _, s_star = gp.predict(x_star.reshape(1, -1))
    s2_star = float(s_star[0] ** 2)
    rho = s2_star / SIGMA_F2          # close to 1 far from the data

    # prediction without data: r_tau = l * sqrt(log(sf2 / (tau * (sf2 + noise))))
    r_regime = ell * np.sqrt(np.log(SIGMA_F2 / (TAU * (SIGMA_F2 + NOISE))))

    # rays from x*: first-exit radius of the connected suppression region
    rng = np.random.RandomState(seed + 900)
    dirs = rng.randn(n_dirs, d)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = np.linspace(1e-4, min(4.0 * ell, np.sqrt(d)), n_radii)

    all_pts, keep = [], []
    for u in dirs:
        pts = x_star[None, :] + radii[:, None] * u[None, :]
        keep.append(np.all((pts >= 0.0) & (pts <= 1.0), axis=1))
        all_pts.append(pts)
    all_pts = np.vstack(all_pts)

    _, s_before = gp.predict(all_pts)
    gp.condition_on(x_star, float(gp.predict(x_star.reshape(1, -1))[0][0]))
    _, s_after = gp.predict(all_pts)
    frac = 1.0 - (s_after ** 2) / np.maximum(s_before ** 2, 1e-300)

    r_measured = []
    for i in range(n_dirs):
        f = frac[i * n_radii:(i + 1) * n_radii]
        inside = keep[i]
        r_dir = None
        for j in range(n_radii):
            if not inside[j]:
                r_dir = None            # censored by the domain boundary
                break
            if f[j] < TAU:
                r_dir = radii[j - 1] if j > 0 else 0.0
                break
        else:
            r_dir = None                # never crossed within range
        if r_dir is not None:
            r_measured.append(r_dir)
    r_med = float(np.median(r_measured)) if r_measured else float("nan")
    return r_med, r_regime, rho


# ================================================================
#  PART B - SDD batch separation vs lengthscale
# ================================================================

def sdd_separation(setup, ell, seed, strategy):
    d = setup["d"]
    bounds = np.array([[0.0, 1.0]] * d)
    starts = sdd_starts(bounds)

    X = latin_hypercube(setup["n_init"], d, bounds, seed=seed)
    y = standardize(setup["func"](X))

    gp = FixedGP(ell)
    gp.fit(X, y)
    f_best = float(np.min(y))

    batch = []
    for _ in range(Q):
        best_x, best_ei = None, -np.inf
        for s in starts:
            x, ei = optimize_acquisition(gp, bounds, f_best,
                                         fixed_start=np.array(s, float))
            if ei > best_ei:
                best_ei, best_x = ei, x.copy()
        batch.append(best_x.copy())
        if strategy == "kb":
            y_lie = float(gp.predict(best_x.reshape(1, -1))[0][0])
        else:                                            # cl-min
            y_lie = float(np.min(y))
        gp.condition_on(best_x, y_lie)
        f_best = min(f_best, y_lie)

    return float(np.min(pairwise_distances(np.array(batch))))


# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="d=2 only, 3 seeds")
    args = ap.parse_args()
    seeds = SEEDS[:3] if args.quick else SEEDS
    setups = SETUPS[:1] if args.quick else SETUPS

    print("=" * 78)
    print("EXPERIMENT 16 - Suppression radius and batch separation vs lengthscale")
    print(f"  tau={TAU}, sigma_f^2={SIGMA_F2}, sigma_n^2={NOISE}, "
          f"q={Q}, seeds={seeds}")
    print("=" * 78)

    results = {}   # (setup, ell) -> dict

    for setup in setups:
        print(f"\n{'=' * 60}\n  {setup['name']}  "
              f"(d={setup['d']}, n_init={setup['n_init']})\n{'=' * 60}")
        for ell in setup["ells"]:
            rm, rt, rhos, sep_kb, sep_cl = [], [], [], [], []
            for seed in seeds:
                r_med, r_regime, rho = measure_radius(setup, ell, seed)
                rm.append(r_med)
                rt.append(r_regime)
                rhos.append(rho)
                sep_kb.append(sdd_separation(setup, ell, seed, "kb"))
                sep_cl.append(sdd_separation(setup, ell, seed, "cl"))
            results[(setup["name"], ell)] = dict(
                r_meas=(np.nanmean(rm), np.nanstd(rm)),
                r_theo=(np.mean(rt), np.std(rt)),
                rho=float(np.mean(rhos)),
                kb=(np.mean(sep_kb), np.std(sep_kb)),
                cl=(np.mean(sep_cl), np.std(sep_cl)),
            )
            r = results[(setup["name"], ell)]
            in_regime = "in-regime" if r['rho'] >= 0.9 else "regime exit"
            print(f"  l={ell:<5} rho={r['rho']:.2f} ({in_regime:<11}) "
                  f"r_0.5 meas={r['r_meas'][0]:.3f}+-{r['r_meas'][1]:.3f} "
                  f"pred 0.83l={r['r_theo'][0]:.3f} | "
                  f"KB sep={r['kb'][0]:.3f}+-{r['kb'][1]:.3f} "
                  f"CL={r['cl'][0]:.3f}+-{r['cl'][1]:.3f}")

        # summary statistics for the sweep
        ells = np.array(setup["ells"])
        rmeas = np.array([results[(setup["name"], e)]["r_meas"][0]
                          for e in ells])
        rpred = np.array([results[(setup["name"], e)]["r_theo"][0]
                          for e in ells])
        rhos = np.array([results[(setup["name"], e)]["rho"]
                         for e in ells])
        ok = rhos >= 0.9
        if ok.sum() >= 2:
            rel = np.abs(rmeas[ok] - rpred[ok]) / rpred[ok]
            c_r = np.corrcoef(ells[ok], rmeas[ok])[0, 1]
            print(f"\n  In-regime rows (rho>=0.9): mean |r_meas - 0.83l| / "
                  f"0.83l = {100*rel.mean():.1f}%   "
                  f"Pearson corr(l, r_meas) = {c_r:.4f}")

    # -- figure -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"Ackley-2D": "#1E88E5", "Hartmann-6D": "#E53935"}
    for setup in setups:
        name = setup["name"]
        ells = np.array(setup["ells"])
        rmeas = np.array([results[(name, e)]["r_meas"][0] for e in ells])
        rmstd = np.array([results[(name, e)]["r_meas"][1] for e in ells])
        rtheo = np.array([results[(name, e)]["r_theo"][0] for e in ells])
        kbm = np.array([results[(name, e)]["kb"][0] for e in ells])
        kbs = np.array([results[(name, e)]["kb"][1] for e in ells])

        axes[0].errorbar(ells, rmeas, yerr=rmstd, fmt="o-", lw=2,
                         color=colors[name], label=f"{name} measured")
        axes[0].plot(ells, rtheo, "--", lw=1.5, color=colors[name],
                     alpha=0.7, label=f"{name} 0.83l pred.")
        axes[1].errorbar(ells, kbm, yerr=kbs, fmt="o-", lw=2,
                         color=colors[name], label=f"{name} KB min sep")
    lmax = max(max(s["ells"]) for s in setups)
    lgrid = np.linspace(0, lmax, 50)
    axes[1].plot(lgrid, 0.83 * lgrid, "k--", lw=1.5, alpha=0.7,
                 label="0.83 l")
    axes[0].set_xlabel("lengthscale l")
    axes[0].set_ylabel("suppression radius r_0.5")
    axes[0].set_title("Part A: measured radius vs 0.83 l")
    axes[1].set_xlabel("lengthscale l")
    axes[1].set_ylabel("min pairwise separation (q=3)")
    axes[1].set_title("Part B: SDD separation vs lengthscale")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "exp16_lengthscale.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved -> {out}")

    # -- LaTeX --------------------------------------------------
    print("\n" + "=" * 78)
    print("LATEX")
    print("=" * 78)
    print(r"\begin{tabular}{llccccc}")
    print(r"\toprule")
    print(r"Setup & $\ell$ & $\rho$ & $r_{0.5}$ meas. & $0.83\ell$ pred. & KB min sep & CL min sep \\")
    print(r"\midrule")
    for setup in setups:
        for e in setup["ells"]:
            r = results[(setup["name"], e)]
            print(f"  {setup['name']} & {e} & {r['rho']:.2f} & "
                  f"${r['r_meas'][0]:.3f} \\pm {r['r_meas'][1]:.3f}$ & "
                  f"{r['r_theo'][0]:.3f} & "
                  f"${r['kb'][0]:.3f} \\pm {r['kb'][1]:.3f}$ & "
                  f"${r['cl'][0]:.3f} \\pm {r['cl'][1]:.3f}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal wall-clock: {(time.time()-t0)/60:.1f} min")
