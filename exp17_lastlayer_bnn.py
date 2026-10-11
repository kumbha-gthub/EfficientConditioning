#!/usr/bin/env python3
"""Last-layer Bayesian neural network in the SDD, with an uncertainty profile.

The last-layer BNN (a DNGO-style neural-linear model) keeps the feature map
of a trained ReLU network fixed and applies Bayesian linear regression to the
last-layer features, so a pseudo-observation is absorbed in closed form
(C1).  For the GP, the last-layer BNN and an NN ensemble without retraining,
the script runs the SDD (q = 3, 20 fixed starts, CL-min) and measures where
the uncertainty contracts after conditioning on a CL-min lie at the point of
largest uncertainty.

Hartmann-6D and Ackley-8D, 5 seeds.  Use --quick for Hartmann-6D with 2 seeds.
"""

import os, sys, time, argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import sdd_starts, dup_threshold
from utils import (
    hartmann6, ackley, GPSurrogate, NNEnsembleSurrogate,
    optimize_acquisition, latin_hypercube, pairwise_distances,
)
from sklearn.neural_network import MLPRegressor

SEEDS = [7, 21, 42, 73, 99]
Q = 3
DUP = 1e-3


class LastLayerBNN:
    """Neural-linear (last-layer Bayesian) surrogate.

    Fixed feature map phi from a trained MLP's penultimate layer;
    Bayesian linear regression head with closed-form conditioning.
    Predictive variance = alpha_noise + phi(x)^T Sigma phi(x), the
    standard last-layer-Laplace predictive.
    """

    def __init__(self, hidden=(64, 64), prior_var=1.0, noise_var=1e-4):
        self.hidden = hidden
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.net = None
        self.X_train = None
        self.y_train = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def _features(self, X):
        """Penultimate-layer activations (ReLU MLP), with a bias column."""
        A = np.asarray(X, dtype=float)
        # forward through all but the last Linear layer
        for W, b in zip(self.net.coefs_[:-1], self.net.intercepts_[:-1]):
            A = np.maximum(A @ W + b, 0.0)            # ReLU
        return np.hstack([A, np.ones((A.shape[0], 1))])

    def _posterior(self):
        Phi = self._features(self.X_train)
        d = Phi.shape[1]
        A = (Phi.T @ Phi) / self.noise_var + np.eye(d) / self.prior_var
        self.Sigma = np.linalg.inv(A)
        yb = (self.y_train - self.y_mean) / self.y_std
        self.mN = self.Sigma @ (Phi.T @ yb) / self.noise_var

    def fit(self, X, y, retrain=True):
        self.X_train = np.atleast_2d(X).copy()
        self.y_train = np.asarray(y, float).copy()
        self.y_mean = float(np.mean(self.y_train))
        self.y_std = float(max(np.std(self.y_train), 1e-8))
        if retrain or self.net is None:
            yb = (self.y_train - self.y_mean) / self.y_std
            self.net = MLPRegressor(hidden_layer_sizes=self.hidden,
                                    max_iter=2000, random_state=0,
                                    learning_rate="adaptive")
            self.net.fit(self.X_train, yb)
        self._posterior()

    def predict(self, X):
        Phi = self._features(np.atleast_2d(X))
        mu = (Phi @ self.mN) * self.y_std + self.y_mean
        var = self.noise_var + np.einsum("ij,jk,ik->i", Phi, self.Sigma, Phi)
        sigma = np.sqrt(np.maximum(var, 1e-12)) * self.y_std
        return mu, sigma

    def condition_on(self, x_new, y_new):
        """Append data and update the last-layer posterior in closed form;
        the feature map stays fixed."""
        self.X_train = np.vstack([self.X_train, np.atleast_2d(x_new)])
        self.y_train = np.append(self.y_train, y_new)
        self._posterior()


def sdd_batch(surr, y_train, q, bounds, d):
    starts = sdd_starts(bounds)
    f_best = float(np.min(y_train))
    batch = []
    for _ in range(q):
        bx, be = None, -np.inf
        for s in starts:
            x, ei = optimize_acquisition(surr, bounds, f_best,
                                         fixed_start=np.array(s, float))
            if ei > be:
                be, bx = ei, x.copy()
        batch.append(bx.copy())
        surr.condition_on(bx, float(np.min(y_train)))     # CL-min
        f_best = min(f_best, float(np.min(y_train)))
    return np.array(batch)


RADII = [0.0, 0.05, 0.1, 0.2, 0.4]     # fractions of the domain side length


def locality_profile(ctor, X, y, bounds, seed):
    """Uncertainty profile, as in exp19 Part A.

    Fit the model, take x* = the point of largest sigma among 3000 random
    points, condition on a CL-min pseudo-observation at x*, and report
    sigma_after / sigma_before at x* and at distance r (a fraction of the
    side length) from x*, averaged over 8 random directions, and averaged
    over all 3000 points (far field).
    """
    lo, hi = bounds[:, 0], bounds[:, 1]
    w = float(np.mean(hi - lo))
    rng = np.random.RandomState(seed)
    P = lo + rng.rand(3000, len(lo)) * (hi - lo)
    m = ctor(bounds); m.fit(X.copy(), y.copy())
    _, s0 = m.predict(P)
    xs = P[int(np.argmax(s0))]
    m2 = ctor(bounds); m2.fit(X.copy(), y.copy())
    m2.condition_on(xs, float(np.min(y)))
    _, s1 = m2.predict(P)
    far = float(np.mean(s1 / np.maximum(s0, 1e-300)))
    dirs = rng.randn(8, len(lo))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    prof = []
    for r in RADII:
        pts = np.clip(xs + r * w * dirs, lo, hi)
        prof.append(float(np.mean(m2.predict(pts)[1] /
                                  np.maximum(m.predict(pts)[1], 1e-300))))
    return prof, far


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = SEEDS[:2] if args.quick else SEEDS
    setups = ([dict(name="Hartmann-6D", func=hartmann6, d=6,
                    bounds=np.array([[0.0, 1.0]] * 6), n_init=30)]
              if args.quick else
              [dict(name="Hartmann-6D", func=hartmann6, d=6,
                    bounds=np.array([[0.0, 1.0]] * 6), n_init=30),
               dict(name="Ackley-8D", func=ackley, d=8,
                    bounds=np.array([[-5.0, 5.0]] * 8), n_init=30)])

    print("=" * 78)
    print("EXPERIMENT 17 - Last-layer Bayesian NN in the SDD")
    print(f"  seeds={seeds}, q={Q}")
    print("=" * 78)

    models = [("GP (C1+C2)", lambda b: GPSurrogate(bounds=b)),   # inputs scaled to the unit cube
              ("Last-layer BNN (C1; C2?)", lambda b: LastLayerBNN()),
              ("Frozen NN (neither)", lambda b: NNEnsembleSurrogate())]

    for setup in setups:
        print(f"\n{'-'*70}\n  {setup['name']}\n{'-'*70}")
        dup = dup_threshold(setup["bounds"])
        for mname, ctor in models:
            mins, means, profs, fars = [], [], [], []
            for seed in seeds:
                X = latin_hypercube(setup["n_init"], setup["d"],
                                    setup["bounds"], seed=seed)
                y = setup["func"](X)

                s1 = ctor(setup["bounds"]); s1.fit(X.copy(), y.copy())
                b = sdd_batch(s1, y, Q, setup["bounds"], setup["d"])
                dd = pairwise_distances(b)
                mins.append(float(np.min(dd)))
                means.append(float(np.mean(dd)))

                prof, far = locality_profile(ctor, X, y, setup["bounds"], seed)
                profs.append(prof); fars.append(far)

            diverse = np.mean(mins) > dup
            pr = np.mean(profs, axis=0)
            print(f"  {mname:<26} SDD min={np.mean(mins):.4f}"
                  f"  diverse={'YES' if diverse else 'NO':<3}")
            print(f"  {'':<26} sigma_after/sigma_before at r x side from x*: "
                  + "  ".join(f"r={r:<4}:{v:6.3f}" for r, v in zip(RADII, pr))
                  + f"   far field:{np.mean(fars):6.3f}")
            print(f"  {'':<26} per-seed SDD mins: "
                  + "  ".join(f"{v:.4f}" for v in mins)
                  + f"   [{sum(v > dup for v in mins)}/{len(mins)} diverse]")

    print("\n" + "=" * 78)
    print("How to read: (C1) = ratio ~0 at r=0 (the update fired).")
    print("(C2) = ratio rising towards 1 with r and far field ~1 (the dent is")
    print("local).  Compare the per-seed SDD counts with these profiles;")
    print("exp17b repeats the SDD on 20 seeds under CL-min and KB.")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal wall-clock: {(time.time()-t0)/60:.1f} min")
