"""
Shared code for the experiments: benchmark functions, surrogate models,
acquisition functions, the fixed starting points of the Structural Diversity
Diagnostic (SDD) and small helpers.

Settings used throughout:
  * GP: inputs scaled to [0,1]^d, ARD Matern-5/2 kernel with lengthscales in
    [0.005, 2].  The output standardization is fixed when the hyperparameters
    are fitted, so conditioning on a pseudo-observation is exact GP
    conditioning with fixed hyperparameters.  EXP_GP_LS=legacy and
    EXP_GP_YNORM=refit select the alternative settings compared in exp20.
  * MQRBFSurrogate (= MQRBFPowerSurrogate) is the multiquadric interpolant
    with its power function as uncertainty, MQ-RBF (power) in the paper;
    MQRBFHeuristicSurrogate is MQ-RBF (heuristic).
  * SDD: 20 fixed Latin-hypercube starts; duplicate threshold 1e-3 times the
    side length of the domain.
  * Every run is seeded, so a rerun on the same machine gives the same
    numbers.
  * Figures and data go to results/figures and results/data next to this
    file (EXP_FIG_DIR, EXP_DATA_DIR).  EXP_QUICK=1 selects short test runs.
"""

import os
import json
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.spatial.distance import cdist, pdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import warnings

warnings.filterwarnings("ignore")

# ================================================================
# OUTPUT LOCATIONS + QUICK MODE
# ================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.environ.get("EXP_FIG_DIR", os.path.join(_HERE, "results", "figures"))
DATA_DIR = os.environ.get("EXP_DATA_DIR", os.path.join(_HERE, "results", "data"))
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
QUICK = os.environ.get("EXP_QUICK", "0") == "1"
# GP output standardization during conditioning: "frozen" (default, exact
# conditioning) or "refit" (re-standardize y after every pseudo-observation).
GP_YNORM = os.environ.get("EXP_GP_YNORM", "frozen").lower()
# ARD lengthscale bounds on the unit cube, [0.005, 2] as in TuRBO
# (Eriksson et al., 2019).  EXP_GP_LS=legacy selects [0.01, 100] with initial
# value 1.
if os.environ.get("EXP_GP_LS", "turbo").lower() == "legacy":
    GP_LS_MIN, GP_LS_MAX, GP_LS_INIT = 1e-2, 1e2, 1.0
else:
    GP_LS_MIN, GP_LS_MAX, GP_LS_INIT = 5e-3, 2.0, 0.5


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_json(name, obj):
    """Save raw per-seed results to results/data/<name>.json (for tables/tests)."""
    path = os.path.join(DATA_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=1)
    print(f"  [data saved -> {path}]")
    return path


# ================================================================
# BENCHMARK FUNCTIONS  (all for MINIMIZATION)
# ================================================================

def hartmann6(X):
    """Hartmann 6D. Domain [0,1]^6. Global min ~ -3.3224."""
    X = np.atleast_2d(X)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ])
    out = np.zeros(X.shape[0])
    for idx in range(X.shape[0]):
        s = 0.0
        for i in range(4):
            s += alpha[i] * np.exp(-np.sum(A[i] * (X[idx] - P[i]) ** 2))
        out[idx] = -s
    return out


def ackley(X):
    """Ackley (any dim). Domain [-5,5]^d. Global min = 0 at origin."""
    X = np.atleast_2d(X)
    a, b, c = 20, 0.2, 2 * np.pi
    d = X.shape[1]
    t1 = -a * np.exp(-b * np.sqrt(np.sum(X ** 2, axis=1) / d))
    t2 = -np.exp(np.sum(np.cos(c * X), axis=1) / d)
    return t1 + t2 + a + np.e


def levy(X):
    """Levy (any dim). Domain [-10,10]^d. Global min = 0 at (1,...,1)."""
    X = np.atleast_2d(X)
    d = X.shape[1]
    w = 1.0 + (X - 1.0) / 4.0
    term1 = np.sin(np.pi * w[:, 0]) ** 2
    term3 = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
    s = np.zeros(X.shape[0])
    for i in range(d - 1):
        s += (w[:, i] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, i] + 1) ** 2)
    return term1 + s + term3


# ================================================================
# ACQUISITION FUNCTION
# ================================================================

def expected_improvement(mu, sigma, f_best, xi=0.01):
    """Expected Improvement for minimization."""
    sigma = np.maximum(sigma, 1e-10)
    imp = f_best - mu - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei = np.where(sigma > 1e-10, ei, 0.0)
    return ei


def upper_confidence_bound(mu, sigma, f_best=None, beta=2.0):
    """UCB for minimization: -mu + sqrt(beta) * sigma, to be maximized."""
    sigma = np.maximum(sigma, 1e-10)
    return -mu + np.sqrt(beta) * sigma


def probability_of_improvement(mu, sigma, f_best, xi=0.01):
    """PI for minimization: Phi((f_best - xi - mu) / sigma).

    PI is not monotone in sigma: d PI / d sigma = -phi(Z) Z / sigma < 0
    wherever mu < f_best - xi (Z > 0).  Used by exp21.
    """
    sigma = np.maximum(sigma, 1e-10)
    Z = (f_best - xi - mu) / sigma
    return norm.cdf(Z)


# ================================================================
# SURROGATE WRAPPERS
# ================================================================

class GPSurrogate:
    """
    Gaussian process surrogate.
    After a fit with hyperparameter optimization, new points are absorbed by
    re-solving the linear system with fixed hyperparameters (efficient
    conditioning).
    """

    def __init__(self, noise=1e-6, bounds=None, ls=None, ynorm=None):
        """ls    : 'turbo' ([0.005, 2], init 0.5) or 'legacy' ([0.01, 100],
                   init 1.0); None -> module default (EXP_GP_LS).
        ynorm : 'frozen' or 'refit'; None -> module default (EXP_GP_YNORM).
        exp20 uses ls='legacy', ynorm='refit' (with and without bounds) for
        configurations A and B of the paper's sensitivity study."""
        self.noise = noise
        self.kernel_ = None
        self._ynorm = None
        if ls is None:
            self._ls = (GP_LS_MIN, GP_LS_MAX, GP_LS_INIT)
        elif ls == "legacy":
            self._ls = (1e-2, 1e2, 1.0)
        elif ls == "turbo":
            self._ls = (5e-3, 2.0, 0.5)
        else:
            raise ValueError(ls)
        self._ynorm_mode = (ynorm or GP_YNORM).lower()
        self.model = None
        self.X_train = None
        self.y_train = None
        # optional rescaling of the inputs to the unit cube
        if bounds is None:
            self._lo, self._w = None, None
        else:
            b = np.asarray(bounds, dtype=float)
            self._lo, self._w = b[:, 0], b[:, 1] - b[:, 0]

    def _s(self, X):
        X = np.atleast_2d(X)
        return X if self._lo is None else (X - self._lo) / self._w

    def fit(self, X, y, optimize=True):
        """optimize=True : fit the hyperparameters by marginal likelihood.
        optimize=False: condition on (X, y) with all hyperparameters fixed,
        that is, the kernel and the output standardization (mean, std) of the
        last optimize=True fit.  Re-standardizing y here would change the
        prior mean and signal variance after every pseudo-observation.
        EXP_GP_YNORM=refit does exactly that, for comparison."""
        self.X_train = X.copy()
        self.y_train = y.copy()
        d = X.shape[1]
        X = self._s(X)
        y = np.asarray(y, dtype=float)
        if optimize:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                nu=2.5,
                length_scale=np.full(d, self._ls[2]),
                length_scale_bounds=(self._ls[0], self._ls[1]),
            )
            m, sd = float(np.mean(y)), float(np.std(y))
            self._ynorm = (m, sd if sd > 0 else 1.0)
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.noise,
                n_restarts_optimizer=5,
                normalize_y=False,
                random_state=0,
            )
        else:
            if self.kernel_ is None:
                raise ValueError("optimize=False needs fitted hyper-parameters")
            if (self._ynorm_mode == "refit"
                    or getattr(self.kernel_, "_ynorm", None) is None):
                m, sd = float(np.mean(y)), float(np.std(y))
                self._ynorm = (m, sd if sd > 0 else 1.0)
            else:
                self._ynorm = self.kernel_._ynorm      # frozen with the kernel
            self.model = GaussianProcessRegressor(
                kernel=self.kernel_,
                alpha=self.noise,
                optimizer=None,
                normalize_y=False,
            )
        m, sd = self._ynorm
        self.model.fit(X, (y - m) / sd)
        if optimize:
            self.kernel_ = self.model.kernel_
            # store the output standardization with the kernel, so that
            # `other.kernel_ = gp.kernel_` copies both
            self.kernel_._ynorm = self._ynorm

    def predict(self, X):
        X = self._s(np.atleast_2d(X))
        mu, sigma = self.model.predict(X, return_std=True)
        m, sd = self._ynorm
        return mu * sd + m, sigma * sd

    def condition_on(self, x_new, y_new):
        """Add a pseudo-observation with fixed hyperparameters."""
        self.X_train = np.vstack([self.X_train, np.atleast_2d(x_new)])
        self.y_train = np.append(self.y_train, y_new)
        self.fit(self.X_train, self.y_train, optimize=False)


class NNEnsembleSurrogate:
    """
    Neural-network ensemble.
    Predictions do not change when data is appended without retraining.
    """

    def __init__(self, n_ensemble=10, hidden=(64, 64)):
        self.n_ensemble = n_ensemble
        self.hidden = hidden
        self.models = []
        self.X_train = None
        self.y_train = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, X, y, retrain=True):
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.y_mean = np.mean(y)
        self.y_std = max(np.std(y), 1e-8)
        if retrain:
            y_norm = (y - self.y_mean) / self.y_std
            self.models = []
            for i in range(self.n_ensemble):
                m = MLPRegressor(
                    hidden_layer_sizes=self.hidden,
                    max_iter=2000,
                    random_state=i,
                    learning_rate="adaptive",
                    early_stopping=(len(X) > 15),
                    validation_fraction=0.15 if len(X) > 15 else 0.0,
                )
                m.fit(X, y_norm)
                self.models.append(m)

    def predict(self, X):
        X = np.atleast_2d(X)
        preds = np.array(
            [m.predict(X) * self.y_std + self.y_mean for m in self.models]
        )
        return np.mean(preds, axis=0), np.std(preds, axis=0) + 1e-6

    def condition_on(self, x_new, y_new):
        """Append data without retraining; predictions are unchanged."""
        self.X_train = np.vstack([self.X_train, np.atleast_2d(x_new)])
        self.y_train = np.append(self.y_train, y_new)


class RFSurrogate:
    """
    Random forest surrogate.
    Predictions do not change when data is appended without rebuilding.
    """

    def __init__(self, n_estimators=200):
        self.n_estimators = n_estimators
        self.model = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y, retrain=True):
        self.X_train = X.copy()
        self.y_train = y.copy()
        if retrain:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=42,
                min_samples_leaf=2,
            )
            self.model.fit(X, y)

    def predict(self, X):
        X = np.atleast_2d(X)
        preds = np.array([t.predict(X) for t in self.model.estimators_])
        return np.mean(preds, axis=0), np.std(preds, axis=0) + 1e-6

    def condition_on(self, x_new, y_new):
        """Append data without retraining."""
        self.X_train = np.vstack([self.X_train, np.atleast_2d(x_new)])
        self.y_train = np.append(self.y_train, y_new)


class MQRBFHeuristicSurrogate:
    """
    Multiquadric RBF interpolant with a heuristic uncertainty, MQ-RBF
    (heuristic) in the paper.

    Centres are placed at every training point, so a new observation adds a
    centre and the linear system is re-solved with the spread fixed (C1).
    The uncertainty is sigma(x) = s_y / (1 + phi(x)^T Phi^-1 phi(x)).  It is
    not the power function: it equals about s_y / 2 at the centres and does
    not vanish at a conditioned point, so C2 fails.
    """

    def __init__(self, spread=None, reg=1e-4):
        self.spread = spread          # set from data on first fit
        self.reg = reg
        self.X_train = None
        self.y_train = None
        self.weights = None
        self._fixed_spread = None     # locked after first fit

    # -- kernel ----------------------------------------------------
    def _phi(self, X1, X2):
        """Multiquadric: sqrt(1 + r^2 / spread^2)."""
        r2 = cdist(X1, X2, "sqeuclidean")
        return np.sqrt(1.0 + r2 / self._fixed_spread ** 2)

    # -- fit / predict / condition ---------------------------------
    def fit(self, X, y, optimize=True):
        self.X_train = X.copy()
        self.y_train = y.copy()
        n = len(X)

        if optimize or self._fixed_spread is None:
            self._fixed_spread = float(np.median(pdist(X))) if n > 1 else 1.0
            self._fixed_spread = max(self._fixed_spread, 0.01)

        Phi = self._phi(X, X) + self.reg * np.eye(n)
        self.weights = np.linalg.solve(Phi, y)

    def predict(self, X):
        X = np.atleast_2d(X)
        phi_star = self._phi(X, self.X_train)
        mu = phi_star @ self.weights

        # heuristic uncertainty from the interpolation matrix
        n = len(self.X_train)
        Phi_reg = self._phi(self.X_train, self.X_train) + self.reg * np.eye(n)
        Phi_inv = np.linalg.inv(Phi_reg)

        # quad(x) = phi(x)^T Phi^{-1} phi(x)
        quad = np.einsum("ij,jk,ik->i", phi_star, Phi_inv, phi_star)
        # sigma = s_y / (1 + quad) stays positive and smooth
        y_scale = max(np.ptp(self.y_train), np.std(self.y_train), 1e-6)
        sigma = y_scale / (1.0 + quad)
        sigma = np.maximum(sigma, 1e-8)
        return mu, sigma

    def condition_on(self, x_new, y_new):
        """Add a centre and re-solve with the spread fixed."""
        self.X_train = np.vstack([self.X_train, np.atleast_2d(x_new)])
        self.y_train = np.append(self.y_train, y_new)
        self.fit(self.X_train, self.y_train, optimize=False)


class MQRBFPowerSurrogate:
    """
    Multiquadric RBF interpolant with its power function as the uncertainty,
    MQ-RBF (power) in the paper.

    The multiquadric phi(r) = sqrt(1 + r^2/eps^2) is conditionally negative
    definite of order 1, so K = -phi is conditionally positive definite of
    order 1 and the interpolant is augmented with a constant:
        s(x) = sum_i c_i K(x, x_i) + d,   sum_i c_i = 0.
    With the augmented matrix A = [[K, 1], [1^T, 0]] and b(x) = [k(x); 1],
    the power function is
        P^2(x) = K(x, x) - b(x)^T A^{-1} b(x)          (Wendland 2005, Ch. 11)
    which is zero at the centres and grows away from them.  sigma(x) is
    P(x) times a scale fixed at the first fit, s^2 = c^T K c / n (the
    native-space semi-norm per point).  Spread eps and scale are frozen after
    the first fit, like the GP hyper-parameters, so condition_on() is a pure
    closed-form re-solve (C1).
    """

    def __init__(self, reg=1e-8):
        self.reg = reg
        self._fixed_spread = None
        self.scale = None
        self.X_train = None
        self.y_train = None

    def _K(self, X1, X2):
        r2 = cdist(X1, X2, "sqeuclidean")
        return -np.sqrt(1.0 + r2 / self._fixed_spread ** 2)

    def _solve(self):
        n = len(self.X_train)
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = self._K(self.X_train, self.X_train) - self.reg * np.eye(n)
        A[:n, n] = 1.0
        A[n, :n] = 1.0
        self._Ainv = np.linalg.inv(A)
        sol = self._Ainv @ np.append(self.y_train, 0.0)
        self.c, self.d = sol[:n], sol[n]

    def fit(self, X, y, optimize=True):
        self.X_train = X.copy()
        self.y_train = y.copy()
        if optimize or self._fixed_spread is None:
            n = len(X)
            self._fixed_spread = float(np.median(pdist(X))) if n > 1 else 1.0
            self._fixed_spread = max(self._fixed_spread, 0.01)
        self._solve()
        if optimize or self.scale is None:
            Kc = self._K(self.X_train, self.X_train) @ self.c
            self.scale = float(np.sqrt(max(float(self.c @ Kc), 1e-12) / len(y)))

    def power_function_sq(self, X):
        X = np.atleast_2d(X)
        b = np.hstack([self._K(X, self.X_train), np.ones((len(X), 1))])
        quad = np.einsum("ij,jk,ik->i", b, self._Ainv, b)
        return -1.0 - quad            # K(x, x) = -phi(0) = -1

    def predict(self, X):
        X = np.atleast_2d(X)
        mu = self._K(X, self.X_train) @ self.c + self.d
        p2 = np.maximum(self.power_function_sq(X), 1e-16)
        return mu, self.scale * np.sqrt(p2)

    def condition_on(self, x_new, y_new):
        """Add a centre and re-solve; spread and scale stay fixed."""
        self.X_train = np.vstack([self.X_train, np.atleast_2d(x_new)])
        self.y_train = np.append(self.y_train, y_new)
        self._solve()


# "MQ-RBF" in the experiments is the power-function model.
MQRBFSurrogate = MQRBFPowerSurrogate


# ================================================================
# SDD STARTING POINTS
# ================================================================
SDD_STARTS = os.environ.get("EXP_SDD_STARTS", "lhs20")   # or "diag3"


def sdd_starts(bounds, kind=None):
    """Fixed optimizer starts for the Structural Diversity Diagnostic.
    'lhs20' (default): 20 Latin-hypercube points (seed 12345), the same at
    every conditioning step.  'diag3': 0.2, 0.5 and 0.8 along the diagonal
    of the domain (used in exp20)."""
    kind = kind or SDD_STARTS
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    if kind == "diag3":
        return [lo + f * (hi - lo) for f in (0.2, 0.5, 0.8)]
    return list(latin_hypercube(20, bounds.shape[0], bounds, seed=12345))


def dup_threshold(bounds):
    """Duplicate threshold = 1e-3 x domain side length (1e-3 on [0,1]^d)."""
    bounds = np.asarray(bounds, dtype=float)
    return 1e-3 * float(np.mean(bounds[:, 1] - bounds[:, 0]))


# ================================================================
# LIPSCHITZ LOCAL PENALIZATION  (Gonzalez, Dai, Hennig & Lawrence, 2016)
# ================================================================

def _grad_mean_norm(surrogate, X, h):
    """Finite-difference norm of the posterior-mean gradient at rows of X."""
    X = np.atleast_2d(X)
    n, d = X.shape
    g = np.zeros((n, d))
    for i in range(d):
        e = np.zeros(d)
        e[i] = h[i]
        mp, _ = surrogate.predict(X + e)
        mm, _ = surrogate.predict(X - e)
        g[:, i] = (mp - mm) / (2.0 * h[i])
    return np.linalg.norm(g, axis=1)


def estimate_lipschitz(surrogate, bounds, n_samples=1000, seed=0):
    """L = max_x ||grad mu(x)||, as in Gonzalez et al. (2016): evaluate the
    posterior-mean gradient norm on uniform samples, then refine around the
    five largest values with a vectorised local search (500 Gaussian
    perturbations, scale 5% of the domain width).  Falls back to L = 10
    (GPyOpt's default) if the estimate is degenerate."""
    rng = np.random.RandomState(seed)
    dim = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    width = hi - lo
    h = 1e-4 * width
    S = rng.uniform(lo, hi, size=(n_samples, dim))
    gn = _grad_mean_norm(surrogate, S, h)
    top = S[np.argsort(gn)[-5:]]
    P = (top[:, None, :] + 0.05 * width * rng.randn(5, 100, dim)).reshape(-1, dim)
    P = np.clip(P, lo, hi)
    L = float(max(np.max(gn), np.max(_grad_mean_norm(surrogate, P, h))))
    if not np.isfinite(L) or L < 1e-7:
        L = 10.0
    return L


def optimize_acq_lp_lipschitz(surrogate, bounds, f_best, batch_so_far,
                              L, M, n_restarts=10):
    """Maximise EI(x) * prod_j phi(x; x_j) with the Lipschitz penalizer
        phi(x; x_j) = Phi( (L ||x - x_j|| - (mu(x_j) - M)) / sigma(x_j) ),
    the minimisation form of Gonzalez et al. (2016, Eq. 4); M = min observed y.
    The GP is not conditioned on the batch points."""
    if len(batch_so_far):
        B = np.atleast_2d(np.array(batch_so_far))
        mu_b, sd_b = surrogate.predict(B)
        sd_b = np.maximum(sd_b, 1e-8)
        r_b = (mu_b - M)
    else:
        B = None

    def neg_penalised_acq(x):
        x2d = x.reshape(1, -1)
        mu, sigma = surrogate.predict(x2d)
        val = float(expected_improvement(mu, sigma, f_best)[0])
        if B is not None:
            dist = np.linalg.norm(B - x2d, axis=1)
            val *= float(np.prod(norm.cdf((L * dist - r_b) / sd_b)))
        return -val

    starts = [np.random.uniform(bounds[:, 0], bounds[:, 1])
              for _ in range(n_restarts)]
    best_x, best_val = None, np.inf
    for x0 in starts:
        try:
            res = minimize(neg_penalised_acq, x0,
                           bounds=list(bounds), method="L-BFGS-B")
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue
    return best_x, -best_val


# ================================================================
# HELPERS
# ================================================================

def latin_hypercube(n, d, bounds, seed=42):
    """Latin Hypercube Sampling within given bounds."""
    rng = np.random.RandomState(seed)
    result = np.zeros((n, d))
    for i in range(d):
        perm = rng.permutation(n)
        result[:, i] = (perm + rng.uniform(size=n)) / n
        result[:, i] = bounds[i, 0] + result[:, i] * (bounds[i, 1] - bounds[i, 0])
    return result


def optimize_acquisition(surrogate, bounds, f_best,
                         fixed_start=None, n_restarts=10,
                         acq_func="ei", acq_kwargs=None):
    """Maximise an acquisition function via L-BFGS-B.

    acq_func : 'ei', 'ucb' or 'pi'
    """
    kw = acq_kwargs or {}

    def neg_acq(x):
        mu, sigma = surrogate.predict(x.reshape(1, -1))
        if acq_func == "ucb":
            return -upper_confidence_bound(mu, sigma, f_best, **kw)[0]
        if acq_func == "pi":
            return -probability_of_improvement(mu, sigma, f_best, **kw)[0]
        return -expected_improvement(mu, sigma, f_best, **kw)[0]

    if fixed_start is not None:
        starts = [np.array(fixed_start, dtype=float)]
    else:
        starts = [
            np.random.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(n_restarts)
        ]

    best_x, best_val = None, np.inf
    for x0 in starts:
        try:
            res = minimize(neg_acq, x0, bounds=list(bounds), method="L-BFGS-B")
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue
    return best_x, -best_val


def get_gp_lengthscale(gp_surrogate):
    """Extract lengthscale from a fitted GPSurrogate's kernel."""
    try:
        k = gp_surrogate.model.kernel_
        w = getattr(gp_surrogate, "_w", None)
        scale = 1.0 if w is None else w          # in the original input units
        if hasattr(k, 'k2') and hasattr(k.k2, 'length_scale'):
            return np.atleast_1d(k.k2.length_scale) * scale
        if hasattr(k, 'length_scale'):
            return np.atleast_1d(k.length_scale) * scale
    except Exception:
        pass
    return np.array([1.0])


def pairwise_distances(batch):
    """All pairwise Euclidean distances in a batch (returns array)."""
    batch = np.atleast_2d(batch)
    n = len(batch)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(batch[i] - batch[j]))
    return np.array(dists)
