#!/usr/bin/env python3
"""Problems of the batch-size study.

Every problem is a minimization problem with attributes name, dim, bounds
(dim x 2 array) and f_star (the known minimum, or None for the tuning
problems), and is called as p(X) -> 1-D array.

Synthetic functions (Surjanovic and Bingham, Virtual Library of Simulation
Experiments): Hartmann-6, Shekel-4 (m = 10), Ackley-8, Styblinski-Tang-8,
Rosenbrock-8, Levy-10.

Hyperparameter tuning: the instances of the YAHPO-SO v1.0 suite of YAHPO Gym
(Pfisterer et al., 2022) whose search space is a box once the categorical
hyperparameters are fixed.
  lcbench       167168, 189873, 189906   7-D, 100 - validation accuracy at epoch 52
  rbv2_xgboost  12, 1501, 16, 40499      10-D, booster = gbtree
  rbv2_ranger   16, 42                   4-D, splitrule = gini,
                                         respect.unordered.factors = ignore
  rbv2_glmnet   375, 458                 2-D
  rbv2_rpart    14, 40499                4-D
The nb301 instance (categorical search space) and the rbv2_super instances
(hierarchical search space) are not used.  For rbv2 the objective is
100 x (1 - accuracy) at full fidelity (trainsize = 1, repl = 10) with
num.impute.selected.cpo = impute.mean.  The ONNX models are evaluated
directly with ONNX Runtime, and inactive hyperparameters are encoded as in
yahpo_gym (0 for numbers, '#na#' for categories).  Run yahpo_data.py once to
download the model files; exp24_check_yahpo.py compares the outputs with
yahpo_gym.
"""

import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EXP24_VERSION = "exp24-v1"       # stamped into every result file


# ================================================================
# SYNTHETIC FUNCTIONS (minimization)
# ================================================================

def hartmann6(X):
    X = np.atleast_2d(X)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])
    inner = np.sum(A[None, :, :] * (X[:, None, :] - P[None, :, :]) ** 2, axis=2)
    return -np.sum(alpha[None, :] * np.exp(-inner), axis=1)


_SHEKEL_C = np.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                      [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
                      [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                      [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]], dtype=float)
_SHEKEL_B = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5], dtype=float)


def shekel4(X):
    X = np.atleast_2d(X)
    d2 = np.sum((X[:, :, None] - _SHEKEL_C[None, :, :]) ** 2, axis=1)   # n x m
    return -np.sum(1.0 / (d2 + _SHEKEL_B[None, :]), axis=1)


def ackley(X):
    X = np.atleast_2d(X)
    d = X.shape[1]
    t1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(X ** 2, axis=1) / d))
    t2 = -np.exp(np.sum(np.cos(2 * np.pi * X), axis=1) / d)
    return t1 + t2 + 20.0 + np.e


def styblinski_tang(X):
    X = np.atleast_2d(X)
    return 0.5 * np.sum(X ** 4 - 16 * X ** 2 + 5 * X, axis=1)


def rosenbrock(X):
    X = np.atleast_2d(X)
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (X[:, :-1] - 1) ** 2, axis=1)


def levy(X):
    X = np.atleast_2d(X)
    w = 1.0 + (X - 1.0) / 4.0
    t1 = np.sin(np.pi * w[:, 0]) ** 2
    t3 = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
    s = np.sum((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2), axis=1)
    return t1 + s + t3


class Synthetic:
    kind = "synthetic"

    def __init__(self, name, fn, dim, lo, hi, f_star):
        self.name, self.fn, self.dim = name, fn, dim
        self.bounds = np.array([[lo, hi]] * dim, dtype=float)
        self.f_star = f_star

    def __call__(self, X):
        return self.fn(np.atleast_2d(X)).astype(float)


# Known minima, refined by L-BFGS-B from the published minimizers
# (Hartmann-6: -3.32237 in the literature; Shekel-4: -10.5364).
# Styblinski-Tang: d * f1(-2.903534...), f1 = one-dimensional minimum.
_ST1 = 0.5 * ((-2.903534027771177) ** 4 - 16 * (-2.903534027771177) ** 2
              + 5 * (-2.903534027771177))
SYNTHETIC = {
    "Hartmann-6":         (hartmann6, 6, 0.0, 1.0, -3.322368011415515),
    "Shekel-4":           (shekel4, 4, 0.0, 10.0, -10.53644315348353),
    "Ackley-8":           (ackley, 8, -5.0, 5.0, 0.0),
    "Styblinski-Tang-8":  (styblinski_tang, 8, -5.0, 5.0, 8 * _ST1),
    "Rosenbrock-8":       (rosenbrock, 8, -5.0, 10.0, 0.0),
    "Levy-10":            (levy, 10, -10.0, 10.0, 0.0),
}


# ================================================================
# YAHPO GYM SURROGATE BENCHMARKS
# ================================================================

_SESSIONS = {}


def _session(scenario):
    """One ONNX Runtime session per scenario and process, single-threaded."""
    if scenario not in _SESSIONS:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        _SESSIONS[scenario] = ort.InferenceSession(
            os.path.join(DATA, scenario, "model.onnx"), sess_options=so,
            providers=["CPUExecutionProvider"])
    return _SESSIONS[scenario]


# Column order of the ONNX inputs, copied from yahpo_gym/benchmarks/rbv2.py.
_RBV2 = {
    "rbv2_xgboost": dict(
        cont=["nrounds", "eta", "gamma", "lambda", "alpha", "subsample", "max_depth",
              "min_child_weight", "colsample_bytree", "colsample_bylevel",
              "rate_drop", "skip_drop", "trainsize", "repl"],
        cat=["task_id", "booster", "num.impute.selected.cpo"],
        # (name, lower, upper, log, integer) -- ranges from config_space.json
        active=[("nrounds", 7, 2981, True, True),
                ("eta", 0.0009118819655545162, 1.0, True, False),
                ("gamma", 4.5399929762484854e-05, 7.38905609893065, True, False),
                ("lambda", 0.0009118819655545162, 1096.6331584284585, True, False),
                ("alpha", 0.0009118819655545162, 1096.6331584284585, True, False),
                ("subsample", 0.1, 1.0, False, False),
                ("max_depth", 1, 15, False, True),
                ("min_child_weight", 2.718281828459045, 148.4131591025766, True, False),
                ("colsample_bytree", 0.01, 1.0, False, False),
                ("colsample_bylevel", 0.01, 1.0, False, False)],
        fixed_cat={"booster": "gbtree", "num.impute.selected.cpo": "impute.mean"},
    ),
    "rbv2_ranger": dict(
        cont=["num.trees", "sample.fraction", "mtry.power", "min.node.size",
              "num.random.splits", "trainsize", "repl"],
        cat=["task_id", "respect.unordered.factors", "splitrule", "num.impute.selected.cpo"],
        active=[("num.trees", 1, 2000, False, True),
                ("sample.fraction", 0.1, 1.0, False, False),
                ("mtry.power", 0.0, 1.0, False, False),
                ("min.node.size", 1, 100, False, True)],
        fixed_cat={"respect.unordered.factors": "ignore", "splitrule": "gini",
                   "num.impute.selected.cpo": "impute.mean"},
    ),
    "rbv2_glmnet": dict(
        cont=["alpha", "s", "trainsize", "repl"],
        cat=["task_id", "num.impute.selected.cpo"],
        active=[("alpha", 0.0, 1.0, False, False),
                ("s", 0.0009118819655545162, 1096.6331584284585, True, False)],
        fixed_cat={"num.impute.selected.cpo": "impute.mean"},
    ),
    "rbv2_rpart": dict(
        cont=["cp", "maxdepth", "minbucket", "minsplit", "trainsize", "repl"],
        cat=["task_id", "num.impute.selected.cpo"],
        active=[("cp", 0.0009118819655545162, 1.0, True, False),
                ("maxdepth", 1, 30, False, True),
                ("minbucket", 1, 100, False, True),
                ("minsplit", 1, 100, False, True)],
        fixed_cat={"num.impute.selected.cpo": "impute.mean"},
    ),
}
_RBV2_FIDELITY = {"trainsize": 1.0, "repl": 10}


class RBV2:
    """100 x (1 - accuracy) of a YAHPO rbv2 model on one OpenML dataset."""
    kind = "yahpo"
    f_star = None

    def __init__(self, scenario, task_id, label):
        spec = _RBV2[scenario]
        self.scenario, self.task_id = scenario, str(task_id)
        self.name = label
        self.spec = spec
        self.dim = len(spec["active"])
        self.bounds = np.array([[0.0, 1.0]] * self.dim)
        with open(os.path.join(DATA, scenario, "encoding.json"), encoding="utf-8") as f:
            self.enc = json.load(f)
        cats = dict(spec["fixed_cat"], task_id=self.task_id)
        self.x_cat_row = np.array([self.enc[c][cats.get(c, "#na#")] for c in spec["cat"]],
                                  dtype=np.int32)

    def decode(self, U):
        """Unit cube -> dict of hyperparameter columns (numpy arrays)."""
        U = np.clip(np.atleast_2d(U), 0.0, 1.0)
        vals = {}
        for k, (name, lo, hi, log, integer) in enumerate(self.spec["active"]):
            u = U[:, k]
            v = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if log else lo + u * (hi - lo)
            if integer:
                v = np.clip(np.round(v), lo, hi)
            vals[name] = v
        return vals

    def _x_cont(self, U):
        vals = self.decode(U)
        n = len(next(iter(vals.values())))
        cols = []
        for c in self.spec["cont"]:
            if c in vals:
                cols.append(vals[c])
            elif c in _RBV2_FIDELITY:
                cols.append(np.full(n, _RBV2_FIDELITY[c], dtype=float))
            else:                                   # inactive -> 0, as yahpo_gym
                cols.append(np.zeros(n))
        return np.column_stack(cols).astype(np.float32)

    def outputs(self, U):
        xc = self._x_cont(U)
        xk = np.tile(self.x_cat_row, (len(xc), 1))
        sess = _session(self.scenario)
        names = [i.name for i in sess.get_inputs()]
        return sess.run([sess.get_outputs()[0].name], {names[0]: xk, names[1]: xc})[0]

    def __call__(self, U):
        return 100.0 * (1.0 - self.outputs(U)[:, 0].astype(float))   # y_names[0] = acc


class LCBench:
    """Validation error (%) of LCBench at epoch 52 (same decoding as exp23)."""
    kind = "yahpo"
    f_star = None
    dim = 7

    def __init__(self, task, label):
        self.task, self.name = str(task), label
        self.bounds = np.array([[0.0, 1.0]] * self.dim)
        with open(os.path.join(DATA, "lcbench", "encoding.json"), encoding="utf-8") as f:
            self.code = json.load(f)["OpenML_task_id"][self.task]

    @staticmethod
    def decode(U):
        U = np.clip(np.atleast_2d(U), 0.0, 1.0)
        logu = lambda u, a, b: np.exp(np.log(a) + u * (np.log(b) - np.log(a)))
        return np.column_stack([
            np.full(len(U), 52.0),                        # epoch (full fidelity)
            np.round(logu(U[:, 0], 16, 512)),             # batch_size
            logu(U[:, 1], 1e-4, 1e-1),                    # learning_rate
            0.1 + U[:, 2] * 0.89,                         # momentum
            1e-5 + U[:, 3] * (0.1 - 1e-5),                # weight_decay
            np.clip(np.floor(U[:, 4] * 5) + 1, 1, 5),     # num_layers
            np.round(logu(U[:, 5], 64, 1024)),            # max_units
            U[:, 6],                                      # max_dropout
        ]).astype(np.float32)

    def __call__(self, U):
        xc = self.decode(U)
        xk = np.full((len(xc), 1), self.code, dtype=np.int32)
        out = _session("lcbench").run(["output"], {"x_cat": xk, "x_cont": xc})[0]
        return 100.0 - out[:, 1].astype(float)


# ================================================================
# REGISTRY
# ================================================================

# YAHPO-SO v1.0 instances with a box search space (see module docstring).
LCBENCH_TASKS = ["167168", "189873", "189906"]
RBV2_TASKS = {"rbv2_xgboost": ["12", "1501", "16", "40499"],
              "rbv2_ranger": ["16", "42"],
              "rbv2_glmnet": ["375", "458"],
              "rbv2_rpart": ["14", "40499"]}
_LABEL = {"rbv2_xgboost": "XGBoost", "rbv2_ranger": "RandomForest",
          "rbv2_glmnet": "ElasticNet", "rbv2_rpart": "DecisionTree"}


def problem_names():
    names = list(SYNTHETIC)
    names += [f"LCBench-{t}" for t in LCBENCH_TASKS]
    for scen, tasks in RBV2_TASKS.items():
        names += [f"{_LABEL[scen]}-{t}" for t in tasks]
    return names


def get_problem(name):
    if name in SYNTHETIC:
        fn, d, lo, hi, fs = SYNTHETIC[name]
        return Synthetic(name, fn, d, lo, hi, fs)
    if name.startswith("LCBench-"):
        return LCBench(name[len("LCBench-"):], name)
    for scen, tasks in RBV2_TASKS.items():
        for t in tasks:
            if name == f"{_LABEL[scen]}-{t}":
                return RBV2(scen, t, name)
    raise KeyError(name)
