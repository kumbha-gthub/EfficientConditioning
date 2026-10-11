#!/usr/bin/env python3
"""
Structural Diversity Diagnostic on LCBench.

Objective: validation error (%) at epoch 52 of LCBench (Zimmer et al., 2021),
from the YAHPO Gym surrogate (Pfisterer et al., 2022) evaluated with ONNX
Runtime.  Seven hyperparameters, optimized in [0,1]^7:
    batch_size     16 .. 512    (log, rounded)
    learning_rate  1e-4 .. 1e-1 (log)
    momentum       0.1 .. 0.99
    weight_decay   1e-5 .. 0.1
    num_layers     1 .. 5       (floor(5u) + 1)
    max_units      64 .. 1024   (log, rounded)
    max_dropout    0 .. 1
Tasks (OpenML task id): 126025 adult, 167200 HIGGS, 168331 volkert,
189908 Fashion-MNIST.

SDD with the 20 fixed starts, n_init = 30, 5 seeds, q = 3 and 10, CL-min and
KB, for the GP, MQ-RBF (power), the last-layer BNN, the NN ensemble and the
random forest (the last two at q = 3 only).

Run yahpo_data.py first.  EXP_LCBENCH_TASKS=126025,189908 selects tasks;
EXP_QUICK=1 runs one task, one seed and q = 3.
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    GPSurrogate, MQRBFSurrogate, NNEnsembleSurrogate, RFSurrogate,
    optimize_acquisition, latin_hypercube, pairwise_distances, sdd_starts,
    dup_threshold, QUICK, save_json,
)
from exp17_lastlayer_bnn import LastLayerBNN
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "lcbench")
DIM = 7
BOUNDS = np.array([[0.0, 1.0]] * DIM)
TASKS = {"126025": "adult", "167200": "HIGGS", "168331": "volkert",
         "189908": "Fashion-MNIST"}
_sel = os.environ.get("EXP_LCBENCH_TASKS")
if _sel:
    TASKS = {k: v for k, v in TASKS.items() if k in _sel.split(",")}
# unique output names when tasks are run as separate parallel jobs
TAG = "" if not _sel else "_" + "-".join(sorted(TASKS))

SDD_SEEDS = [7, 21, 42, 73, 99]
SDD_QS = [3, 10]
if QUICK:
    TASKS = dict(list(TASKS.items())[:1])
    SDD_SEEDS, SDD_QS = SDD_SEEDS[:1], [3]


# ------------------------------------------------------------------
class LCBench:
    """Validation error (%) of LCBench at epoch 52 for one OpenML task."""

    _sess = None

    def __init__(self, task):
        if LCBench._sess is None:
            so = ort.SessionOptions()              # one thread per process
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            LCBench._sess = ort.InferenceSession(
                os.path.join(DATA, "model.onnx"), sess_options=so,
                providers=["CPUExecutionProvider"])
        with open(os.path.join(DATA, "encoding.json"), encoding="utf-8") as f:
            self.code = json.load(f)["OpenML_task_id"][task]
        self.task = task

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

    def _run(self, U):
        xc = self.decode(U)
        xk = np.full((len(xc), 1), self.code, dtype=np.int32)
        return self._sess.run(["output"], {"x_cat": xk, "x_cont": xc})[0]

    def __call__(self, U):                     # minimise validation error (%)
        return 100.0 - self._run(U)[:, 1].astype(float)

    def test_bal_acc(self, U):
        return self._run(U)[:, 5].astype(float)


# ------------------------------------------------------------------
def sdd(surr, y, q, lie, starts):
    f_best = float(np.min(y)); batch = []
    for _ in range(q):
        bx, be = None, -np.inf
        for s in starts:
            x, v = optimize_acquisition(surr, BOUNDS, f_best,
                                        fixed_start=np.array(s, float))
            if v > be:
                be, bx = v, x.copy()
        batch.append(bx.copy())
        yl = (float(np.min(y)) if lie == "cl_min"
              else float(surr.predict(bx.reshape(1, -1))[0][0]))
        surr.condition_on(bx, yl)
        f_best = min(f_best, yl)
    return np.array(batch)


def part_a():
    print("\n" + "=" * 84)
    print("PART A -- SDD on LCBench (20 fixed starts, n_init=30)")
    print("=" * 84)
    models = [("GP", lambda: GPSurrogate(bounds=BOUNDS)),
              ("MQ-RBF (power fn)", MQRBFSurrogate),
              ("Last-layer BNN", LastLayerBNN),
              ("NN ensemble", NNEnsembleSurrogate),
              ("Random forest", RFSurrogate)]
    starts, dup = sdd_starts(BOUNDS), dup_threshold(BOUNDS)
    out = {}
    for task, tname in TASKS.items():
        f = LCBench(task)
        print(f"\n  {tname} (task {task})")
        for q in SDD_QS:
            print(f"    q = {q}")
            for mname, ctor in models:
                # the non-responsive NN ensemble and RF are run at q = 3 only
                if q > 3 and mname in ("NN ensemble", "Random forest"):
                    continue
                for lie in ["cl_min", "kb"]:
                    mins = []
                    for seed in SDD_SEEDS:
                        X = latin_hypercube(30, DIM, BOUNDS, seed=seed)
                        y = f(X)
                        m = ctor(); m.fit(X.copy(), y.copy())
                        mins.append(float(pairwise_distances(
                            sdd(m, y, q, lie, starts)).min()))
                    nd = sum(v > dup for v in mins)
                    out[f"{task}|q{q}|{mname}|{lie}"] = mins
                    print(f"      {mname:<18} {lie:<7} "
                          f"{'  '.join(f'{v:.4f}' for v in mins):<46} "
                          f"{nd}/{len(mins)} diverse")
    return out


def main():
    t0 = time.time()
    print("=" * 84)
    print("EXPERIMENT 23 -- SDD on LCBench")
    print(f"  datasets: {', '.join(f'{v} ({k})' for k, v in TASKS.items())}")
    print("=" * 84)
    if not os.path.exists(os.path.join(DATA, "model.onnx")):
        sys.exit("data/lcbench is missing; run  python yahpo_data.py  first.")
    res = {"A": part_a()}
    save_json("exp23_lcbench_hpo_A" + TAG, res)
    print(f"\nTotal wall-clock: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
