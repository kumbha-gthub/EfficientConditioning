#!/usr/bin/env python3
"""Compare the ONNX evaluation of exp24_problems.py with yahpo_gym.

Evaluates random configurations with both implementations and prints the
largest absolute difference: 100 per instance for the ten rbv2 and the three
LCBench instances of the batch-size study, and 50 per task (200 in total) for
the four LCBench tasks of exp23, which uses the same decoding.  Needs the
yahpo_gym package and a yahpo_data folder (for example the data folder
filled by yahpo_data.py).

Usage: python exp24_check_yahpo.py path/to/yahpo_data
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp24_problems import RBV2, _RBV2, _RBV2_FIDELITY, LCBench, LCBENCH_TASKS
from yahpo_gym import local_config, benchmark_set

local_config.init_config()
local_config.set_data_path(sys.argv[1])
from exp24_problems import RBV2_TASKS as tasks
rng = np.random.default_rng(0)
worst = 0.0
for scen, tids in tasks.items():
    bench = benchmark_set.BenchmarkSet(scen, check=False)
    for tid in tids:
        bench.set_instance(tid)
        p = RBV2(scen, tid, "check")
        U = rng.uniform(size=(100, p.dim))
        ours = p.outputs(U)
        vals = p.decode(U)
        cfgs = []
        for i in range(len(U)):
            c = {"task_id": tid}
            c.update(_RBV2[scen]["fixed_cat"])
            c.update(_RBV2_FIDELITY)
            for (name, lo, hi, log, integer) in _RBV2[scen]["active"]:
                v = float(vals[name][i])
                c[name] = int(v) if integer else v
            cfgs.append(c)
        res = bench.objective_function(cfgs, multithread=False)
        theirs = np.array([[r[k] for k in bench.config.y_names] for r in res], dtype=np.float32)
        diff = float(np.max(np.abs(ours - theirs)))
        worst = max(worst, diff)
        print(f"{scen:13s} task {tid:>6s}: max abs difference over all outputs = {diff:.3g}")
bench = benchmark_set.BenchmarkSet("lcbench", check=False)
names = ["epoch", "batch_size", "learning_rate", "momentum", "weight_decay",
         "num_layers", "max_units", "max_dropout"]
EXP23_TASKS = ["126025", "167200", "168331", "189908"]
for tid, n_cfg in [(t, 100) for t in LCBENCH_TASKS] + [(t, 50) for t in EXP23_TASKS]:
    bench.set_instance(tid)
    p = LCBench(tid, "check")
    U = rng.uniform(size=(n_cfg, p.dim))
    ours = p(U)
    X = LCBench.decode(U)
    cfgs = []
    for i in range(len(U)):
        c = {"OpenML_task_id": tid}
        for j, n in enumerate(names):
            v = float(X[i, j])
            c[n] = int(round(v)) if n in ("epoch", "batch_size", "num_layers", "max_units") else v
        cfgs.append(c)
    res = bench.objective_function(cfgs, multithread=False)
    theirs = 100.0 - np.array([r["val_accuracy"] for r in res], dtype=float)
    diff = float(np.max(np.abs(ours - theirs)))
    worst = max(worst, diff)
    print(f"{'lcbench':13s} task {tid:>6s}: max abs difference of the objective = {diff:.3g}"
          f"  ({n_cfg} configurations)")
print("WORST", worst)
