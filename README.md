# Efficient Conditioning: Why Pseudo-Observation Batch Bayesian Optimization Works, and When It Does Not

Code for all experiments in the paper. The results, logs and figures are not
included. Every script writes them to `results/` when it runs.

## Setup

The paper's results were computed with Python 3.12 and the package versions
pinned in `requirements.txt`.

```
pip install -r requirements.txt
python check_setup.py
python yahpo_data.py
```

`yahpo_data.py` downloads the YAHPO Gym surrogate models used by the LCBench
experiment and the batch-size study: three files for each of five scenarios,
about 48 MB in total. The files come from
<https://github.com/slds-lmu/yahpo_data> at a fixed commit (Apache License
2.0), and each one is checked against its SHA-256 hash. They are saved in
`data/`.

## Running the experiments

**Figure 1, Section 6 and Appendices C, D.1-D.3 and D.6.** One command runs
all of these experiments:

```
python run_all.py --resume
python analysis/make_tables.py
```

`run_all.py` runs several experiments at a time (`--jobs N`). Each experiment
runs in a single-threaded process. The time estimates in the script are
conservative and add up to 4.8 hours on one core. In the run for the paper,
the run times of these experiments added up to about 2.1 hours. Other
options:

- `--quick` makes a short test run of every script.
- `--only exp19,exp20` selects experiments.
- `--list` shows the plan.

Each experiment writes its output to `results/logs/<id>_full.txt` and its
figures to `results/figures/`. Experiments that store per-seed results write
them to `results/data/`. The two timing experiments (exp5, exp19) run last
and alone. `make_tables.py` recomputes the SDD tables, the noise comparison
and the duplicate counts from `results/data/`. For the experiments that
report only to their logs, it prints the relevant log lines.

**Batch-size study: Section 7 and Appendices B.1, B.2, D.4 and D.6.**

```
python exp24_batch_study.py --check
python exp24_batch_study.py --workers 12
python analysis/exp24_paper_numbers.py
```

- `--check` evaluates every problem at three fixed points and compares the
  values with stored ones. It also runs four evaluations of each main method.
- The full study has 18,240 runs. For the paper it took 20.9 hours with 18
  worker processes.
- Finished runs are skipped, so an interrupted study continues where it
  stopped. `--status` shows the progress, and `--quick` makes a short test
  run.
- When all runs are done, the study script calls
  `analysis/exp24_analysis.py`. This script computes the normalized regrets,
  bootstrap intervals, ranks and tests, and draws Figures 2 and 3 and a
  figure with the results of every problem.
- `exp24_paper_numbers.py` reads the summary file written by the analysis. It
  prints Tables 2 and 6, the across-problem tests with Holm's adjustment, and
  the per-problem statements of Section 7.2 and Appendix D.4.

**q-LogEI budget and real training runs: Appendices D.4 and D.5.**

```
python exp25_qlogei_budget.py run 7
python exp25_qlogei_budget.py summarize
python exp26_rf_real_training.py run 7
python exp26_rf_real_training.py summarize
```

The last argument of `run` is the number of worker processes. With seven
workers, the first execution of exp25 took 14 minutes and exp26 took 110
minutes. `validate` checks the components of each script. exp26 trains real
random forests on OpenML dataset 16 (mfeat-karhunen), which scikit-learn
downloads on the first run; this needs an internet connection and pandas.
These two experiments ran on other computers than the rest; exp25 was run
on two of them (Appendix E).

**GP of q-LogEI: Appendix D.4 (Table 7).**

```
python exp27_qlogei_paper_gp.py validate
python exp27_qlogei_paper_gp.py run 18
python exp27_analysis.py
```

- `validate` checks that the GPyTorch copy of the GP of Appendix B has the
  same posterior as the original on all 19 problems, and runs a few
  evaluations of every arm.
- The study has 2,090 runs: sequential EI, CL-min, KB and q-LogEI with
  BoTorch's GP and with the GP of Appendix B, on the 19 problems with
  q = 1, 4, 16 and seeds 0-9. For the paper it took 3.5 hours with 18 worker
  processes. Finished runs are skipped, and `status` shows the progress.
- `exp27_analysis.py` writes `results/EXP27_NUMBERS.md` (means, bootstrap
  intervals and the tests with Holm's adjustment), Table 7 as
  `results/exp27_table.tex`, a per-problem summary and two figures.

## Files

| File | Content | Paper |
|---|---|---|
| `utils.py` | Benchmarks, surrogates (GP, MQ-RBF, NN ensemble, RF), EI/UCB/PI, SDD starts, Lipschitz local penalization | all |
| `exp5_wallclock.py` | Time per batch step | App. D.6 |
| `exp7_retrained_and_ucb.py` | Refitted NN ensemble and RF; SDD with UCB | Sec. 6.1 |
| `exp13_naive_topq.py` | Naive top-q selection vs CL-min | Fig. 1, App. D.2 |
| `exp15_large_q_sdd.py` | SDD on Levy-10D with q = 10, 20, 50 | Sec. 6.1 |
| `exp16_lengthscale.py` | Suppression radius and batch separation vs lengthscale | App. D.2 |
| `exp17_lastlayer_bnn.py` | Uncertainty profile of the GP, last-layer BNN and NN ensemble | Table 4 |
| `exp17b_bnn_perseed.py` | Last-layer BNN, 20 seeds | Table 1 |
| `exp18_refit_lengthscale.py` | Hyperparameters refitted after every pseudo-observation | App. D.2 |
| `exp19_mqrbf_power_function.py` | MQ-RBF with heuristic and power-function uncertainty | Tables 1, 4; Sec. 6.1 |
| `exp20_sdd_start_sensitivity.py` | SDD under GP configurations A, B, C and two sets of starts | Table 1, App. C |
| `exp21_acquisition_monotonicity.py` | Duplicates by acquisition function, lie and incumbent rule | Table 5 |
| `exp22_noisy_hartmann.py` | Noisy Hartmann-6D | App. D.3 |
| `exp23_lcbench_hpo.py` | SDD on four LCBench tasks | App. D.1 |
| `exp24_batch_study.py` | Batch-size study | Sec. 7 |
| `exp24_problems.py` | The 19 problems of the batch-size study | Table 3 |
| `exp24_check_yahpo.py` | Compares the ONNX evaluation with the `yahpo_gym` package | Sec. 5, App. B.2 |
| `exp25_qlogei_budget.py` | q-LogEI with two optimizer budgets at q = 16 | App. D.4 |
| `exp26_rf_real_training.py` | Batch BO with real random-forest training runs | App. D.5 |
| `exp27_qlogei_paper_gp.py` | q-LogEI on the GP of Appendix B, with the arms of the batch-size study | App. D.4 |
| `exp27_analysis.py` | Analysis of the GP control | Table 7, App. D.4 |
| `yahpo_data.py` | Download and hash check of the YAHPO model files | |
| `run_all.py`, `check_setup.py` | Driver and installation check | |
| `analysis/` | `make_tables.py`, `exp24_analysis.py`, `exp24_paper_numbers.py` | |

`exp24_check_yahpo.py` compares our evaluation of the YAHPO models with the
`yahpo_gym` package. That package (version 1.0.2) requires `numpy<2`, so the
check should run in a separate environment:

```
pip install yahpo_gym==1.0.2
python exp24_check_yahpo.py data
```

## Reproducibility

Every run sets its own seeds. On the same computer with the same package
versions, a rerun therefore gives the same numbers, apart from the measured
times. On other hardware or with other versions, individual numbers can
change, because BO trajectories are sensitive to rounding.
