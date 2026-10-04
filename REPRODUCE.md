# Reproducing the qspr_bench results

Every table and analysis in the paper is regenerated from the code and configs in this
repository. The steps below take a clean machine to the full result set.

## 1. Environment

Requires **Python 3.12** (tested on 3.12.4). Create a fresh environment, then:

```bash
pip install -r requirements-lock.txt
pip install -e .
```

`requirements-lock.txt` pins the exact tested versions. Two constraints are load-bearing:
`numpy<2` (rdkit/pandas are compiled against numpy 1.x) and `shap<0.46` (0.46+ needs
numpy>=2). Relaxing either breaks the environment.

## 2. Confirm the install

```bash
pytest -q      # expect: 35 passed
```

A passing suite confirms the featurization, splits, engine, baselines, and analysis all
work in your environment. The CLI is available as `qspr-bench` (or `python -m
qspr_bench.cli` if the console script is not on PATH).

## 3. Data

Regression datasets (ESOL, FreeSolv, Lipophilicity) ship in `data/`. Classification
datasets are downloaded on demand (see `data/README.md` for provenance):

```bash
qspr-bench fetch BBBP BACE
```

## 4. Regenerate results

Each command writes `results.csv` (and `dataset_meta.csv`, plus `predictions.csv` when
predictions are saved) into its `output_dir`.

```bash
# In-distribution holdout benchmark (regression)
qspr-bench run configs/regression_default.yaml

# Unbiased nested cross-validation (regression) + subgroup analysis
qspr-bench run configs/nested_cv_regression.yaml
qspr-bench analyze subgroups results/nested_cv_regression

# Out-of-distribution splits (random vs nn_distance vs target extrapolation)
qspr-bench run configs/ood_regression.yaml

# Compute-efficiency vs accuracy, then the Pareto frontier
qspr-bench run configs/efficiency_regression.yaml --measure-cost
qspr-bench analyze efficiency results/efficiency_regression

# Classification benchmark (after fetching BBBP/BACE)
qspr-bench run configs/classification_default.yaml
```

Optional deep-learning baselines (install the extras first):

```bash
pip install -e ".[gnn]"       # GIN, MPNN
pip install -e ".[chemprop]"  # ChempropDMPNN
qspr-bench run configs/regression_default.yaml --dataset FreeSolv \
  --models RF,GIN,ChempropDMPNN --splits random --features Desc \
  --output-dir results/gnn_compare
```

## 5. Determinism and known caveats

- **Seeds** are fixed in every config (`seeds`, `random_state`), so classical-model and
  split results are deterministic on a given platform.
- **Target extrapolation** is seed-free (deterministic sort), so one seed suffices.
- **GNN / Chemprop** training uses PyTorch; results are reproducible up to the usual
  floating-point nondeterminism across hardware/BLAS/driver versions, not bit-exact.
- **KNN** is excluded from the shipped configs: on some Windows/numpy environments it
  triggers a `threadpoolctl` bug during parallel distance computation (unrelated to this
  package). It can still be requested explicitly on platforms where it works.

## 6. Outputs

`results/<run_name>/` holds `results.csv` (one row per
dataset x split x feature_set x model x seed), `dataset_meta.csv`, and — for nested CV or
when `save_predictions` is set — `predictions.csv`. Analysis commands add `subgroups.csv`
and `efficiency.csv` to the same directory.
