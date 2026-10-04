"""Tests for compute-cost instrumentation and the efficiency Pareto analysis."""
import numpy as np
import pandas as pd
import pytest

from qspr_bench.analysis.efficiency import efficiency_table, pareto_frontier
from qspr_bench.config import RunConfig
from qspr_bench.data import DatasetSpec
from qspr_bench.engine import COST_COLUMNS, run_benchmark


# ---- Pareto frontier ----

def test_pareto_frontier_higher_better():
    # (cost, score): (1,0.9) cheap-ok, (2,0.95) more-expensive-better, (3,0.8) dominated,
    # (1.5,0.85) dominated by (1,0.9)
    cost = [1.0, 2.0, 3.0, 1.5]
    score = [0.9, 0.95, 0.8, 0.85]
    mask = pareto_frontier(cost, score, higher_is_better=True)
    assert list(mask) == [True, True, False, False]


def test_pareto_frontier_lower_better():
    # higher_is_better=False: want low cost AND low score
    cost = [1.0, 2.0]
    score = [0.5, 0.4]
    mask = pareto_frontier(cost, score, higher_is_better=False)
    assert list(mask) == [True, True]  # tradeoff: each is best on one axis


# ---- engine cost instrumentation ----

def _cfg(spec, out, measure):
    return RunConfig(name="t", datasets=[spec], models=["RF"], feature_sets=["Desc"],
                     splits=["random"], test_frac=0.3, cv_folds=2, seeds=[42],
                     fp_bits=128, output_dir=out, measure_cost=measure)


def test_cost_columns_present_only_when_enabled(regression_csv, tmp_path):
    spec = DatasetSpec("SynthReg", regression_csv, "smiles", "y", "regression")
    off = run_benchmark(_cfg(spec, tmp_path / "off", False))
    for c in COST_COLUMNS:
        assert c not in off.columns
    on = run_benchmark(_cfg(spec, tmp_path / "on", True))
    for c in COST_COLUMNS:
        assert c in on.columns
    row = on.iloc[0]
    assert row["refit_seconds"] > 0
    assert np.isfinite(row["predict_seconds"])
    assert row["peak_mem_mb"] > 0


def test_nested_cost_is_fold_mean(regression_csv, tmp_path):
    spec = DatasetSpec("SynthReg", regression_csv, "smiles", "y", "regression")
    cfg = RunConfig(name="t", datasets=[spec], models=["RF"], feature_sets=["Desc"],
                    splits=["random"], cv_folds=2, seeds=[42], fp_bits=128,
                    output_dir=tmp_path / "o", evaluation="nested_cv", outer_folds=3,
                    measure_cost=True)
    df = run_benchmark(cfg)
    assert "refit_seconds" in df.columns
    assert df.iloc[0]["refit_seconds"] > 0


# ---- efficiency table ----

def test_efficiency_table_flags_dominated_model():
    # fast+accurate should be optimal; slow+worse should not be
    results = pd.DataFrame([
        {"dataset": "D", "split": "random", "feature_set": "Desc", "model": "fast",
         "test_R2": 0.90, "refit_seconds": 0.1, "peak_mem_mb": 5.0},
        {"dataset": "D", "split": "random", "feature_set": "Desc", "model": "slow_worse",
         "test_R2": 0.80, "refit_seconds": 2.0, "peak_mem_mb": 50.0},
        {"dataset": "D", "split": "random", "feature_set": "Desc", "model": "slow_better",
         "test_R2": 0.93, "refit_seconds": 3.0, "peak_mem_mb": 80.0},
    ])
    meta = pd.DataFrame([{"dataset": "D", "task": "regression"}])
    table = efficiency_table(results, meta, cost_col="refit_seconds")
    opt = dict(zip(table["model"], table["is_pareto_optimal"]))
    assert opt["fast"] is True
    assert opt["slow_better"] is True   # best accuracy -> on frontier
    assert opt["slow_worse"] is False   # dominated by fast on both axes


def test_efficiency_table_requires_cost_column():
    results = pd.DataFrame([{"dataset": "D", "split": "random", "feature_set": "Desc",
                             "model": "m", "test_R2": 0.9}])
    meta = pd.DataFrame([{"dataset": "D", "task": "regression"}])
    with pytest.raises(ValueError):
        efficiency_table(results, meta, cost_col="refit_seconds")


def test_classification_uses_roc_auc():
    results = pd.DataFrame([
        {"dataset": "C", "split": "random", "feature_set": "Desc", "model": "a",
         "test_ROC_AUC": 0.85, "refit_seconds": 0.1, "peak_mem_mb": 5.0},
        {"dataset": "C", "split": "random", "feature_set": "Desc", "model": "b",
         "test_ROC_AUC": 0.80, "refit_seconds": 1.0, "peak_mem_mb": 5.0},
    ])
    meta = pd.DataFrame([{"dataset": "C", "task": "classification"}])
    table = efficiency_table(results, meta, cost_col="refit_seconds")
    assert set(table["metric"]) == {"ROC_AUC"}
    opt = dict(zip(table["model"], table["is_pareto_optimal"]))
    assert opt["a"] is True and opt["b"] is False
