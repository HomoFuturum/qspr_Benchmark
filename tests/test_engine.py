"""End-to-end engine smoke tests on tiny synthetic datasets."""
import math

import pandas as pd

from qspr_bench.config import RunConfig
from qspr_bench.data import DatasetSpec
from qspr_bench.engine import run_benchmark


def _run(spec, models, tmp_out):
    cfg = RunConfig(
        name="test",
        datasets=[spec],
        models=models,
        feature_sets=["Desc"],
        splits=["random"],
        test_frac=0.3,
        cv_folds=2,
        seeds=[42],
        fp_bits=128,
        output_dir=tmp_out,
    )
    return run_benchmark(cfg)


def test_engine_regression(regression_csv, tmp_path):
    spec = DatasetSpec("SynthReg", regression_csv, "smiles", "y", "regression")
    out = tmp_path / "out_reg"
    df = _run(spec, ["RF"], out)
    assert (out / "results.csv").exists()
    assert len(df) == 1
    row = df.iloc[0]
    for col in ("test_R2", "test_RMSE", "test_MAE"):
        assert col in df.columns
    assert -5.0 < row["test_R2"] <= 1.0
    assert row["test_RMSE"] >= 0.0
    assert row["task"] == "regression"


def test_engine_classification(classification_csv, tmp_path):
    spec = DatasetSpec("SynthClf", classification_csv, "smiles", "label",
                       "classification")
    out = tmp_path / "out_clf"
    df = _run(spec, ["LogReg"], out)
    assert (out / "results.csv").exists()
    assert len(df) == 1
    row = df.iloc[0]
    for col in ("test_ROC_AUC", "test_PR_AUC", "test_ACC", "test_F1"):
        assert col in df.columns
    assert 0.0 <= row["test_ACC"] <= 1.0
    # ROC-AUC may be NaN only if a test fold is single-class; ACC must be finite
    assert math.isfinite(row["test_ACC"])
    assert row["task"] == "classification"
