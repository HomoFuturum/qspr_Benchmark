"""Tests for the nested cross-validation evaluation mode."""
import numpy as np
import pandas as pd
import pytest

from qspr_bench.config import RunConfig
from qspr_bench.data import DatasetSpec
from qspr_bench.engine import run_benchmark
from qspr_bench.splits import make_outer_folds, scaffold_labels
from qspr_bench.task import make_task


def _nested_cfg(spec, models, out, splits=("random",), folds=3):
    return RunConfig(
        name="nested", datasets=[spec], models=list(models),
        feature_sets=["Desc"], splits=list(splits), cv_folds=2, seeds=[42],
        fp_bits=128, output_dir=out, evaluation="nested_cv", outer_folds=folds,
        save_predictions=True,
    )


def test_nested_cv_regression_schema_and_oof(regression_csv, tmp_path):
    spec = DatasetSpec("SynthReg", regression_csv, "smiles", "y", "regression")
    out = tmp_path / "o"
    df = run_benchmark(_nested_cfg(spec, ["RF"], out, folds=3))
    assert len(df) == 1
    row = df.iloc[0]
    for col in ("test_R2_mean", "test_R2_std", "test_R2_pooled", "n_outer_folds"):
        assert col in df.columns
    assert row["n_outer_folds"] == 3
    assert row["evaluation"] == "nested_cv"
    assert -5.0 < row["test_R2_pooled"] <= 1.0
    # OOF predictions cover every molecule exactly once (one split, one seed)
    preds = pd.read_csv(out / "predictions.csv")
    n_mol = len(pd.read_csv(regression_csv))
    assert len(preds) == n_mol
    assert sorted(preds["idx"]) == list(range(n_mol))


def test_nested_cv_scaffold_folds_disjoint(regression_csv):
    df = pd.read_csv(regression_csv).rename(columns={"y": "target"})
    y = df["target"].to_numpy()
    task = make_task("regression")
    folds = make_outer_folds(df, y, task, "scaffold", n_folds=3, seed=42)
    labels = scaffold_labels(df)
    # every row is tested exactly once across folds
    tested = np.concatenate([te for _, te in folds])
    assert sorted(tested) == list(range(len(df)))
    # no scaffold shared between train and test of any fold
    for tr, te in folds:
        assert set(labels[tr]).isdisjoint(set(labels[te]))


def test_nested_cv_classification(classification_csv, tmp_path):
    spec = DatasetSpec("SynthClf", classification_csv, "smiles", "label",
                       "classification")
    out = tmp_path / "oc"
    df = run_benchmark(_nested_cfg(spec, ["LogReg"], out, folds=3))
    assert len(df) == 1
    for col in ("test_ROC_AUC_mean", "test_ACC_pooled"):
        assert col in df.columns
    assert 0.0 <= df.iloc[0]["test_ACC_pooled"] <= 1.0


def test_nested_cv_graph_model(regression_csv, tmp_path):
    pytest.importorskip("torch_geometric")
    from qspr_bench.baselines import gnn  # noqa: F401 (self-registers)
    spec = DatasetSpec("SynthReg", regression_csv, "smiles", "y", "regression")
    out = tmp_path / "og"
    cfg = _nested_cfg(spec, ["GIN"], out, folds=2)
    df = run_benchmark(cfg)
    assert len(df) == 1
    assert df.iloc[0]["feature_set"] == "graph"
    assert np.isfinite(df.iloc[0]["test_RMSE_pooled"])
