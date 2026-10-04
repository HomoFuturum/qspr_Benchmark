"""Tests for the pluggable registry and heavy baselines.

The GNN/chemprop tests are skipped automatically when the optional extras are absent.
They use tiny subsamples + few epochs to stay fast.
"""
import numpy as np
import pandas as pd
import pytest

from qspr_bench import registry
from qspr_bench.task import make_task


def test_builtin_tabular_registered():
    reg_task = make_task("regression")
    clf_task = make_task("classification")
    assert "RF" in registry.names_for_task(reg_task)
    assert "XGBoost" in registry.names_for_task(reg_task)
    assert "LogReg" in registry.names_for_task(clf_task)
    # MLR is regression-only; LogReg is classification-only
    assert "MLR" in registry.names_for_task(reg_task)
    assert "MLR" not in registry.names_for_task(clf_task)


def test_register_custom_tabular_and_run(regression_csv, tmp_path):
    from sklearn.linear_model import HuberRegressor

    registry.register_tabular(
        "Huber", ["regression"],
        lambda task: (HuberRegressor(), {"epsilon": [1.2, 1.5]}),
        overwrite=True,
    )
    from qspr_bench.config import RunConfig
    from qspr_bench.data import DatasetSpec
    from qspr_bench.engine import run_benchmark

    spec = DatasetSpec("SynthReg", regression_csv, "smiles", "y", "regression")
    cfg = RunConfig(name="t", datasets=[spec], models=["Huber"],
                    feature_sets=["Desc"], splits=["random"], test_frac=0.3,
                    cv_folds=2, seeds=[42], fp_bits=128, output_dir=tmp_path / "o")
    df = run_benchmark(cfg)
    assert len(df) == 1
    assert df.iloc[0]["model"] == "Huber"
    assert "test_R2" in df.columns


def test_unknown_model_raises():
    from qspr_bench.errors import ConfigError
    with pytest.raises(ConfigError):
        registry.get("NoSuchModel")


# ---- heavy baselines (skipped if extras missing) ----

def test_gnn_registers_and_runs(regression_csv, tmp_path):
    pytest.importorskip("torch_geometric")
    from qspr_bench.baselines import gnn  # noqa: F401 (self-registers)
    assert "GIN" in registry.all_specs()
    assert registry.get("GIN").kind == registry.GRAPH

    df = pd.read_csv(regression_csv)
    y = df["y"].values.astype(float)
    tr = np.arange(0, 18)
    te = np.arange(18, len(y))
    out = gnn._train("GIN", df["smiles"].tolist(), y, tr, te,
                     make_task("regression"), seed=0, epochs=5, patience=3)
    assert len(out["y_pred"]) == len(te)
    assert np.all(np.isfinite(out["y_pred"]))


def test_chemprop_registers():
    pytest.importorskip("chemprop")
    from qspr_bench.baselines import chemprop  # noqa: F401
    assert "ChempropDMPNN" in registry.all_specs()
    assert registry.get("ChempropDMPNN").kind == registry.GRAPH
