"""Tests for subgroup (disaggregated) performance analysis."""
import numpy as np
import pandas as pd

from qspr_bench.analysis.subgroups import subgroup_report


def _preds_frame(smiles, y_true, y_pred):
    return pd.DataFrame({
        "dataset": "D", "split": "random", "feature_set": "Desc", "model": "RF",
        "seed": 42, "idx": range(len(smiles)), "smiles": smiles,
        "y_true": y_true, "y_pred": y_pred, "y_score": "",
    })


def test_subgroup_report_regression_shape_and_coverage():
    smiles = ["CCO", "CCN", "CCC", "CCCC", "c1ccccc1", "c1ccccc1C",
              "CC(=O)O", "CCCCCCO", "C1CCCCC1", "c1ccncc1", "CCBr", "COc1ccccc1"]
    rng = np.random.default_rng(0)
    y_true = np.arange(len(smiles), dtype=float)
    y_pred = y_true + rng.normal(0, 0.5, len(smiles))
    preds = _preds_frame(smiles, y_true, y_pred)
    meta = pd.DataFrame([{"dataset": "D", "task": "regression"}])

    rep = subgroup_report(preds, meta)
    assert set(["subgroup_axis", "subgroup_value", "n", "test_R2"]).issubset(rep.columns)
    assert set(rep["subgroup_axis"].unique()) == {"scaffold_frequency", "mol_size",
                                                  "target_bin"}
    # within each axis, the subgroup counts sum to the number of molecules
    for axis, g in rep.groupby("subgroup_axis"):
        assert g["n"].sum() == len(smiles)


def test_subgroup_report_classification_uses_class_bins():
    smiles = ["CCO", "CCN", "CCC", "c1ccccc1", "c1ccccc1C", "CC(=O)O",
              "C1CCCCC1", "c1ccncc1"]
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    preds = _preds_frame(smiles, y_true, y_pred)
    preds["y_score"] = [0.2, 0.9, 0.3, 0.8, 0.4, 0.1, 0.6, 0.7]
    meta = pd.DataFrame([{"dataset": "D", "task": "classification"}])

    rep = subgroup_report(preds, meta)
    tb = rep[rep["subgroup_axis"] == "target_bin"]
    assert set(tb["subgroup_value"]).issubset({"class_0", "class_1"})
    assert "test_ACC" in rep.columns
    for _, row in rep.iterrows():
        assert 0.0 <= row["test_ACC"] <= 1.0
