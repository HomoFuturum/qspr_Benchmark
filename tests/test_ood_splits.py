"""Tests for out-of-distribution splits and the new sklearn models."""
import numpy as np
import pandas as pd
import pytest

from qspr_bench import registry
from qspr_bench.config import RunConfig
from qspr_bench.data import DatasetSpec
from qspr_bench.engine import run_benchmark
from qspr_bench.errors import ConfigError
from qspr_bench.splits import (
    _morgan_fps,
    nn_distance_split,
    random_split,
    target_extrapolation_split,
)
from qspr_bench.task import make_task
from rdkit import DataStructs


# ---- new models ----

def test_new_models_registered():
    reg = set(registry.names_for_task(make_task("regression")))
    clf = set(registry.names_for_task(make_task("classification")))
    assert {"ExtraTrees", "HistGBDT", "ElasticNet"} <= reg
    assert {"ExtraTrees", "HistGBDT"} <= clf
    assert "ElasticNet" not in clf  # regression-only, like MLR


def test_extratrees_engine_smoke(regression_csv, tmp_path):
    spec = DatasetSpec("SynthReg", regression_csv, "smiles", "y", "regression")
    cfg = RunConfig(name="t", datasets=[spec], models=["ExtraTrees"],
                    feature_sets=["Desc"], splits=["random"], test_frac=0.3,
                    cv_folds=2, seeds=[42], fp_bits=128, output_dir=tmp_path / "o")
    df = run_benchmark(cfg)
    assert len(df) == 1 and "test_R2" in df.columns


# ---- OOD split generators ----

def _mean_max_sim_to_train(df, tr, te):
    fps = _morgan_fps(df, 256)
    tr_fps = [fps[i] for i in tr]
    sims = [max(DataStructs.BulkTanimotoSimilarity(fps[i], tr_fps)) for i in te]
    return float(np.mean(sims))


def test_nn_distance_is_more_dissimilar_than_random():
    smiles = ["CCO", "CCN", "CCC", "CCCC", "CCCCC", "CCCCCC", "c1ccccc1", "c1ccccc1C",
              "c1ccccc1CC", "CC(=O)O", "CC(=O)N", "CCOC(=O)C", "C1CCCCC1", "C1CCNCC1",
              "c1ccncc1", "c1ccc(cc1)N", "c1ccc(cc1)Cl", "CCBr", "CCS", "COc1ccccc1",
              "OCC(O)CO", "CCCCCCO", "c1ccc2ccccc2c1", "C1CCOCC1", "CC(C)(C)O"]
    df = pd.DataFrame({"smiles": smiles})
    tr, te = nn_distance_split(df, frac=0.3, seed=42)
    assert len(set(tr) & set(te)) == 0
    assert len(tr) + len(te) == len(df)
    rtr, rte = random_split(len(df), frac=0.3, seed=42)
    # OOD test molecules should be less similar to train than a random test set
    assert _mean_max_sim_to_train(df, tr, te) < _mean_max_sim_to_train(df, rtr, rte)


def test_target_extrapolation_high_and_low():
    y = np.arange(20, dtype=float)
    tr, te = target_extrapolation_split(y, frac=0.25, direction="high")
    assert y[te].min() > y[tr].max()        # test strictly above train
    assert len(te) == 5
    tr2, te2 = target_extrapolation_split(y, frac=0.25, direction="low")
    assert y[te2].max() < y[tr2].min()      # test strictly below train


# ---- engine integration ----

def test_engine_ood_holdout_labels(regression_csv, tmp_path):
    spec = DatasetSpec("SynthReg", regression_csv, "smiles", "y", "regression")
    cfg = RunConfig(name="t", datasets=[spec], models=["RF"], feature_sets=["Desc"],
                    splits=["nn_distance", "extrapolation"], test_frac=0.3, cv_folds=2,
                    seeds=[42], fp_bits=128, output_dir=tmp_path / "o",
                    extrapolation_direction="high")
    df = run_benchmark(cfg)
    assert set(df["split"]) == {"nn_distance", "extrapolation_high"}


def test_engine_ood_nested_single_fold(regression_csv, tmp_path):
    spec = DatasetSpec("SynthReg", regression_csv, "smiles", "y", "regression")
    cfg = RunConfig(name="t", datasets=[spec], models=["RF"], feature_sets=["Desc"],
                    splits=["extrapolation"], cv_folds=2, seeds=[42], fp_bits=128,
                    output_dir=tmp_path / "o", evaluation="nested_cv", outer_folds=5)
    df = run_benchmark(cfg)
    assert len(df) == 1
    assert df.iloc[0]["n_outer_folds"] == 1  # OOD splits are single directional folds
    assert df.iloc[0]["split"] == "extrapolation_high"


# ---- config validation ----

def test_classification_extrapolation_rejected(tmp_path):
    df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1", "CCN", "CCC"],
                       "label": [0, 1, 0, 1]})
    p = tmp_path / "clf.csv"
    df.to_csv(p, index=False)
    reg = {"datasets": {"Clf": {"file": str(p), "smiles_col": "smiles",
                                "target_col": "label", "task": "classification"}}}
    import yaml
    reg_path = tmp_path / "datasets.yaml"
    reg_path.write_text(yaml.safe_dump(reg), encoding="utf-8")
    run_path = tmp_path / "run.yaml"
    run_path.write_text(yaml.safe_dump({"run": {
        "name": "x", "datasets": ["Clf"], "models": ["LogReg"],
        "feature_sets": ["Desc"], "splits": ["extrapolation"]}}), encoding="utf-8")
    from qspr_bench.config import load_run_config
    with pytest.raises(ConfigError):
        load_run_config(run_path, registry_path=reg_path)
