"""Tests for the optional analysis modules (require the 'analysis' extra)."""
import numpy as np
import pandas as pd
import pytest

sph = pytest.importorskip("scikit_posthocs")

from qspr_bench.analysis import significance
from qspr_bench.data import DatasetSpec


def test_friedman_nemenyi_ranks_by_metric():
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(5):
        for model, base in [("RF", 0.8), ("XGBoost", 0.6), ("MLR", 1.1)]:
            rows.append({"dataset": "D", "split": "random", "model": model,
                         "seed": seed, "test_RMSE": base + rng.normal(0, 0.03)})
    res = pd.DataFrame(rows)
    summary, ranks, pairs = significance.friedman_nemenyi(res, metric="test_RMSE",
                                                          min_seeds=3)
    assert len(summary) == 1
    assert summary.iloc[0]["k_models"] == 3
    assert summary.iloc[0]["n_seeds"] == 5
    # lower RMSE should earn a better (smaller) mean rank
    rmap = ranks.set_index("model")["mean_rank"]
    assert rmap["XGBoost"] < rmap["RF"] < rmap["MLR"]


def test_friedman_skips_small_groups():
    rows = [{"dataset": "D", "split": "random", "model": m, "seed": s,
             "test_RMSE": 1.0} for s in range(2) for m in ("A", "B", "C")]
    summary, ranks, pairs = significance.friedman_nemenyi(pd.DataFrame(rows),
                                                          min_seeds=3)
    assert len(summary) == 0  # only 2 seeds -> skipped


def test_conformal_rejects_classification(tmp_path):
    from qspr_bench.analysis import conformal
    from qspr_bench.errors import ConfigError
    # build a tiny binary CSV
    df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1", "CCN", "CCC"],
                       "label": [0, 1, 0, 1]})
    p = tmp_path / "clf.csv"
    df.to_csv(p, index=False)
    spec = DatasetSpec("Clf", p, "smiles", "label", "classification")
    with pytest.raises(ConfigError):
        conformal.conformal_intervals(spec)
