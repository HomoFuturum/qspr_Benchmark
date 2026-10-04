"""Split-conformal prediction intervals for regression models.

Generalized from the paper's extended_analysis.py. Produces distribution-free
prediction intervals with finite-sample coverage guarantees using a held-out
calibration set, for any regression dataset in the registry.

Classification conformal (prediction sets) is out of scope for v1; calling this on a
classification dataset raises a clear error.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from ..data import DatasetSpec, load_dataset
from ..errors import ConfigError
from ..features import calc_descriptors
from ..splits import variance_prepare
from ..task import TaskType, make_task

RS = 42


def _best_gbr(Xtr, ytr, folds=3):
    grid = {"n_estimators": [400, 900], "learning_rate": [0.03, 0.08],
            "max_depth": [4, 6, 8], "subsample": [0.8, 1.0]}
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=folds, shuffle=True, random_state=RS)
    gs = GridSearchCV(GradientBoostingRegressor(random_state=RS), grid, cv=kf,
                      scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs.best_params_


def conformal_intervals(spec: DatasetSpec, alphas=(0.1, 0.2), test_frac=0.2,
                        cal_frac=0.25, root=None):
    """Split-conformal intervals for one regression dataset.

    Returns (summary_df, preds_df). summary_df has one row per alpha with empirical
    coverage and mean interval width; preds_df has per-test-compound y_true/y_pred.
    """
    if make_task(spec.task).type is not TaskType.REGRESSION:
        raise ConfigError(
            f"conformal_intervals supports regression only; {spec.name!r} is {spec.task}"
        )
    df, mols = load_dataset(spec, root=root)
    y = df["target"].values.astype(float)
    X = calc_descriptors(mols)
    tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=test_frac,
                                      random_state=RS)
    Xtr, Xte, _ = variance_prepare(X, tr_idx, te_idx)
    ytr, yte = y[tr_idx], y[te_idx]
    ptr, cal = train_test_split(np.arange(len(ytr)), test_size=cal_frac, random_state=RS)

    params = _best_gbr(Xtr.iloc[ptr], ytr[ptr])
    gbr = GradientBoostingRegressor(random_state=RS, **params)
    gbr.fit(Xtr.iloc[ptr], ytr[ptr])
    cal_scores = np.abs(ytr[cal] - gbr.predict(Xtr.iloc[cal]))
    n_cal = len(cal)
    preds_te = gbr.predict(Xte)

    rows = []
    for alpha in alphas:
        level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
        q = float(np.quantile(cal_scores, level, method="higher"))
        lo, hi = preds_te - q, preds_te + q
        rows.append({"dataset": spec.name, "alpha": alpha, "nominal": 1 - alpha,
                     "empirical_coverage": float(np.mean((yte >= lo) & (yte <= hi))),
                     "mean_width": float(np.mean(hi - lo)), "qhat": q, "n_cal": n_cal})
    summary = pd.DataFrame(rows)
    preds = pd.DataFrame({"y_true": yte, "y_pred": preds_te})
    return summary, preds


def run_from_registry(dataset_names, registry_path="configs/datasets.yaml",
                      out_dir=None, root=None):
    """Run conformal_intervals for several registered datasets; optionally write CSVs."""
    from ..config import load_registry
    registry = load_registry(registry_path)
    all_summ = []
    for name in dataset_names:
        if name not in registry:
            raise ConfigError(f"unknown dataset {name!r}")
        summ, preds = conformal_intervals(registry[name], root=root)
        all_summ.append(summ)
        if out_dir is not None:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            preds.to_csv(out / f"conformal_pred_{name}.csv", index=False)
        print(f"conformal {name}: cov90={summ.iloc[0]['empirical_coverage']:.3f} "
              f"width={summ.iloc[0]['mean_width']:.3f}")
    summary = pd.concat(all_summ, ignore_index=True)
    if out_dir is not None:
        summary.to_csv(Path(out_dir) / "conformal_results.csv", index=False)
    return summary
