"""SHAP interpretability for tree-based models.

Generalized from the paper's shap_analysis.py. Fits a gradient-boosted tree on a
dataset's descriptor features and computes TreeExplainer SHAP values, returning the
mean absolute SHAP ranking and (optionally) saving a summary plot. Works for both
regression (GradientBoostingRegressor) and classification (GradientBoostingClassifier).

Requires the 'analysis' extra: shap, matplotlib.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from ..data import DatasetSpec, load_dataset
from ..errors import MissingDependencyError
from ..features import calc_descriptors
from ..splits import variance_prepare
from ..task import TaskType, make_task

RS = 42


def _require():
    try:
        import shap  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            "SHAP analysis needs the 'analysis' extra: pip install qspr_bench[analysis]"
        ) from e
    import shap
    return shap


def shap_ranking(spec: DatasetSpec, test_frac=0.2, max_display=15, plot_path=None,
                 root=None):
    """Compute mean |SHAP| feature ranking for a GBDT on `spec`.

    Returns a DataFrame [feature, mean_abs_shap] sorted descending. If plot_path is
    given, also writes a SHAP summary beeswarm there (needs matplotlib).
    """
    shap = _require()
    task = make_task(spec.task)
    df, mols = load_dataset(spec, root=root)
    y = df["target"].values
    y = y.astype(int) if task.is_classification else y.astype(float)
    X = calc_descriptors(mols)
    tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=test_frac,
                                      random_state=RS,
                                      stratify=y if task.is_classification else None)
    Xtr, Xte, _ = variance_prepare(X, tr_idx, te_idx)

    if task.type is TaskType.CLASSIFICATION:
        model = GradientBoostingClassifier(random_state=RS)
    else:
        model = GradientBoostingRegressor(random_state=RS)
    model.fit(Xtr, y[tr_idx])

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xte, check_additivity=False)
    # for binary classifiers some shap versions return a list / 3-D array; collapse
    sv_arr = np.asarray(sv)
    if sv_arr.ndim == 3:
        sv_arr = sv_arr[..., -1]
    mean_abs = np.abs(sv_arr).mean(axis=0)
    ranking = (pd.DataFrame({"feature": Xte.columns, "mean_abs_shap": mean_abs})
               .sort_values("mean_abs_shap", ascending=False)
               .reset_index(drop=True))

    if plot_path is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise MissingDependencyError(
                "plotting needs matplotlib (part of the 'analysis' extra)"
            ) from e
        plt.figure(figsize=(7.2, 4.6))
        shap.summary_plot(sv_arr, Xte, show=False, max_display=max_display)
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close("all")

    return ranking
