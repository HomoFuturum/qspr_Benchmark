"""Accuracy-vs-compute Pareto analysis.

Given a cost-instrumented results frame (engine run with measure_cost=True), identify
the Pareto-optimal models per (dataset, split, feature_set): those not dominated on both
accuracy and cost. This supports the "classical ML is competitive AND far cheaper than
GNNs" story a benchmark paper needs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# primary accuracy metric per task, and whether higher is better
_PRIMARY = {
    "regression": ("R2", True),
    "classification": ("ROC_AUC", True),
}


def pareto_frontier(cost, score, higher_is_better=True):
    """Boolean mask of Pareto-optimal points minimizing cost and optimizing score.

    A point is optimal if no other point is at least as cheap AND at least as good,
    with at least one strict improvement. cost/score are 1-D array-likes.
    """
    cost = np.asarray(cost, dtype=float)
    score = np.asarray(score, dtype=float)
    s = score if higher_is_better else -score
    n = len(cost)
    optimal = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(cost[i]) or not np.isfinite(s[i]):
            optimal[i] = False
            continue
        for j in range(n):
            if i == j or not np.isfinite(cost[j]) or not np.isfinite(s[j]):
                continue
            # j dominates i: no worse on both, strictly better on at least one
            if cost[j] <= cost[i] and s[j] >= s[i] and (cost[j] < cost[i] or s[j] > s[i]):
                optimal[i] = False
                break
    return optimal


def _score_column(results, metric_name):
    """Find the accuracy column for metric_name across holdout/nested schemas."""
    for cand in (f"test_{metric_name}", f"test_{metric_name}_pooled"):
        if cand in results.columns:
            return cand
    raise ValueError(
        f"no accuracy column for metric {metric_name!r}; have "
        f"{[c for c in results.columns if c.startswith('test_')]}"
    )


def efficiency_table(results, meta, cost_col="refit_seconds"):
    """Per (dataset, split, feature_set): [model, score, cost, mem, is_pareto_optimal].

    results: cost-instrumented results frame. meta: dataset_meta (for task lookup).
    """
    if cost_col not in results.columns:
        raise ValueError(
            f"cost column {cost_col!r} not in results; run with measure_cost=True "
            f"(have {[c for c in results.columns if 'seconds' in c or 'mem' in c]})"
        )
    task_of = dict(zip(meta["dataset"], meta["task"]))
    out = []
    for (ds, split, fs), g in results.groupby(["dataset", "split", "feature_set"]):
        task = task_of.get(ds)
        if task not in _PRIMARY:
            continue
        metric, higher = _PRIMARY[task]
        score_col = _score_column(results, metric)
        g = g.reset_index(drop=True)
        mask = pareto_frontier(g[cost_col].to_numpy(), g[score_col].to_numpy(),
                               higher_is_better=higher)
        for k, rrow in g.iterrows():
            out.append({
                "dataset": ds, "split": split, "feature_set": fs,
                "model": rrow["model"], "metric": metric,
                "score": float(rrow[score_col]),
                "cost_col": cost_col, "cost": float(rrow[cost_col]),
                "peak_mem_mb": float(rrow.get("peak_mem_mb", np.nan)),
                "is_pareto_optimal": bool(mask[k]),
            })
    return pd.DataFrame(out)


def run_from_csv(results_csv, meta_csv, out_dir=None, cost_col="refit_seconds"):
    """Load results.csv + dataset_meta.csv, build the table, optionally write it."""
    results = pd.read_csv(results_csv)
    meta = pd.read_csv(meta_csv)
    table = efficiency_table(results, meta, cost_col=cost_col)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "efficiency.csv", index=False)
    return table
