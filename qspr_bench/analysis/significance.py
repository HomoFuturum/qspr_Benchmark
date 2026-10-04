"""Statistical significance testing over benchmark results.

Generalized from the paper's significance_analysis.py. Operates on a long-format
results frame (as written by engine.run_benchmark) that contains multiple seeds per
model so that seeds act as blocks for a repeated-measures comparison.

Part A - Friedman omnibus test + Nemenyi post-hoc (Demsar 2006): across models,
blocked by seed, per (dataset, split). Gives a single critical difference (CD) in
mean ranks that controls family-wise error over all pairwise comparisons.

Requires the 'analysis' extra: scipy, scikit-posthocs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import MissingDependencyError


def _require():
    try:
        import scikit_posthocs as sp  # noqa: F401
        from scipy.stats import friedmanchisquare, studentized_range  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            "significance analysis needs the 'analysis' extra: "
            "pip install qspr_bench[analysis]"
        ) from e


def _rank_metric_ascending(metric):
    """True if lower is better for this metric (so rank 1 = best)."""
    m = metric.lower()
    return any(k in m for k in ("rmse", "mae", "error", "loss"))


def friedman_nemenyi(results, metric="test_RMSE", min_seeds=3):
    """Friedman + Nemenyi across models for each (dataset, split).

    results: long-format DataFrame with columns dataset, split, model, seed, <metric>.
    Returns (summary_df, mean_ranks_df, pairs_df). Groups with fewer than `min_seeds`
    distinct seeds or fewer than 3 models are skipped (test is undefined).
    """
    _require()
    import scikit_posthocs as sp
    from scipy.stats import friedmanchisquare, studentized_range

    if metric not in results.columns:
        raise ValueError(f"metric {metric!r} not in results columns {list(results.columns)}")

    ascending = _rank_metric_ascending(metric)
    summary, rank_rows, pair_rows = [], [], []

    for (ds, split), g in results.groupby(["dataset", "split"]):
        wide = g.pivot_table(index="seed", columns="model", values=metric)
        wide = wide.dropna(axis=0, how="any")
        models = list(wide.columns)
        n, k = wide.shape[0], wide.shape[1]
        if n < min_seeds or k < 3:
            continue
        stat, p = friedmanchisquare(*[wide[m].values for m in models])
        # rank within each seed; rank 1 = best. For "higher is better" metrics,
        # rank the negated values so that larger -> rank 1.
        rank_input = wide if ascending else -wide
        mean_rank = rank_input.rank(axis=1, method="average").mean()
        q = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
        cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
        summary.append({"dataset": ds, "split": split, "metric": metric,
                        "friedman_chi2": float(stat), "friedman_p": float(p),
                        "k_models": k, "n_seeds": n, "nemenyi_CD": float(cd)})
        for m in models:
            rank_rows.append({"dataset": ds, "split": split, "model": m,
                              "mean_rank": float(mean_rank[m])})
        nem = sp.posthoc_nemenyi_friedman(wide[models].values)
        nem.index = nem.columns = models
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                pair_rows.append({"dataset": ds, "split": split,
                                  "model_a": a, "model_b": b,
                                  "rank_diff": float(abs(mean_rank[a] - mean_rank[b])),
                                  "nemenyi_p": float(nem.loc[a, b]),
                                  "sig_0.05": bool(nem.loc[a, b] < 0.05)})

    return (pd.DataFrame(summary), pd.DataFrame(rank_rows), pd.DataFrame(pair_rows))


def run_from_csv(results_csv, metric="test_RMSE", out_dir=None):
    """Convenience: load a results.csv, run friedman_nemenyi, optionally write CSVs."""
    from pathlib import Path
    results = pd.read_csv(results_csv)
    summary, ranks, pairs = friedman_nemenyi(results, metric=metric)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "sig_friedman.csv", index=False)
        ranks.to_csv(out / "sig_mean_ranks.csv", index=False)
        pairs.to_csv(out / "sig_nemenyi_pairs.csv", index=False)
    return summary, ranks, pairs
