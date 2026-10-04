"""Core benchmark engine.

run_benchmark iterates dataset x split x feature_set x model x seed, fits each model
via GridSearchCV (task-appropriate scoring + CV splitter), evaluates on the held-out
test set, and writes long-format results.csv + dataset_meta.csv to the output dir.
"""
from __future__ import annotations

import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_validate

from . import registry
from .config import RunConfig
from .data import load_dataset
from .errors import MissingDependencyError
from .features import calc_descriptors, calc_fps
from .registry import GRAPH, TABULAR
from .splits import (
    inner_cv,
    make_outer_folds,
    n_scaffolds,
    nn_distance_split,
    random_split,
    scaffold_labels,
    scaffold_split,
    scaffold_split_seeded,
    target_extrapolation_split,
    variance_prepare,
)
from .task import make_task

warnings.filterwarnings("ignore")

GRAPH_FEATURE_LABEL = "graph"


def _feature_frames(mols, feature_sets, fp_bits):
    """Build only the feature frames requested by the run config."""
    need_desc = any(fs in ("Desc", "Desc+FP") for fs in feature_sets)
    need_fp = any(fs in ("FP", "Desc+FP") for fs in feature_sets)
    Xd = calc_descriptors(mols) if need_desc else None
    Xf = calc_fps(mols, fp_bits) if need_fp else None
    frames = {}
    for fs in feature_sets:
        if fs == "Desc":
            frames[fs] = Xd
        elif fs == "FP":
            frames[fs] = Xf
        elif fs == "Desc+FP":
            frames[fs] = pd.concat(
                [Xd.reset_index(drop=True), Xf.reset_index(drop=True)], axis=1
            )
    return frames


def _score_vector(estimator, Xte, needs_proba):
    """Positive-class scores for classification ROC/PR-AUC."""
    if not needs_proba:
        return None
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(Xte)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(Xte)
    return estimator.predict(Xte)


COST_COLUMNS = ("search_seconds", "refit_seconds", "predict_seconds", "peak_mem_mb")


def _timed_cost(estimator, Xtr, ytr, Xte):
    """Measure a clean single-process refit + predict cost of `estimator`.

    Clones the estimator (fresh, unfitted), forces n_jobs=1 where supported so the
    measurement is single-process and tracemalloc-visible, times fit and predict
    separately, and captures the Python-heap peak during fit. Returns
    {refit_seconds, predict_seconds, peak_mem_mb}. peak_mem_mb is a comparative
    Python-heap figure, not absolute RSS (native buffers are not fully captured).
    """
    est = clone(estimator)
    try:
        est.set_params(n_jobs=1)
    except (ValueError, TypeError):
        pass  # estimator (or pipeline) has no n_jobs
    tracemalloc.start()
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    refit_seconds = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t1 = time.perf_counter()
    est.predict(Xte)
    predict_seconds = time.perf_counter() - t1
    return {"refit_seconds": float(refit_seconds),
            "predict_seconds": float(predict_seconds),
            "peak_mem_mb": float(peak) / (1024 * 1024)}


def _call_trainer(trainer, smiles, y, tr_idx, te_idx, task, seed, measure_cost):
    """Run a graph trainer, optionally timing it. Returns (out, cost_or_None).

    Graph trainers fit and predict internally, so time is not separable: the whole
    trainer wall-clock is recorded as refit_seconds; search/predict are NaN.
    """
    if not measure_cost:
        return trainer(smiles, y, tr_idx, te_idx, task, seed), None
    tracemalloc.start()
    t0 = time.perf_counter()
    out = trainer(smiles, y, tr_idx, te_idx, task, seed)
    refit_seconds = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cost = {"search_seconds": float("nan"), "refit_seconds": float(refit_seconds),
            "predict_seconds": float("nan"), "peak_mem_mb": float(peak) / (1024 * 1024)}
    return out, cost


def _splits_for(df, y, task, split_names, test_frac, seed, extrapolation_direction="high"):
    """Build {split_label: (tr_idx, te_idx)} for one seed (holdout path).

    OOD splits are directional single holdouts; extrapolation is labeled by direction.
    """
    out = {}
    for name in split_names:
        if name == "random":
            strat = y if task.is_classification else None
            out[name] = random_split(len(y), frac=test_frac, seed=seed, stratify=strat)
        elif name == "scaffold":
            # deterministic for the base seed, randomized ordering otherwise so
            # multi-seed runs probe scaffold-split robustness
            if seed == 42:
                out[name] = scaffold_split(df, frac=test_frac)
            else:
                out[name] = scaffold_split_seeded(df, seed=seed, frac=test_frac)
        elif name == "nn_distance":
            out[name] = nn_distance_split(df, frac=test_frac, seed=seed)
        elif name == "extrapolation":
            label = f"extrapolation_{extrapolation_direction}"
            out[label] = target_extrapolation_split(y, frac=test_frac,
                                                    direction=extrapolation_direction)
    return out


def run_benchmark(cfg: RunConfig) -> pd.DataFrame:
    """Run the benchmark described by cfg; write results and return the results frame."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_baselines_registered(cfg.models)
    rows, meta_rows, pred_rows = [], [], []

    # partition requested models by registry kind
    tabular_models, graph_models = [], []
    for mname in cfg.models:
        spec_m = registry.get(mname)
        (graph_models if spec_m.kind == GRAPH else tabular_models).append(mname)

    for spec in cfg.datasets:
        task = make_task(spec.task)
        df, mols = load_dataset(spec, root=_project_root(cfg))
        smiles = df["smiles"].tolist()
        y = df["target"].values
        y = y.astype(int) if task.is_classification else y.astype(float)
        frames = _feature_frames(mols, cfg.feature_sets, cfg.fp_bits) if tabular_models else {}

        meta = {"dataset": spec.name, "task": spec.task, "n": int(len(y)),
                "n_scaffolds": n_scaffolds(df)}
        if task.is_classification:
            pos = int(np.sum(y == 1))
            meta.update({"n_pos": pos, "n_neg": int(len(y) - pos),
                         "pos_frac": float(pos / len(y))})
        else:
            meta.update({"y_min": float(y.min()), "y_max": float(y.max()),
                         "y_mean": float(y.mean()), "y_std": float(y.std())})
        meta_rows.append(meta)
        print(f"== {spec.name}: {len(y)} molecules ({spec.task}, {cfg.evaluation}) ==")

        if cfg.evaluation == "nested_cv":
            _run_dataset_nested(cfg, spec, task, df, smiles, y, frames,
                                tabular_models, graph_models, rows, pred_rows)
        else:
            _run_dataset_holdout(cfg, spec, task, df, smiles, y, frames,
                                 tabular_models, graph_models, rows, pred_rows)

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(out_dir / "dataset_meta.csv", index=False)
    if cfg.save_predictions and pred_rows:
        pd.DataFrame(pred_rows).to_csv(out_dir / "predictions.csv", index=False)
        print(f"wrote {out_dir / 'predictions.csv'} ({len(pred_rows)} rows)")
    print(f"\nwrote {out_dir / 'results.csv'} ({len(results)} rows)")
    return results


def _append_predictions(pred_rows, spec, mname, sp_name, fs_name, seed, idx,
                        smiles, y, y_pred, y_score):
    """Record per-molecule predictions for the given test indices."""
    for k, i in enumerate(idx):
        pred_rows.append({
            "dataset": spec.name, "split": sp_name, "feature_set": fs_name,
            "model": mname, "seed": seed, "idx": int(i), "smiles": smiles[i],
            "y_true": float(y[i]), "y_pred": float(y_pred[k]),
            "y_score": (float(y_score[k]) if y_score is not None else ""),
        })


def _run_dataset_holdout(cfg, spec, task, df, smiles, y, frames,
                         tabular_models, graph_models, rows, pred_rows):
    for seed in cfg.seeds:
        splits = _splits_for(df, y, task, cfg.splits, cfg.test_frac, seed,
                             cfg.extrapolation_direction)
        for sp_name, (tr_idx, te_idx) in splits.items():
            for fs_name in cfg.feature_sets:
                if not tabular_models:
                    break
                Xtr, Xte, n_feat = variance_prepare(frames[fs_name], tr_idx, te_idx)
                for mname in tabular_models:
                    row, (yp, ys) = _run_tabular(spec, task, mname, cfg, seed, sp_name,
                                                 fs_name, Xtr, Xte, y, tr_idx, te_idx,
                                                 n_feat)
                    rows.append(row)
                    if cfg.save_predictions:
                        _append_predictions(pred_rows, spec, mname, sp_name, fs_name,
                                            seed, te_idx, smiles, y, yp, ys)
            for mname in graph_models:
                row, (yp, ys) = _run_graph(spec, task, mname, df, y, tr_idx, te_idx,
                                           seed, sp_name, measure_cost=cfg.measure_cost)
                rows.append(row)
                if cfg.save_predictions:
                    _append_predictions(pred_rows, spec, mname, sp_name,
                                        GRAPH_FEATURE_LABEL, seed, te_idx, smiles, y,
                                        yp, ys)


def _aggregate_nested(spec, task, mname, sp_name, fs_name, seed, fold_metrics,
                      oof_true, oof_pred, oof_score, n_feats, n_total, best_params_list,
                      fold_costs=None):
    """Build one aggregated nested-CV result row from per-fold metrics + pooled OOF."""
    pooled = task.metrics_of(np.asarray(oof_true),
                             y_pred=np.asarray(oof_pred),
                             y_score=(np.asarray(oof_score) if oof_score else None))
    row = {
        "dataset": spec.name, "task": spec.task, "evaluation": "nested_cv",
        "split": sp_name, "feature_set": fs_name, "model": mname, "seed": seed,
        "n_train": n_total, "n_test": len(oof_true),
        "n_features": int(np.mean(n_feats)) if n_feats else 0,
        "n_outer_folds": len(fold_metrics),
        "best_params": str(best_params_list),
    }
    for mk in task.metric_names:
        vals = np.array([fm[mk] for fm in fold_metrics], dtype=float)
        row[f"test_{mk}_mean"] = float(np.nanmean(vals))
        row[f"test_{mk}_std"] = float(np.nanstd(vals))
        row[f"test_{mk}_pooled"] = pooled[mk]
    # cost columns: mean across outer folds (NaN-safe)
    costs = [c for c in (fold_costs or []) if c]
    if costs:
        for col in COST_COLUMNS:
            vals = np.array([c.get(col, np.nan) for c in costs], dtype=float)
            row[col] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else np.nan
    return row


def _run_dataset_nested(cfg, spec, task, df, smiles, y, frames,
                        tabular_models, graph_models, rows, pred_rows):
    labels = scaffold_labels(df) if "scaffold" in cfg.splits else None
    for seed in cfg.seeds:
        for sp_name in cfg.splits:
            folds = make_outer_folds(df, y, task, sp_name, cfg.outer_folds, seed,
                                     cfg.extrapolation_direction)
            # direction-aware result label; sp_name stays the fold-logic key
            sp_label = (f"extrapolation_{cfg.extrapolation_direction}"
                        if sp_name == "extrapolation" else sp_name)

            # tabular models: inner GridSearchCV per outer fold, per feature set
            for fs_name in cfg.feature_sets:
                if not tabular_models:
                    break
                for mname in tabular_models:
                    estimator, grid = registry.get(mname).builder(task)
                    fold_metrics, n_feats, bps, fold_costs = [], [], [], []
                    o_true, o_pred, o_score = [], [], []
                    for tr_idx, te_idx in folds:
                        Xtr, Xte, n_feat = variance_prepare(frames[fs_name], tr_idx, te_idx)
                        groups = labels[tr_idx] if sp_name == "scaffold" else None
                        m, yp, ys, bp, cost = _fit_predict_tabular_fold(
                            estimator, grid, task, cfg, sp_name, groups,
                            Xtr, Xte, y[tr_idx], y[te_idx])
                        fold_metrics.append(m); n_feats.append(n_feat); bps.append(bp)
                        fold_costs.append(cost)
                        o_true.extend(y[te_idx]); o_pred.extend(yp)
                        if ys is not None:
                            o_score.extend(ys)
                        if cfg.save_predictions:
                            _append_predictions(pred_rows, spec, mname, sp_label, fs_name,
                                                seed, te_idx, smiles, y, yp, ys)
                    row = _aggregate_nested(spec, task, mname, sp_label, fs_name, seed,
                                            fold_metrics, o_true, o_pred, o_score,
                                            n_feats, len(y), bps, fold_costs=fold_costs)
                    rows.append(row)
                    _print_nested(sp_label, fs_name, mname, seed, row, task)

            # graph models: own training loop per outer fold (no GridSearchCV)
            for mname in graph_models:
                trainer = registry.get(mname).trainer
                fold_metrics, fold_costs = [], []
                o_true, o_pred, o_score = [], [], []
                for tr_idx, te_idx in folds:
                    out, cost = _call_trainer(trainer, smiles, y, tr_idx, te_idx, task,
                                              seed, cfg.measure_cost)
                    yp, ys = out["y_pred"], out.get("y_score")
                    fold_metrics.append(task.metrics_of(y[te_idx], y_pred=yp, y_score=ys))
                    fold_costs.append(cost)
                    o_true.extend(y[te_idx]); o_pred.extend(yp)
                    if ys is not None:
                        o_score.extend(ys)
                    if cfg.save_predictions:
                        _append_predictions(pred_rows, spec, mname, sp_label,
                                            GRAPH_FEATURE_LABEL, seed, te_idx, smiles, y,
                                            yp, ys)
                row = _aggregate_nested(spec, task, mname, sp_label, GRAPH_FEATURE_LABEL,
                                        seed, fold_metrics, o_true, o_pred, o_score,
                                        [0], len(y), [{}], fold_costs=fold_costs)
                rows.append(row)
                _print_nested(sp_label, GRAPH_FEATURE_LABEL, mname, seed, row, task)


def _print_nested(sp_name, fs_name, mname, seed, row, task):
    primary = task.metric_names[0]
    print(f"{sp_name:>8} {fs_name:>9} {mname:>13} s{seed} | "
          f"{primary}: {row[f'test_{primary}_mean']:.3f}+/-{row[f'test_{primary}_std']:.3f} "
          f"(pooled {row[f'test_{primary}_pooled']:.3f})")


def _base_row(spec, task, mname, sp_name, fs_name, seed, tr_idx, te_idx, n_feat, te,
              evaluation="holdout", cost=None):
    row = {
        "dataset": spec.name, "task": spec.task, "evaluation": evaluation,
        "split": sp_name, "feature_set": fs_name, "model": mname, "seed": seed,
        "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
        "n_features": n_feat,
    }
    for mk in task.metric_names:
        row[f"test_{mk}"] = te[mk]
    if cost:
        for col in COST_COLUMNS:
            if col in cost:
                row[col] = cost[col]
    return row


def _fit_predict_tabular_fold(estimator, grid, task, cfg, split_name, groups,
                              Xtr, Xte, ytr, yte):
    """Tune via GridSearchCV on (Xtr, ytr) and predict on Xte.

    `groups` is the scaffold labels of the outer-train subset when split_name is
    "scaffold" (used for grouped inner CV); None otherwise. Returns
    (metrics, y_pred, y_score, best_params).
    """
    cv, fit_groups = inner_cv(split_name, task, groups, cfg.cv_folds)
    gs = GridSearchCV(estimator, grid, cv=cv, scoring=task.gridsearch_scoring, n_jobs=-1)
    t0 = time.perf_counter()
    gs.fit(Xtr, ytr, groups=fit_groups)
    search_seconds = time.perf_counter() - t0
    best = gs.best_estimator_
    y_pred = best.predict(Xte)
    y_score = _score_vector(best, Xte, task.needs_proba)
    metrics = task.metrics_of(yte, y_pred=y_pred, y_score=y_score)
    cost = None
    if cfg.measure_cost:
        cost = {"search_seconds": float(search_seconds)}
        cost.update(_timed_cost(estimator, Xtr, ytr, Xte))
    return metrics, y_pred, y_score, gs.best_params_, cost


def _run_tabular(spec, task, mname, cfg, seed, sp_name, fs_name, Xtr, Xte, y,
                 tr_idx, te_idx, n_feat):
    estimator, grid = registry.get(mname).builder(task)
    cv = task.make_cv(cfg.cv_folds, seed)
    gs = GridSearchCV(estimator, grid, cv=cv, scoring=task.gridsearch_scoring, n_jobs=-1)
    t0 = time.perf_counter()
    gs.fit(Xtr, y[tr_idx])
    search_seconds = time.perf_counter() - t0
    best = gs.best_estimator_
    cvres = cross_validate(best, Xtr, y[tr_idx], cv=cv, scoring=task.cv_scoring, n_jobs=-1)
    y_pred = best.predict(Xte)
    y_score = _score_vector(best, Xte, task.needs_proba)
    te = task.metrics_of(y[te_idx], y_pred=y_pred, y_score=y_score)

    cost = None
    if cfg.measure_cost:
        cost = {"search_seconds": float(search_seconds)}
        cost.update(_timed_cost(estimator, Xtr, y[tr_idx], Xte))
    row = _base_row(spec, task, mname, sp_name, fs_name, seed, tr_idx, te_idx, n_feat, te,
                    cost=cost)
    row["best_params"] = str(gs.best_params_)
    for key, scorer in task.cv_scoring.items():
        vals = cvres[f"test_{key}"]
        sign = -1.0 if scorer.startswith("neg_") else 1.0  # sklearn negates error scorers
        row[f"cv_{key}"] = float(sign * vals.mean())
        row[f"cv_{key}_std"] = float(vals.std())
    _print_row(sp_name, fs_name, mname, seed, te, task)
    return row, (y_pred, y_score)


def _run_graph(spec, task, mname, df, y, tr_idx, te_idx, seed, sp_name, measure_cost=False):
    trainer = registry.get(mname).trainer
    smiles = df["smiles"].tolist()
    out, cost = _call_trainer(trainer, smiles, y, tr_idx, te_idx, task, seed, measure_cost)
    y_pred = out["y_pred"]
    y_score = out.get("y_score")
    te = task.metrics_of(y[te_idx], y_pred=y_pred, y_score=y_score)
    row = _base_row(spec, task, mname, sp_name, GRAPH_FEATURE_LABEL, seed,
                    tr_idx, te_idx, out.get("n_features", 0), te, cost=cost)
    row["best_params"] = str(out.get("info", {}))
    _print_row(sp_name, GRAPH_FEATURE_LABEL, mname, seed, te, task)
    return row, (y_pred, y_score)


def _ensure_baselines_registered(model_names):
    """Lazily import baseline subpackages so their models self-register.

    Only imports what a run actually requests, keeping heavy deps out of the path
    for runs that use only built-in tabular models.
    """
    want = set(model_names)
    if want & {"GIN", "MPNN"}:
        from .baselines import gnn  # noqa: F401  (registers GIN/MPNN on import)
    if "ChempropDMPNN" in want:
        from .baselines import chemprop  # noqa: F401
    # any still-unregistered requested names will raise a clear error in registry.get
    missing = [m for m in want if m not in registry.all_specs()]
    if missing:
        raise MissingDependencyError(
            f"models {missing} are not registered. If these are heavy baselines, "
            f"install the extra (qspr_bench[gnn] or qspr_bench[chemprop]); otherwise "
            f"register them with qspr_bench.registry.register_tabular/register_graph."
        )


def _print_row(sp_name, fs_name, mname, seed, te, task):
    primary = task.metric_names[0]
    extra = task.metric_names[1]
    print(f"{sp_name:>8} {fs_name:>9} {mname:>13} s{seed} | "
          f"{primary}={te[primary]:.3f} {extra}={te[extra]:.3f}")


def _project_root(cfg: RunConfig):
    """Resolve relative dataset paths against the parent of the output dir's anchor.

    Dataset file paths in the registry are relative to the project root (cwd when the
    CLI is invoked), so returning None lets load_dataset resolve them against cwd.
    """
    return None
