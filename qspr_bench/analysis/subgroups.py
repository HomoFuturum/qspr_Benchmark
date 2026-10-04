"""Subgroup (disaggregated) performance analysis.

Aggregate benchmark metrics hide where a model fails. Given per-molecule predictions
(as written by engine.run_benchmark when save_predictions is on), this module computes
metrics within chemically meaningful subgroups so systematic weaknesses - novel
scaffolds, unusually large molecules, extreme target values - become visible.

Axes:
  * scaffold_frequency : how often a molecule's Bemis-Murcko scaffold occurs in the
                         dataset -> singleton (1) / rare (2-5) / common (>5).
  * mol_size           : heavy-atom-count tertile -> small / medium / large.
  * target_bin         : regression -> target quantile tertile (low/mid/high);
                         classification -> the class label (0/1).

Reuses splits.murcko_scaffold for scaffolds and RDKit for heavy-atom counts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from ..splits import murcko_scaffold
from ..task import make_task

RDLogger.DisableLog("rdApp.*")


def _heavy_atom_count(smi):
    m = Chem.MolFromSmiles(smi)
    return m.GetNumHeavyAtoms() if m is not None else np.nan


def _tertile_labels(values, names):
    """Assign each value to a tertile label; robust to ties/degenerate ranges."""
    v = np.asarray(values, dtype=float)
    try:
        cats = pd.qcut(v, 3, labels=names, duplicates="drop")
    except (ValueError, IndexError):
        return np.array(["all"] * len(v), dtype=object)
    # qcut may yield fewer than 3 bins when duplicates are dropped
    return np.asarray(cats, dtype=object)


def _assign_axes(pred_df, task):
    """Return a DataFrame with one subgroup-label column per axis, aligned to pred_df."""
    smiles = pred_df["smiles"].tolist()

    # scaffold frequency within this (dataset-level) prediction set
    scaffolds = [murcko_scaffold(s) for s in smiles]
    counts = pd.Series(scaffolds).value_counts().to_dict()

    def freq_label(sc):
        c = counts.get(sc, 0)
        if c <= 1:
            return "singleton"
        if c <= 5:
            return "rare(2-5)"
        return "common(>5)"

    scaffold_frequency = [freq_label(sc) for sc in scaffolds]
    mol_size = _tertile_labels([_heavy_atom_count(s) for s in smiles],
                               ["small", "medium", "large"])
    if make_task(task).is_classification:
        target_bin = pred_df["y_true"].astype(int).map({0: "class_0", 1: "class_1"}).to_numpy()
    else:
        target_bin = _tertile_labels(pred_df["y_true"].to_numpy(),
                                     ["low", "mid", "high"])

    return pd.DataFrame({
        "scaffold_frequency": scaffold_frequency,
        "mol_size": mol_size,
        "target_bin": target_bin,
    }, index=pred_df.index)


def subgroup_report(predictions, meta, axes=("scaffold_frequency", "mol_size",
                                             "target_bin")):
    """Compute per-subgroup metrics for each (dataset, split, feature_set, model, seed).

    predictions: DataFrame with columns dataset, split, feature_set, model, seed,
                 smiles, y_true, y_pred, y_score (y_score may be blank for regression).
    meta:        dataset_meta DataFrame (used to look up each dataset's task).
    Returns a long-format DataFrame.
    """
    task_of = dict(zip(meta["dataset"], meta["task"]))
    out = []
    group_keys = ["dataset", "split", "feature_set", "model", "seed"]

    for keys, g in predictions.groupby(group_keys):
        ds = keys[0]
        task_str = task_of.get(ds)
        if task_str is None:
            continue
        task = make_task(task_str)
        g = g.reset_index(drop=True)
        axis_labels = _assign_axes(g, task_str)

        y_true = g["y_true"].to_numpy()
        y_pred = g["y_pred"].to_numpy()
        has_score = g["y_score"].astype(str).str.len().gt(0).any()
        y_score = pd.to_numeric(g["y_score"], errors="coerce").to_numpy() if has_score else None

        for axis in axes:
            labels = axis_labels[axis].to_numpy()
            for sub in pd.unique(labels):
                mask = labels == sub
                if mask.sum() == 0:
                    continue
                ys = y_score[mask] if y_score is not None else None
                m = task.metrics_of(y_true[mask], y_pred=y_pred[mask], y_score=ys)
                row = {k: v for k, v in zip(group_keys, keys)}
                row.update({"subgroup_axis": axis, "subgroup_value": str(sub),
                            "n": int(mask.sum())})
                for mk in task.metric_names:
                    row[f"test_{mk}"] = m[mk]
                out.append(row)

    return pd.DataFrame(out)


def run_from_csv(predictions_csv, meta_csv, out_dir=None):
    """Load predictions.csv + dataset_meta.csv, build the report, optionally write it."""
    predictions = pd.read_csv(predictions_csv)
    meta = pd.read_csv(meta_csv)
    report = subgroup_report(predictions, meta)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        report.to_csv(out / "subgroups.csv", index=False)
    return report
