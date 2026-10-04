"""Featurization primitives and metrics.

Ported from the original features.py. calc_descriptors / calc_fps are unchanged
(they are molecule-set agnostic). Metrics are split into regression and classification
variants; Task.metrics_of dispatches to the right one.
"""
import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

RDLogger.DisableLog("rdApp.*")


def calc_descriptors(mols):
    """Compute RDKit's full descriptor suite (~210 columns) for a list of mols."""
    rows = []
    for mol in mols:
        try:
            rows.append(Descriptors.CalcMolDescriptors(mol))
        except Exception:
            rows.append({})
    names = sorted(set().union(*[set(r) for r in rows]))
    X = pd.DataFrame([{k: r.get(k, np.nan) for k in names} for r in rows], columns=names)
    X = X.replace([np.inf, -np.inf], np.nan)
    # Some RDKit descriptors (e.g. Ipc) can grow astronomically large and, while
    # finite in float64, overflow float32 inside sklearn/xgboost tree estimators.
    # Mask anything beyond the float32 range so it is median-imputed downstream.
    f32_max = float(np.finfo(np.float32).max)
    return X.where(X.abs() <= f32_max, np.nan)


def calc_fps(mols, n_bits=1024):
    """Morgan fingerprints (radius 2) as a DataFrame with columns FP0..FP{n_bits-1}."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    cols = [f"FP{i}" for i in range(n_bits)]
    return pd.DataFrame([list(gen.GetFingerprint(m)) for m in mols], columns=cols)


def regression_metrics(y_true, y_pred):
    """Regression metrics: R2, RMSE, MAE."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred, y_score):
    """Binary-classification metrics: ROC_AUC, PR_AUC, ACC, F1.

    y_pred: hard 0/1 labels. y_score: positive-class scores/probabilities.
    ROC_AUC / PR_AUC fall back to NaN if only one class is present in y_true.
    """
    try:
        roc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y_true, y_score))
    except ValueError:
        pr = float("nan")
    return {
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "ACC": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
