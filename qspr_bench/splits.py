"""Splitting and feature-preparation utilities.

Ported verbatim (logic-wise) from run_all_benchmarks.py and multi_scaffold_split.py.
scaffold_split / scaffold_split_seeded partition by Bemis-Murcko framework and are
label-agnostic, so they serve both regression and classification. random_split uses
stratification for classification.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, train_test_split


def variance_prepare(X_full, tr_idx, te_idx):
    """Median-impute and drop zero-variance columns using train statistics only.

    Returns (Xtr, Xte, n_features_kept).
    """
    tr_raw, te_raw = X_full.iloc[tr_idx], X_full.iloc[te_idx]
    filled = tr_raw.fillna(tr_raw.median())
    keep = [c for c in tr_raw.columns if filled[c].std() > 0]
    med = tr_raw[keep].median()
    return tr_raw[keep].fillna(med), te_raw[keep].fillna(med), len(keep)


def murcko_scaffold(smile):
    """Bemis-Murcko scaffold SMILES for a molecule, or '' on failure."""
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(df, frac=0.2):
    """Deterministic Bemis-Murcko scaffold split.

    Groups by scaffold, sorts by (size desc, scaffold SMILES), and greedily fills the
    test set until it reaches `frac`. No scaffold is shared across train/test.
    """
    scaffolds = {}
    for i, s in enumerate(df["smiles"]):
        scaffolds.setdefault(murcko_scaffold(s), []).append(i)
    groups = sorted(scaffolds.values(), key=lambda g: (-len(g), df["smiles"].iloc[g[0]]))
    n_te = int(np.ceil(frac * len(df)))
    te, tr = [], []
    for g in groups:
        if len(te) < n_te:
            te.extend(g)
        else:
            tr.extend(g)
    return np.array(tr), np.array(te)


def scaffold_split_seeded(df, seed, frac=0.2):
    """Scaffold split with randomized group ordering (reproducible via seed)."""
    scaffolds = {}
    for i, s in enumerate(df["smiles"]):
        scaffolds.setdefault(murcko_scaffold(s), []).append(i)
    groups = list(scaffolds.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_te = int(np.ceil(frac * len(df)))
    te, tr = [], []
    for g in groups:
        if len(te) < n_te:
            te.extend(g)
        else:
            tr.extend(g)
    return np.array(tr), np.array(te)


def random_split(n, frac=0.2, seed=42, stratify=None):
    """Random train/test index split. `stratify` (labels) enables stratified splitting."""
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=frac, random_state=seed, stratify=stratify)
    return np.array(tr), np.array(te)


def n_scaffolds(df):
    return len({murcko_scaffold(s) for s in df["smiles"]})


# --------------------------------------------------------------------------- #
# Fold generators for nested cross-validation
# --------------------------------------------------------------------------- #

def scaffold_labels(df):
    """Per-row Bemis-Murcko scaffold label (used to group rows for scaffold folds)."""
    return np.array([murcko_scaffold(s) for s in df["smiles"]])


def kfold_indices(n, n_folds, seed, stratify=None):
    """Return a list of (train_idx, test_idx) for shuffled K-fold over n samples.

    Uses StratifiedKFold when `stratify` (label array) is given, else KFold.
    """
    idx = np.arange(n)
    if stratify is not None:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = splitter.split(idx, stratify)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = splitter.split(idx)
    return [(np.array(tr), np.array(te)) for tr, te in folds]


def scaffold_kfold(df, n_folds, seed):
    """Scaffold-grouped K-fold: no scaffold appears in more than one test fold.

    Scaffolds are sorted by descending group size (with a seed-shuffled tie order) and
    distributed greedily into the currently-smallest fold, which keeps folds balanced
    while preserving the scaffold grouping. Returns a list of (train_idx, test_idx).
    """
    scaffolds = {}
    for i, s in enumerate(df["smiles"]):
        scaffolds.setdefault(murcko_scaffold(s), []).append(i)
    groups = list(scaffolds.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)  # randomize tie order
    groups.sort(key=len, reverse=True)  # stable: largest first, shuffled within ties

    fold_members = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for g in groups:
        j = int(np.argmin(fold_sizes))
        fold_members[j].extend(g)
        fold_sizes[j] += len(g)

    all_idx = np.arange(len(df))
    folds = []
    for j in range(n_folds):
        te = np.array(sorted(fold_members[j]))
        tr = np.setdiff1d(all_idx, te)
        folds.append((tr, te))
    return folds


def make_outer_folds(df, y, task, split_name, n_folds, seed, extrapolation_direction="high"):
    """Dispatch outer folds by split type.

    random -> stratified/K-fold; scaffold -> scaffold K-fold. OOD splits (nn_distance,
    extrapolation) are inherently directional, so they return a SINGLE fold [(tr,te)]
    rather than faking K-fold -- under nested_cv this means n_outer_folds=1 (honest).
    """
    if split_name == "random":
        stratify = y if task.is_classification else None
        return kfold_indices(len(y), n_folds, seed, stratify=stratify)
    if split_name == "scaffold":
        return scaffold_kfold(df, n_folds, seed)
    if split_name == "nn_distance":
        return [nn_distance_split(df, seed=seed)]
    if split_name == "extrapolation":
        return [target_extrapolation_split(y, direction=extrapolation_direction)]
    raise ValueError(f"unknown split {split_name!r}")


def inner_cv(split_name, task, groups_subset, cv_folds):
    """Return (cv_object, groups_for_fit) for the inner GridSearchCV of one outer fold.

    For scaffold evaluation the inner CV is scaffold-grouped (GroupKFold) using the
    scaffold labels of the outer-train subset, so tuning never leaks a scaffold across
    inner folds. For random evaluation it is the task's standard KFold/StratifiedKFold.
    """
    if split_name == "scaffold":
        n_unique = len(set(groups_subset))
        n_splits = max(2, min(cv_folds, n_unique))
        return GroupKFold(n_splits=n_splits), groups_subset
    return task.make_cv(cv_folds, 0), None


# --------------------------------------------------------------------------- #
# Out-of-distribution splits
# --------------------------------------------------------------------------- #

def _morgan_fps(df, n_bits=1024):
    """Morgan r2 fingerprints as RDKit ExplicitBitVects (for Tanimoto similarity).

    Invalid SMILES yield an all-zero vector so indices stay aligned with df.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fps = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            fps.append(DataStructs.ExplicitBitVect(n_bits))
        else:
            fps.append(gen.GetFingerprint(mol))
    return fps


def nn_distance_split(df, frac=0.2, seed=42, n_bits=1024):
    """Nearest-neighbor distance (dissimilarity) OOD split.

    Grows a training core greedily from an anchor by repeatedly admitting the molecule
    most similar to the current train set. The leftover `frac` of molecules are those
    with the LOWEST maximum Tanimoto similarity to the training set -> an out-of-
    distribution test set that is chemically far from training. The anchor is the medoid
    (max total similarity) for seed 42, else a seeded-random molecule, so seeds vary the
    split while keeping it reproducible. Returns (train_idx, test_idx).
    """
    n = len(df)
    fps = _morgan_fps(df, n_bits)
    n_tr = n - int(np.ceil(frac * n))
    if n_tr <= 0 or n_tr >= n:
        raise ValueError(f"nn_distance_split: degenerate train size {n_tr} for n={n}")

    # similarity of every molecule to each candidate, cached row-by-row as needed
    if seed == 42:
        # medoid: highest total similarity to all others
        totals = np.array([sum(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
                           for i in range(n)])
        anchor = int(np.argmax(totals))
    else:
        anchor = int(np.random.default_rng(seed).integers(0, n))

    in_train = np.zeros(n, dtype=bool)
    in_train[anchor] = True
    # running max Tanimoto of each molecule to the current training set
    max_sim = np.array(DataStructs.BulkTanimotoSimilarity(fps[anchor], fps))
    max_sim[anchor] = -1.0  # exclude already-selected

    while in_train.sum() < n_tr:
        nxt = int(np.argmax(max_sim))
        in_train[nxt] = True
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[nxt], fps))
        max_sim = np.maximum(max_sim, sims)
        max_sim[in_train] = -1.0

    tr = np.where(in_train)[0]
    te = np.where(~in_train)[0]
    return tr, te


def target_extrapolation_split(y, frac=0.2, direction="high"):
    """Target-extrapolation OOD split (regression only).

    Sort by target value; the test set is the top `frac` (direction="high") or the
    bottom `frac` (direction="low"); train is the rest. Deterministic (no seed).
    Returns (train_idx, test_idx).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_te = int(np.ceil(frac * n))
    order = np.argsort(y)  # ascending
    if direction == "high":
        te = order[-n_te:]
    elif direction == "low":
        te = order[:n_te]
    else:
        raise ValueError(f"direction must be 'high' or 'low', got {direction!r}")
    mask = np.ones(n, dtype=bool)
    mask[te] = False
    tr = np.where(mask)[0]
    return tr, np.array(sorted(te))
