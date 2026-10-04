"""Dataset specification and generalized loader.

Replaces the ESOL-hardcoded features.load_data with a config-driven loader usable
for any CSV that has a SMILES column and a numeric target column.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from .errors import ConfigError
from .task import TaskType, make_task

RDLogger.DisableLog("rdApp.*")


@dataclass
class DatasetSpec:
    name: str
    file: Path
    smiles_col: str
    target_col: str
    task: str  # "regression" | "classification"

    def __post_init__(self):
        self.file = Path(self.file)
        # validate the task string eagerly
        make_task(self.task)


def load_dataset(spec: DatasetSpec, root: Path | None = None):
    """Load a dataset into a tidy DataFrame + parallel list of RDKit mols.

    Returns (df, mols) where df has canonical columns 'smiles' and 'target', with
    invalid SMILES dropped and the index reset. For classification, the target is
    validated to be binary {0,1} (after coercion). Rows with NaN targets are dropped.
    """
    path = spec.file
    if root is not None and not path.is_absolute():
        path = Path(root) / path
    if not path.exists():
        raise ConfigError(f"dataset file not found for {spec.name!r}: {path}")

    raw = pd.read_csv(path)
    for col in (spec.smiles_col, spec.target_col):
        if col not in raw.columns:
            raise ConfigError(
                f"dataset {spec.name!r}: column {col!r} not in {path.name} "
                f"(have: {list(raw.columns)[:12]}{'...' if len(raw.columns) > 12 else ''})"
            )

    raw = raw.rename(columns={spec.smiles_col: "smiles", spec.target_col: "target"})
    raw = raw[["smiles", "target"]].copy()
    raw["target"] = pd.to_numeric(raw["target"], errors="coerce")

    mols, keep = [], []
    for i, smi in enumerate(raw["smiles"]):
        if not isinstance(smi, str):
            continue
        m = Chem.MolFromSmiles(smi)
        if m is not None and not pd.isna(raw["target"].iloc[i]):
            mols.append(m)
            keep.append(i)
    df = raw.iloc[keep].reset_index(drop=True)
    if len(df) == 0:
        raise ConfigError(f"dataset {spec.name!r}: no valid (SMILES, target) rows")

    if make_task(spec.task).type is TaskType.CLASSIFICATION:
        uniq = set(np.unique(df["target"].values))
        if not uniq.issubset({0.0, 1.0}):
            raise ConfigError(
                f"dataset {spec.name!r} is classification but target has non-binary "
                f"values {sorted(uniq)[:6]}; expected 0/1 labels"
            )
        df["target"] = df["target"].astype(int)

    return df, mols
