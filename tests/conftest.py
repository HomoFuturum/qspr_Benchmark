"""Shared pytest fixtures: tiny synthetic datasets built from real SMILES.

These keep the suite fast (no network, no large data) while exercising the real
RDKit featurization and the full engine path.
"""
import numpy as np
import pandas as pd
import pytest

# A small set of valid, structurally varied SMILES.
SMILES = [
    "CCO", "CCN", "CCC", "CCCC", "CCCCC", "c1ccccc1", "c1ccccc1C", "c1ccccc1O",
    "CC(=O)O", "CC(=O)N", "CCOC(=O)C", "CC(C)O", "CC(C)CO", "C1CCCCC1",
    "C1CCNCC1", "c1ccncc1", "c1ccc(cc1)N", "c1ccc(cc1)Cl", "CCBr", "CCS",
    "COc1ccccc1", "CC(=O)Nc1ccccc1", "OCC(O)CO", "CCCCCCO", "c1ccc2ccccc2c1",
]


@pytest.fixture
def smiles_list():
    return list(SMILES)


@pytest.fixture
def regression_csv(tmp_path, smiles_list):
    """A regression CSV with a smooth numeric target correlated with length."""
    rng = np.random.default_rng(0)
    y = [len(s) + rng.normal(0, 0.5) for s in smiles_list]
    path = tmp_path / "synth_reg.csv"
    pd.DataFrame({"smiles": smiles_list, "y": y}).to_csv(path, index=False)
    return path


@pytest.fixture
def classification_csv(tmp_path, smiles_list):
    """A binary CSV: aromatic (contains lowercase 'c') vs not, with both classes present."""
    labels = [1 if "c" in s else 0 for s in smiles_list]
    path = tmp_path / "synth_clf.csv"
    pd.DataFrame({"smiles": smiles_list, "label": labels}).to_csv(path, index=False)
    return path
