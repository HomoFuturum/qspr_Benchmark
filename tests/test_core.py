"""Smoke tests for features, splits, and the task abstraction."""
import numpy as np
from rdkit import Chem
from sklearn.model_selection import KFold, StratifiedKFold

from qspr_bench import features
from qspr_bench.splits import scaffold_split, variance_prepare
from qspr_bench.task import TaskType, make_task


def _mols(smiles):
    return [Chem.MolFromSmiles(s) for s in smiles]


def test_descriptors_and_fps(smiles_list):
    mols = _mols(smiles_list)
    Xd = features.calc_descriptors(mols)
    assert Xd.shape[0] == len(mols)
    assert Xd.shape[1] > 100
    assert not np.isinf(Xd.to_numpy(dtype=float, na_value=0.0)).any()
    Xf = features.calc_fps(mols, n_bits=128)
    assert Xf.shape == (len(mols), 128)


def test_metric_keys():
    reg = features.regression_metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    assert set(reg) == {"R2", "RMSE", "MAE"}
    clf = features.classification_metrics([0, 1, 1, 0], [0, 1, 0, 0],
                                          [0.2, 0.9, 0.4, 0.1])
    assert set(clf) == {"ROC_AUC", "PR_AUC", "ACC", "F1"}


def test_task_cv_types():
    assert make_task("regression").type is TaskType.REGRESSION
    assert isinstance(make_task("regression").make_cv(3, 0), KFold)
    assert isinstance(make_task("classification").make_cv(3, 0), StratifiedKFold)


def test_scaffold_split_disjoint_and_complete(regression_csv):
    import pandas as pd
    df = pd.read_csv(regression_csv).rename(columns={"y": "target"})
    tr, te = scaffold_split(df, frac=0.3)
    assert len(set(tr) & set(te)) == 0
    assert len(tr) + len(te) == len(df)
    assert len(te) > 0 and len(tr) > 0


def test_variance_prepare_drops_constant_and_fills(smiles_list):
    import pandas as pd
    mols = _mols(smiles_list)
    X = features.calc_descriptors(mols)
    X["__const__"] = 1.0  # zero-variance column should be dropped
    tr = np.arange(0, 20)
    te = np.arange(20, len(mols))
    Xtr, Xte, n_feat = variance_prepare(X, tr, te)
    assert "__const__" not in Xtr.columns
    assert n_feat == Xtr.shape[1]
    assert not Xtr.isna().any().any()
    assert not Xte.isna().any().any()
