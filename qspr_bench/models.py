"""Model registry with hyperparameter grids, task-aware.

Ported from models_grid.py. get_models(task) returns regressor OR classifier
variants. Scaler-wrapped pipelines (via pipe) are reused verbatim for the models
that need standardized features; tree models are used bare.
"""
from __future__ import annotations

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from .errors import ConfigError
from .task import Task, TaskType

RS = 42


def pipe(est):
    return Pipeline([("sc", StandardScaler()), ("m", est)])


def _regressors():
    models = {
        "MLR": pipe(LinearRegression()),
        "Ridge": pipe(Ridge(random_state=RS)),
        "LASSO": pipe(Lasso(random_state=RS, max_iter=50000)),
        "KNN": pipe(KNeighborsRegressor()),
        "SVR": pipe(SVR()),
        "MLP": pipe(MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=2000,
                                 early_stopping=True, random_state=RS)),
        "RF": RandomForestRegressor(n_estimators=500, random_state=RS, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=RS, n_jobs=-1),
        "GBDT": GradientBoostingRegressor(random_state=RS),
        "HistGBDT": HistGradientBoostingRegressor(random_state=RS),
        "XGBoost": XGBRegressor(random_state=RS, n_jobs=-1, tree_method="hist"),
        "ElasticNet": pipe(ElasticNet(random_state=RS, max_iter=50000)),
    }
    params = {
        "Ridge": {"m__alpha": [0.1, 1.0, 10.0]},
        "LASSO": {"m__alpha": [0.001, 0.01, 0.1]},
        "ElasticNet": {"m__alpha": [0.001, 0.01, 0.1],
                       "m__l1_ratio": [0.2, 0.5, 0.8]},
        "KNN": {"m__n_neighbors": [3, 5, 7, 10],
                "m__weights": ["uniform", "distance"]},
        "SVR": {"m__C": [1, 10, 100], "m__gamma": ["scale", 0.01, 0.1],
                "m__epsilon": [0.05, 0.1]},
        "MLP": {"m__alpha": [1e-4, 1e-3], "m__learning_rate_init": [1e-3, 5e-4]},
        "RF": {"n_estimators": [300, 800], "max_depth": [None, 16, 24],
               "min_samples_leaf": [1, 2]},
        "ExtraTrees": {"n_estimators": [300, 800], "max_depth": [None, 16, 24],
                       "min_samples_leaf": [1, 2]},
        "GBDT": {"n_estimators": [300, 800], "learning_rate": [0.03, 0.1],
                 "max_depth": [3, 5]},
        "HistGBDT": {"learning_rate": [0.03, 0.1], "max_iter": [300, 800],
                     "max_leaf_nodes": [31, 63]},
        "XGBoost": {"n_estimators": [400, 900], "learning_rate": [0.03, 0.08],
                    "max_depth": [4, 6, 8], "subsample": [0.8, 1.0]},
    }
    return models, params


def _classifiers():
    # LogReg replaces the three linear regressors; grids mirror the regression side
    # wherever a parameter has a classification analogue (tree depths, C/gamma, k).
    models = {
        "LogReg": pipe(LogisticRegression(random_state=RS, max_iter=5000)),
        "KNN": pipe(KNeighborsClassifier()),
        "SVC": pipe(SVC(probability=True, random_state=RS)),
        "MLP": pipe(MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=2000,
                                  early_stopping=True, random_state=RS)),
        "RF": RandomForestClassifier(n_estimators=500, random_state=RS, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=500, random_state=RS, n_jobs=-1),
        "GBDT": GradientBoostingClassifier(random_state=RS),
        "HistGBDT": HistGradientBoostingClassifier(random_state=RS),
        "XGBoost": XGBClassifier(random_state=RS, n_jobs=-1, tree_method="hist",
                                 eval_metric="logloss"),
    }
    params = {
        "LogReg": {"m__C": [0.1, 1.0, 10.0], "m__penalty": ["l2"]},
        "KNN": {"m__n_neighbors": [3, 5, 7, 10],
                "m__weights": ["uniform", "distance"]},
        "SVC": {"m__C": [1, 10, 100], "m__gamma": ["scale", 0.01, 0.1]},
        "MLP": {"m__alpha": [1e-4, 1e-3], "m__learning_rate_init": [1e-3, 5e-4]},
        "RF": {"n_estimators": [300, 800], "max_depth": [None, 16, 24],
               "min_samples_leaf": [1, 2]},
        "ExtraTrees": {"n_estimators": [300, 800], "max_depth": [None, 16, 24],
                       "min_samples_leaf": [1, 2]},
        "GBDT": {"n_estimators": [300, 800], "learning_rate": [0.03, 0.1],
                 "max_depth": [3, 5]},
        "HistGBDT": {"learning_rate": [0.03, 0.1], "max_iter": [300, 800],
                     "max_leaf_nodes": [31, 63]},
        "XGBoost": {"n_estimators": [400, 900], "learning_rate": [0.03, 0.08],
                    "max_depth": [4, 6, 8], "subsample": [0.8, 1.0]},
    }
    return models, params


# Retained for back-compat / messaging: heavy baselines live in the baselines
# subpackage and register themselves (as GRAPH models) when imported.
HEAVY_MODELS = {"GIN", "MPNN", "ChempropDMPNN"}


def get_models(task: Task):
    """Return (models_dict, params_dict) for the given task's sklearn/xgboost models."""
    if task.type is TaskType.REGRESSION:
        return _regressors()
    if task.type is TaskType.CLASSIFICATION:
        return _classifiers()
    raise ConfigError(f"no model registry for task {task.type!r}")


def available_model_names(task: Task):
    models, _ = get_models(task)
    return list(models.keys())


def _make_builder(model_name):
    """Builder closure: (Task) -> (estimator, param_grid) for one tabular model."""
    def builder(task: Task):
        models, params = get_models(task)
        return models[model_name], params.get(model_name, {})
    return builder


def register_builtin_tabular():
    """Register the built-in sklearn/xgboost models into the shared registry.

    Each model is registered under the tasks for which it exists. Idempotent:
    safe to call more than once (overwrite=True).
    """
    from .registry import register_tabular
    reg_names = set(_regressors()[0])
    clf_names = set(_classifiers()[0])
    for name in reg_names | clf_names:
        tasks = []
        if name in reg_names:
            tasks.append(TaskType.REGRESSION.value)
        if name in clf_names:
            tasks.append(TaskType.CLASSIFICATION.value)
        register_tabular(name, tasks, _make_builder(name), overwrite=True)


# Register built-ins at import time so the registry is populated as soon as the
# package is imported (qspr_bench/__init__ imports engine -> models).
register_builtin_tabular()
