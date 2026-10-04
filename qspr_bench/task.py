"""Task abstraction: the regression/classification polymorphism seam.

A frozen Task bundles everything that differs between the two task types so the
engine can stay task-agnostic. Build one per dataset via make_task(spec.task).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sklearn.model_selection import KFold, StratifiedKFold

from . import features
from .errors import ConfigError


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


@dataclass(frozen=True)
class Task:
    type: TaskType
    gridsearch_scoring: str          # single-metric string for GridSearchCV refit
    cv_scoring: dict                 # multi-metric dict for cross_validate
    metric_names: tuple             # ordered metric keys produced by metrics_of
    needs_proba: bool                # whether evaluation needs predict_proba scores

    def make_cv(self, n_splits, random_state):
        cls = KFold if self.type is TaskType.REGRESSION else StratifiedKFold
        return cls(n_splits=n_splits, shuffle=True, random_state=random_state)

    def metrics_of(self, y_true, y_pred=None, y_score=None):
        if self.type is TaskType.REGRESSION:
            return features.regression_metrics(y_true, y_pred)
        return features.classification_metrics(y_true, y_pred, y_score)

    @property
    def is_classification(self):
        return self.type is TaskType.CLASSIFICATION


_REGRESSION = Task(
    type=TaskType.REGRESSION,
    gridsearch_scoring="neg_root_mean_squared_error",
    cv_scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error"},
    metric_names=("R2", "RMSE", "MAE"),
    needs_proba=False,
)

_CLASSIFICATION = Task(
    type=TaskType.CLASSIFICATION,
    gridsearch_scoring="roc_auc",
    cv_scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision",
                "acc": "accuracy", "f1": "f1"},
    metric_names=("ROC_AUC", "PR_AUC", "ACC", "F1"),
    needs_proba=True,
)


def make_task(task_type) -> Task:
    """Factory. Accepts a TaskType or its string value."""
    value = task_type.value if isinstance(task_type, TaskType) else str(task_type)
    if value == TaskType.REGRESSION.value:
        return _REGRESSION
    if value == TaskType.CLASSIFICATION.value:
        return _CLASSIFICATION
    raise ConfigError(
        f"unknown task type {task_type!r}; expected 'regression' or 'classification'"
    )
