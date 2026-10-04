"""qspr_bench: universal config-driven QSPR benchmark.

Core modules import only lightweight dependencies (numpy, pandas, scikit-learn,
xgboost, rdkit, pyyaml). Heavy optional baselines (torch/torch_geometric/chemprop)
and analysis extras (shap/matplotlib/scikit-posthocs) are lazy-imported inside their
own modules and are never pulled in by importing this package.
"""
from .task import Task, TaskType, make_task
from .data import DatasetSpec, load_dataset
from .config import RunConfig, load_run_config
from .engine import run_benchmark

__version__ = "0.1.0"

__all__ = [
    "Task",
    "TaskType",
    "make_task",
    "DatasetSpec",
    "load_dataset",
    "RunConfig",
    "load_run_config",
    "run_benchmark",
    "__version__",
]
