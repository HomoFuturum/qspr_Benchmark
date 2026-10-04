"""Config loading and validation.

Two config kinds:
  * a dataset registry (configs/datasets.yaml) mapping names -> DatasetSpec fields.
  * a run config describing one benchmark (datasets by name, models, splits, etc.).

load_run_config resolves a run config against the registry into a RunConfig, applying
optional CLI overrides, and validates the whole thing (including the rule that a run
may not mix regression and classification datasets).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .data import DatasetSpec
from .errors import ConfigError
from .models import HEAVY_MODELS
from .task import make_task

VALID_FEATURE_SETS = {"Desc", "Desc+FP", "FP"}
VALID_SPLITS = {"random", "scaffold", "nn_distance", "extrapolation"}
VALID_EVALUATIONS = {"holdout", "nested_cv"}
VALID_EXTRAPOLATION_DIRECTIONS = {"high", "low"}


@dataclass
class RunConfig:
    name: str
    datasets: list            # list[DatasetSpec]
    models: list
    feature_sets: list
    splits: list
    test_frac: float = 0.2
    cv_folds: int = 5
    seeds: list = field(default_factory=lambda: [42])
    fp_bits: int = 1024
    random_state: int = 42
    output_dir: Path = Path("results")
    evaluation: str = "holdout"      # "holdout" | "nested_cv"
    outer_folds: int = 5             # outer K for nested CV
    save_predictions: bool = False   # write per-molecule predictions.csv
    extrapolation_direction: str = "high"  # for the "extrapolation" OOD split
    measure_cost: bool = False       # record per-model time + peak memory

    @property
    def task_type(self):
        """The single task type shared by all datasets in this run."""
        return make_task(self.datasets[0].task).type


def _read_mapping(path):
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def load_registry(registry_path):
    """Load datasets.yaml -> {name: DatasetSpec}."""
    data = _read_mapping(registry_path)
    entries = data.get("datasets", data)
    if not isinstance(entries, dict):
        raise ConfigError(f"{registry_path}: expected a 'datasets' mapping")
    registry = {}
    for name, cfg in entries.items():
        try:
            registry[name] = DatasetSpec(
                name=name,
                file=cfg["file"],
                smiles_col=cfg["smiles_col"],
                target_col=cfg["target_col"],
                task=cfg["task"],
            )
        except KeyError as e:
            raise ConfigError(f"dataset {name!r} missing required key {e}") from None
    return registry


def _apply_overrides(run, overrides):
    """overrides: dict with any of datasets/models/splits/feature_sets/seeds/output_dir."""
    if not overrides:
        return run
    mapping = {
        "datasets": "datasets",
        "models": "models",
        "splits": "splits",
        "feature_sets": "feature_sets",
        "seeds": "seeds",
        "output_dir": "output_dir",
        "evaluation": "evaluation",
        "outer_folds": "outer_folds",
        "extrapolation_direction": "extrapolation_direction",
        "measure_cost": "measure_cost",
    }
    for key, dest in mapping.items():
        if overrides.get(key) is not None:
            run[dest] = overrides[key]
    return run


def load_run_config(path, registry_path="configs/datasets.yaml", overrides=None,
                    root=None):
    """Resolve a run config file into a validated RunConfig."""
    doc = _read_mapping(path)
    run = dict(doc.get("run", doc))
    run = _apply_overrides(run, overrides)

    registry = load_registry(registry_path)

    for req in ("name", "datasets", "models", "feature_sets", "splits"):
        if req not in run:
            raise ConfigError(f"run config missing required key {req!r}")

    # resolve dataset names -> DatasetSpec
    specs = []
    for ds_name in run["datasets"]:
        if ds_name not in registry:
            raise ConfigError(
                f"unknown dataset {ds_name!r}; known: {sorted(registry)}"
            )
        specs.append(registry[ds_name])

    # single-task rule
    task_types = {s.task for s in specs}
    if len(task_types) > 1:
        raise ConfigError(
            f"run config mixes task types {sorted(task_types)}; a run must be all "
            f"regression or all classification"
        )
    task = make_task(specs[0].task)

    # validate feature sets / splits
    bad_fs = set(run["feature_sets"]) - VALID_FEATURE_SETS
    if bad_fs:
        raise ConfigError(f"invalid feature_sets {sorted(bad_fs)}; allowed {sorted(VALID_FEATURE_SETS)}")
    bad_sp = set(run["splits"]) - VALID_SPLITS
    if bad_sp:
        raise ConfigError(f"invalid splits {sorted(bad_sp)}; allowed {sorted(VALID_SPLITS)}")

    evaluation = str(run.get("evaluation", "holdout"))
    if evaluation not in VALID_EVALUATIONS:
        raise ConfigError(
            f"invalid evaluation {evaluation!r}; allowed {sorted(VALID_EVALUATIONS)}"
        )

    extrap_dir = str(run.get("extrapolation_direction", "high"))
    if extrap_dir not in VALID_EXTRAPOLATION_DIRECTIONS:
        raise ConfigError(
            f"invalid extrapolation_direction {extrap_dir!r}; "
            f"allowed {sorted(VALID_EXTRAPOLATION_DIRECTIONS)}"
        )
    if "extrapolation" in run["splits"] and task.is_classification:
        raise ConfigError(
            "the 'extrapolation' split is regression-only (it partitions by target "
            "magnitude); remove it for classification datasets"
        )

    # validate models against the registry (built-in tabular models self-register at
    # import; heavy baselines register lazily, so allow their known names too).
    from . import registry
    known = set(registry.names_for_task(task)) | HEAVY_MODELS
    bad_m = [m for m in run["models"] if m not in known]
    if bad_m:
        raise ConfigError(
            f"unknown models {bad_m} for task {task.type.value}; "
            f"known: {sorted(registry.names_for_task(task))} "
            f"(+ heavy {sorted(HEAVY_MODELS)})"
        )

    out = run.get("output_dir", "results")
    if root is not None and not Path(out).is_absolute():
        out = Path(root) / out

    # save_predictions defaults on for nested_cv (needed for subgroup analysis)
    save_predictions = bool(run.get("save_predictions", evaluation == "nested_cv"))

    return RunConfig(
        name=run["name"],
        datasets=specs,
        models=list(run["models"]),
        feature_sets=list(run["feature_sets"]),
        splits=list(run["splits"]),
        test_frac=float(run.get("test_frac", 0.2)),
        cv_folds=int(run.get("cv_folds", 5)),
        seeds=list(run.get("seeds", [42])),
        fp_bits=int(run.get("fp_bits", 1024)),
        random_state=int(run.get("random_state", 42)),
        output_dir=Path(out),
        evaluation=evaluation,
        outer_folds=int(run.get("outer_folds", 5)),
        save_predictions=save_predictions,
        extrapolation_direction=extrap_dir,
        measure_cost=bool(run.get("measure_cost", False)),
    )
