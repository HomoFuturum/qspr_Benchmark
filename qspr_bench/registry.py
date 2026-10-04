"""Pluggable model registry.

The registry is the single source of truth for which models the benchmark can
assess. Two kinds of model are supported:

  * "tabular"  - a scikit-learn-compatible estimator evaluated on descriptor /
                 fingerprint feature matrices via GridSearchCV. Registered with a
                 `builder(task) -> (estimator, param_grid)`.
  * "graph"    - a model that consumes SMILES directly and runs its own training
                 loop (e.g. a GNN). Registered with a
                 `trainer(smiles, y, tr_idx, te_idx, task, seed, **kw) -> dict`
                 returning at least {"y_pred": ..., "y_score": ...} for te_idx.

Register new models from anywhere (including user code) via register_tabular /
register_graph before calling engine.run_benchmark; the CLI and engine pick them up
automatically. This is what makes the pipeline extensible to new models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .errors import ConfigError
from .task import Task, TaskType

TABULAR = "tabular"
GRAPH = "graph"


@dataclass
class ModelSpec:
    name: str
    kind: str                       # TABULAR | GRAPH
    tasks: frozenset                # task-type string values supported
    builder: Optional[Callable] = None   # tabular: (Task) -> (estimator, param_grid)
    trainer: Optional[Callable] = None    # graph: (...) -> dict
    needs_extra: Optional[str] = None     # name of optional extra, for error messages

    def supports(self, task: Task) -> bool:
        return task.type.value in self.tasks


_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec, overwrite: bool = False) -> ModelSpec:
    if spec.name in _REGISTRY and not overwrite:
        raise ConfigError(f"model {spec.name!r} already registered")
    if spec.kind not in (TABULAR, GRAPH):
        raise ConfigError(f"unknown model kind {spec.kind!r}")
    _REGISTRY[spec.name] = spec
    return spec


def register_tabular(name, tasks, builder, overwrite=False):
    """Register a scikit-learn-style model. `builder(task)->(estimator, param_grid)`."""
    return register(ModelSpec(name=name, kind=TABULAR,
                              tasks=frozenset(tasks), builder=builder),
                    overwrite=overwrite)


def register_graph(name, tasks, trainer, needs_extra=None, overwrite=False):
    """Register a graph/custom model with its own training loop."""
    return register(ModelSpec(name=name, kind=GRAPH, tasks=frozenset(tasks),
                              trainer=trainer, needs_extra=needs_extra),
                    overwrite=overwrite)


def get(name) -> ModelSpec:
    if name not in _REGISTRY:
        raise ConfigError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def names_for_task(task: Task, kind=None):
    return [n for n, s in _REGISTRY.items()
            if s.supports(task) and (kind is None or s.kind == kind)]


def all_specs():
    return dict(_REGISTRY)
