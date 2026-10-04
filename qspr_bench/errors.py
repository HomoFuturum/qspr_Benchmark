"""Shared exceptions for qspr_bench."""


class QsprBenchError(Exception):
    """Base class for all qspr_bench errors."""


class ConfigError(QsprBenchError):
    """Raised when a dataset/run configuration is invalid."""


class MissingDependencyError(QsprBenchError):
    """Raised when an optional extra (gnn/chemprop/analysis) is required but absent."""
