"""Command-line interface for qspr_bench.

Subcommands:
  run           run a benchmark from a config file (with optional inline overrides)
  list-datasets show the dataset registry
  list-models   show available models for a task
  fetch         download classification datasets (opt-in network step)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_registry, load_run_config
from .errors import QsprBenchError
from .models import HEAVY_MODELS, get_models
from .task import make_task

DEFAULT_REGISTRY = "configs/datasets.yaml"


def _csv_list(value):
    """Parse a comma-separated flag value into a list of trimmed strings."""
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _int_list(value):
    items = _csv_list(value)
    return [int(v) for v in items] if items is not None else None


def cmd_run(args):
    overrides = {
        "datasets": _csv_list(args.dataset),
        "models": _csv_list(args.models),
        "splits": _csv_list(args.splits),
        "feature_sets": _csv_list(args.features),
        "seeds": _int_list(args.seeds),
        "output_dir": args.output_dir,
        "evaluation": args.evaluation,
        "outer_folds": args.outer_folds,
        "extrapolation_direction": args.extrapolation_direction,
        "measure_cost": True if args.measure_cost else None,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    cfg = load_run_config(args.config, registry_path=args.registry, overrides=overrides)

    print(f"run: {cfg.name}")
    print(f"  task        : {cfg.task_type.value}")
    print(f"  evaluation  : {cfg.evaluation}"
          + (f" (outer_folds={cfg.outer_folds})" if cfg.evaluation == "nested_cv" else ""))
    print(f"  datasets    : {[d.name for d in cfg.datasets]}")
    print(f"  models      : {cfg.models}")
    print(f"  feature_sets: {cfg.feature_sets}")
    print(f"  splits      : {cfg.splits}")
    print(f"  seeds       : {cfg.seeds}  cv_folds={cfg.cv_folds}  test_frac={cfg.test_frac}")
    print(f"  output_dir  : {cfg.output_dir}  save_predictions={cfg.save_predictions}")
    if args.dry_run:
        print("\n[dry-run] config resolved and valid; not executing.")
        return 0

    from .engine import run_benchmark  # imported here to keep startup light
    run_benchmark(cfg)
    return 0


def cmd_analyze(args):
    out_dir = Path(args.results_dir)
    preds = out_dir / "predictions.csv"
    meta = out_dir / "dataset_meta.csv"
    if args.what == "subgroups":
        if not preds.exists():
            print(f"error: {preds} not found. Run with evaluation=nested_cv or "
                  f"save_predictions=true first.", file=sys.stderr)
            return 2
        from .analysis.subgroups import run_from_csv
        report = run_from_csv(preds, meta, out_dir=out_dir)
        print(f"wrote {out_dir / 'subgroups.csv'} ({len(report)} rows)")
        return 0
    if args.what == "efficiency":
        results = out_dir / "results.csv"
        if not results.exists():
            print(f"error: {results} not found.", file=sys.stderr)
            return 2
        from .analysis.efficiency import run_from_csv
        try:
            table = run_from_csv(results, meta, out_dir=out_dir, cost_col=args.cost_col)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        n_opt = int(table["is_pareto_optimal"].sum()) if len(table) else 0
        print(f"wrote {out_dir / 'efficiency.csv'} ({len(table)} rows, "
              f"{n_opt} pareto-optimal)")
        return 0
    print(f"error: unknown analysis {args.what!r}", file=sys.stderr)
    return 2


def cmd_list_datasets(args):
    registry = load_registry(args.registry)
    print(f"{'NAME':<16}{'TASK':<16}{'FILE':<28}EXISTS")
    for name, spec in registry.items():
        exists = "yes" if Path(spec.file).exists() else "NO"
        print(f"{name:<16}{spec.task:<16}{str(spec.file):<28}{exists}")
    return 0


def cmd_list_models(args):
    # import baseline subpackages so their graph models self-register (best-effort:
    # skip silently if the heavy extra is not installed)
    for mod in ("qspr_bench.baselines.gnn", "qspr_bench.baselines.chemprop"):
        try:
            __import__(mod)
        except Exception:
            pass
    from . import registry

    tasks = [args.task] if args.task else ["regression", "classification"]
    for t in tasks:
        task = make_task(t)
        models, params = get_models(task)
        print(f"\n[{t}] tabular (sklearn/xgboost) models:")
        for name in models:
            grid = params.get(name, {})
            keys = ", ".join(grid.keys()) if grid else "(no grid)"
            print(f"  {name:<14} {keys}")
        graph = registry.names_for_task(task, kind=registry.GRAPH)
        if graph:
            print(f"[{t}] graph models (own training loop):")
            for name in graph:
                extra = registry.get(name).needs_extra
                tag = f"  (extra: {extra})" if extra else ""
                print(f"  {name:<14}{tag}")
    missing = sorted(HEAVY_MODELS - set(registry.all_specs()))
    if missing:
        print(f"\nheavy baselines not available (extra not installed): {missing}")
    return 0


def cmd_fetch(args):
    from .fetch import fetch  # lazy: only needed for this command
    for name in args.datasets:
        path = fetch(name, dest=args.dest, force=args.force)
        print(f"  -> {path}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="qspr-bench",
                                description="Universal QSPR/QSAR benchmark.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a benchmark from a config file")
    r.add_argument("config", help="path to a run config (YAML or JSON)")
    r.add_argument("--registry", default=DEFAULT_REGISTRY, help="dataset registry path")
    r.add_argument("--dataset", help="override datasets (comma-separated names)")
    r.add_argument("--models", help="override models (comma-separated)")
    r.add_argument("--splits", help="override splits (comma-separated: random,scaffold)")
    r.add_argument("--features", help="override feature sets (Desc,Desc+FP,FP)")
    r.add_argument("--seeds", help="override seeds (comma-separated ints)")
    r.add_argument("--output-dir", help="override output directory")
    r.add_argument("--evaluation", choices=["holdout", "nested_cv"], default=None,
                   help="override evaluation mode")
    r.add_argument("--outer-folds", type=int, default=None,
                   help="override outer-fold count for nested CV")
    r.add_argument("--extrapolation-direction", choices=["high", "low"], default=None,
                   help="direction for the 'extrapolation' OOD split")
    r.add_argument("--measure-cost", action="store_true",
                   help="record per-model train/predict time and peak memory")
    r.add_argument("--dry-run", action="store_true",
                   help="resolve and validate config, then exit without running")
    r.set_defaults(func=cmd_run)

    an = sub.add_parser("analyze", help="post-hoc analysis of a results directory")
    an.add_argument("what", choices=["subgroups", "efficiency"], help="analysis to run")
    an.add_argument("results_dir", help="directory with results.csv / predictions.csv + dataset_meta.csv")
    an.add_argument("--cost-col", default="refit_seconds",
                    help="cost column for efficiency Pareto (default refit_seconds)")
    an.set_defaults(func=cmd_analyze)

    ld = sub.add_parser("list-datasets", help="list the dataset registry")
    ld.add_argument("--registry", default=DEFAULT_REGISTRY)
    ld.set_defaults(func=cmd_list_datasets)

    lm = sub.add_parser("list-models", help="list available models for a task")
    lm.add_argument("--task", choices=["regression", "classification"], default=None)
    lm.set_defaults(func=cmd_list_models)

    f = sub.add_parser("fetch", help="download classification datasets (network)")
    f.add_argument("datasets", nargs="+", help="dataset names, e.g. BBBP BACE")
    f.add_argument("--dest", default="data", help="destination directory (default data/)")
    f.add_argument("--force", action="store_true", help="overwrite existing files")
    f.set_defaults(func=cmd_fetch)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except QsprBenchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
