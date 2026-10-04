# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

## [0.1.0] - 2026-01-01

Initial release of the `qspr_bench` benchmarking framework.

### Added
- Config-driven benchmark engine for molecular property prediction supporting both
  regression and binary classification, selected per dataset via a `Task` abstraction.
- Dataset registry and generalized loader (any CSV with a SMILES column and a numeric
  target); shipped regression datasets (ESOL, FreeSolv, Lipophilicity) and an opt-in
  downloader (`qspr-bench fetch`) for MoleculeNet/DeepChem classification sets.
- Pluggable model registry with 12 regression / 9 classification scikit-learn and XGBoost
  models, plus GIN, MPNN (PyTorch Geometric) and Chemprop D-MPNN graph baselines as
  optional extras. New models register with a single call; no core edits needed.
- Feature sets (RDKit descriptors, Morgan fingerprints, and their combination) and
  splitting strategies: random, Bemis-Murcko scaffold, nearest-neighbor distance (OOD),
  and target extrapolation (OOD).
- Evaluation modes: single holdout and nested cross-validation (unbiased model selection
  with per-fold and pooled out-of-fold metrics).
- Analysis modules: Friedman + Nemenyi significance testing, split-conformal prediction
  intervals, SHAP interpretability, subgroup (disaggregated) performance, and
  compute-efficiency vs accuracy Pareto analysis.
- Opt-in compute-cost instrumentation (tuning / refit / inference time + peak memory).
- Command-line interface: `run`, `list-datasets`, `list-models`, `fetch`,
  `analyze subgroups`, `analyze efficiency`.
- Test suite (35 tests) and example run configurations.
