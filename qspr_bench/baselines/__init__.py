"""Optional heavy-model baselines (GIN/MPNN via PyTorch Geometric, Chemprop D-MPNN).

All torch/torch_geometric/chemprop/lightning imports are performed lazily inside the
functions that need them so that importing this subpackage never requires the heavy
extras. Request these models only after installing qspr_bench[gnn] or qspr_bench[chemprop].
"""
