"""Chemprop D-MPNN baseline (directed message-passing network).

Ported from the paper's chemprop_baseline.py and generalized to regression and
binary classification. Registers "ChempropDMPNN" as a graph model. All chemprop /
lightning imports are lazy; install qspr_bench[chemprop] to use it.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..errors import MissingDependencyError
from ..task import TaskType

warnings.filterwarnings("ignore")


def _imports():
    try:
        import torch  # noqa: F401
        from lightning import pytorch as pl  # noqa: F401
        from chemprop import data, featurizers, models, nn  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            "ChempropDMPNN needs the 'chemprop' extra: pip install qspr_bench[chemprop]"
        ) from e
    import torch
    from lightning import pytorch as pl
    from chemprop import data, featurizers, models, nn
    return torch, pl, data, featurizers, models, nn


def _build_dataset(data, featurizers, smis, ys):
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dps = [data.MoleculeDatapoint.from_smi(s, np.array([y], dtype=float))
           for s, y in zip(smis, ys)]
    return data.MoleculeDataset(dps, feat)


def _train(smiles, y, tr_idx, te_idx, task, seed, epochs=200, batch=64):
    torch, pl, data, featurizers, models, nn = _imports()
    pl.seed_everything(seed, verbose=False)
    accel = "gpu" if torch.cuda.is_available() else "cpu"
    is_clf = task.type is TaskType.CLASSIFICATION

    smis = np.asarray(smiles)
    ys = np.asarray(y, dtype=float)
    tr_idx, te_idx = np.asarray(tr_idx), np.asarray(te_idx)
    rng = np.random.default_rng(seed)
    val_i = rng.choice(tr_idx, size=max(1, int(0.1 * len(tr_idx))), replace=False)
    fit_i = np.setdiff1d(tr_idx, val_i)

    train_ds = _build_dataset(data, featurizers, smis[fit_i], ys[fit_i])
    val_ds = _build_dataset(data, featurizers, smis[val_i], ys[val_i])
    test_ds = _build_dataset(data, featurizers, smis[te_idx], ys[te_idx])

    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    if is_clf:
        ffn = nn.BinaryClassificationFFN()
        model = models.MPNN(mp, agg, ffn, batch_norm=True,
                            metrics=[nn.metrics.BinaryAUROC()])
    else:
        scaler = train_ds.normalize_targets()  # standardize targets on train only
        val_ds.normalize_targets(scaler)
        out_tf = nn.UnscaleTransform.from_standard_scaler(scaler)
        ffn = nn.RegressionFFN(output_transform=out_tf)
        model = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=[nn.metrics.RMSE()])

    train_loader = data.build_dataloader(train_ds, batch_size=batch, num_workers=0)
    val_loader = data.build_dataloader(val_ds, batch_size=batch, num_workers=0,
                                       shuffle=False)
    test_loader = data.build_dataloader(test_ds, batch_size=batch, num_workers=0,
                                        shuffle=False)

    early = pl.callbacks.EarlyStopping(monitor="val_loss", patience=25, mode="min")
    trainer = pl.Trainer(accelerator=accel, devices=1, max_epochs=epochs,
                         enable_checkpointing=False, enable_progress_bar=False,
                         logger=False, callbacks=[early], enable_model_summary=False)
    trainer.fit(model, train_loader, val_loader)

    preds = np.concatenate([p.numpy().ravel() for p in trainer.predict(model, test_loader)])
    if is_clf:
        score = preds  # BinaryClassificationFFN outputs probabilities
        return {"y_pred": (score >= 0.5).astype(int), "y_score": score, "info": {}}
    return {"y_pred": preds, "y_score": None, "info": {}}


def register():
    from ..registry import register_graph
    register_graph("ChempropDMPNN", ["regression", "classification"], _train,
                   needs_extra="chemprop", overwrite=True)


register()
