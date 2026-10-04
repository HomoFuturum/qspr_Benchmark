"""GIN and MPNN graph-neural-network baselines (PyTorch Geometric).

Ported from the paper's gnn_baseline.py and generalized to both regression and
classification. All torch/torch_geometric imports are performed lazily inside
functions so importing this module does not require the 'gnn' extra until a GNN
model is actually used. Importing the module registers GIN and MPNN with the
shared model registry.

Install: pip install qspr_bench[gnn]
"""
from __future__ import annotations

import numpy as np

from ..errors import MissingDependencyError
from ..task import TaskType

ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]
N_ATOM_FEAT = len(ATOMS) + 5
N_BOND_FEAT = 4 + 2  # 4 bond types + conjugated + in-ring


def _torch():
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            "GIN/MPNN need the 'gnn' extra: pip install qspr_bench[gnn]"
        ) from e
    import torch
    return torch


def _atom_feat(a):
    from rdkit import Chem  # noqa: F401
    z = [0.0] * len(ATOMS)
    if a.GetAtomicNum() in ATOMS:
        z[ATOMS.index(a.GetAtomicNum())] = 1.0
    other = [0.0] if a.GetAtomicNum() in ATOMS else [1.0]
    return z + other + [a.GetTotalDegree() / 6.0, float(a.GetIsAromatic()),
                        a.GetFormalCharge() / 4.0, a.GetTotalNumHs() / 4.0]


def _bond_feat(b):
    from rdkit import Chem
    bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    t = [0.0] * len(bonds)
    if b.GetBondType() in bonds:
        t[bonds.index(b.GetBondType())] = 1.0
    return t + [float(b.GetIsConjugated()), float(b.IsInRing())]


def _mol_to_graph(smi, y):
    import torch
    from rdkit import Chem
    from torch_geometric.data import Data
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    xf = [_atom_feat(a) for a in mol.GetAtoms()]
    ef, ei = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = _bond_feat(b)
        ei += [[i, j], [j, i]]
        ef += [bf, bf]
    if not ei:
        ei = [[0, 0]]
        ef = [[0.0] * N_BOND_FEAT]
    return Data(x=torch.tensor(xf, dtype=torch.float32),
                edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(ef, dtype=torch.float32),
                y=torch.tensor([float(y)], dtype=torch.float32))


def _build_nets():
    """Return (GINNet, MPNNNet) classes; defined lazily so torch is imported late."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import CGConv, GINConv, global_add_pool

    class GINNet(nn.Module):
        def __init__(self, hid=128, layers=4):
            super().__init__()
            self.convs = nn.ModuleList()
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(N_ATOM_FEAT, hid), nn.ReLU(), nn.Linear(hid, hid))))
            for _ in range(layers - 1):
                self.convs.append(GINConv(nn.Sequential(
                    nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid))))
            self.head = nn.Linear(hid, 1)

        def forward(self, data):
            h = data.x
            for c in self.convs:
                h = F.relu(c(h, data.edge_index))
            return self.head(global_add_pool(h, data.batch)).squeeze(-1)

    class MPNNNet(nn.Module):
        def __init__(self, hid=128, layers=4):
            super().__init__()
            self.inp = nn.Linear(N_ATOM_FEAT, hid)
            self.convs = nn.ModuleList([CGConv(hid, dim=N_BOND_FEAT)
                                        for _ in range(layers)])
            self.head = nn.Linear(hid, 1)

        def forward(self, data):
            h = F.relu(self.inp(data.x))
            for c in self.convs:
                h = c(h, data.edge_index, data.edge_attr)
            return self.head(global_add_pool(h, data.batch)).squeeze(-1)

    return GINNet, MPNNNet


def _train(net_name, smiles, y, tr_idx, te_idx, task, seed, epochs=200, patience=25):
    """Train one GNN and return predictions on te_idx.

    Regression: MSE on standardized targets; predictions de-standardized.
    Classification: BCEWithLogits on raw head output; y_score = sigmoid(logit).
    """
    torch = _torch()
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    is_clf = task.type is TaskType.CLASSIFICATION

    graphs = [_mol_to_graph(s, yy) for s, yy in zip(smiles, np.asarray(y))]
    if any(g is None for g in graphs):
        raise ValueError("GNN featurization produced a None graph for a valid SMILES")
    ys = np.asarray(y, dtype=np.float64)

    tr_i, te_i = np.asarray(tr_idx), np.asarray(te_idx)
    if is_clf:
        mu, sd = 0.0, 1.0
    else:
        mu, sd = ys[tr_i].mean(), ys[tr_i].std() + 1e-8
    for g, raw in zip(graphs, ys):
        g.y = torch.tensor([(raw - mu) / sd], dtype=torch.float32).view(1)

    val_i = rng.choice(tr_i, size=max(1, int(0.1 * len(tr_i))), replace=False)
    fit_i = np.setdiff1d(tr_i, val_i)
    ld_fit = DataLoader([graphs[i] for i in fit_i], batch_size=128, shuffle=True)
    ld_val = DataLoader([graphs[i] for i in val_i], batch_size=256)
    ld_te = DataLoader([graphs[i] for i in te_i], batch_size=256)

    GINNet, MPNNNet = _build_nets()
    net_cls = GINNet if net_name == "GIN" else MPNNNet
    model = net_cls().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    def raw_out(ld):
        model.eval()
        out = []
        with torch.no_grad():
            for b in ld:
                out.append(model(b.to(dev)).cpu().numpy())
        return np.concatenate(out)

    best_val, best_state, bad, ep = np.inf, None, 0, 0
    for ep in range(epochs):
        model.train()
        for b in ld_fit:
            b = b.to(dev)
            pred = model(b)
            target = b.y.view(-1)
            loss = F.binary_cross_entropy_with_logits(pred, target) if is_clf \
                else F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # validation loss (same objective)
        vo = raw_out(ld_val)
        vy = ys[val_i]
        if is_clf:
            p = 1.0 / (1.0 + np.exp(-vo))
            eps = 1e-7
            v = float(-np.mean(vy * np.log(p + eps) + (1 - vy) * np.log(1 - p + eps)))
        else:
            v = float(np.mean((vy - (vo * sd + mu)) ** 2))
        if v < best_val - 1e-6:
            best_val, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    out = raw_out(ld_te)
    if is_clf:
        score = 1.0 / (1.0 + np.exp(-out))
        return {"y_pred": (score >= 0.5).astype(int), "y_score": score,
                "info": {"epoch": ep}}
    return {"y_pred": out * sd + mu, "y_score": None, "info": {"epoch": ep}}


def _make_trainer(net_name):
    def trainer(smiles, y, tr_idx, te_idx, task, seed, **kw):
        return _train(net_name, smiles, y, tr_idx, te_idx, task, seed, **kw)
    return trainer


def register():
    from ..registry import register_graph
    tasks = ["regression", "classification"]
    register_graph("GIN", tasks, _make_trainer("GIN"), needs_extra="gnn", overwrite=True)
    register_graph("MPNN", tasks, _make_trainer("MPNN"), needs_extra="gnn", overwrite=True)


register()
