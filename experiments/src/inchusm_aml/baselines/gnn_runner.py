"""E4+ (journal): strict-inductive GNN detection baselines on Elliptic.

GCN / GraphSAGE trained under the SAME strict-inductive protocol as the RF gate: the model
sees ONLY the subgraph of time steps <= 34 at training time (nodes and edges), and is applied
to the test-period subgraph (steps 35..49) at inference. Elliptic has no cross-time-step
edges, so the two subgraphs are disjoint and the setup is genuinely inductive: no test-period
structure leaks into training.

Reframing (the first paper's controlled null result). These baselines are a REFERENCE, not a
target to beat: under a clean protocol the RF gate already matches them. The contribution of
E4+ is the protocol and the drift analysis, not a new detection score. Under a transductive
setup GNNs look much stronger, but that gain is leakage (Maganti 2026).

EvolveGCN needs torch_geometric_temporal; it is not implemented here. When it (or
torch_geometric itself) is unavailable, the driver reports the RF gate and cites the
published same-protocol GNN numbers instead of inventing them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..eval.metrics import per_step_f1, zone_summary

SUPPORTED = ("gcn", "sage")


def available() -> bool:
    """True if PyTorch Geometric is importable (the GNN baselines can actually run)."""
    try:
        import torch_geometric  # noqa: F401
        return True
    except ImportError:
        return False


def _build(data):
    """Tensors + the two disjoint subgraphs (train: steps <= 34, test: steps >= 35)."""
    import torch

    nodes = data.nodes
    idx = pd.Series(np.arange(len(nodes)), index=nodes["txId"].to_numpy())
    x = torch.tensor(nodes[data.feature_cols].to_numpy(dtype=np.float32))
    y_np = nodes["label"].to_numpy()
    step = nodes["step"].to_numpy()

    e = data.edges
    src = idx.reindex(e["src"].to_numpy()).to_numpy()
    dst = idx.reindex(e["dst"].to_numpy()).to_numpy()
    ok = ~(np.isnan(src) | np.isnan(dst))
    src, dst = src[ok].astype(np.int64), dst[ok].astype(np.int64)

    train_node = step <= 34
    test_node = step >= 35
    # an edge belongs to a subgraph only when BOTH endpoints do (no leakage across the split)
    tr = train_node[src] & train_node[dst]
    te = test_node[src] & test_node[dst]

    def undirected(s, d):
        ei = np.vstack([np.concatenate([s, d]), np.concatenate([d, s])])
        return torch.tensor(ei, dtype=torch.long)

    ei_train = undirected(src[tr], dst[tr])
    ei_test = undirected(src[te], dst[te])

    y = torch.tensor(np.where(y_np < 0, 0, y_np), dtype=torch.long)   # unknown -> 0, masked out anyway
    train_mask = torch.tensor((step <= 34) & (y_np >= 0))
    test_mask = torch.tensor((step >= 35) & (y_np >= 0))
    return x, y, ei_train, ei_test, train_mask, test_mask, step, y_np


def _net(model: str, in_dim: int, hidden: int):
    import torch
    from torch_geometric.nn import GCNConv, SAGEConv

    conv = GCNConv if model == "gcn" else SAGEConv

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = conv(in_dim, hidden)
            self.c2 = conv(hidden, 2)

        def forward(self, x, edge_index):
            h = torch.relu(self.c1(x, edge_index))
            h = torch.dropout(h, p=0.5, train=self.training)
            return self.c2(h, edge_index)

    return Net()


def run_gnn_gate(data, model: str = "gcn", seeds=(0, 1, 2), epochs: int = 200,
                 hidden: int = 64, lr: float = 0.01, weight_decay: float = 5e-4) -> dict:
    """Train a strict-inductive GNN and return per-step F1 + zones, shaped like run_rf_gate."""
    if not available():
        raise ImportError("torch_geometric not installed; cite the same-protocol GNN numbers.")
    if model not in SUPPORTED:
        raise NotImplementedError(
            f"model '{model}' not implemented here (supported: {', '.join(SUPPORTED)}); "
            "EvolveGCN needs torch_geometric_temporal -- cite Maganti 2026 instead."
        )

    import torch
    import torch.nn.functional as F
    from sklearn.metrics import f1_score

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, ei_train, ei_test, train_mask, test_mask, step, y_np = _build(data)
    x, y = x.to(dev), y.to(dev)
    ei_train, ei_test = ei_train.to(dev), ei_test.to(dev)
    train_mask, test_mask = train_mask.to(dev), test_mask.to(dev)

    per_seed: dict[int, float] = {}
    preds = []
    for s in seeds:
        torch.manual_seed(int(s))
        net = _net(model, x.size(1), hidden).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            net.train()
            opt.zero_grad()
            out = net(x, ei_train)                    # training graph only
            F.cross_entropy(out[train_mask], y[train_mask]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(x, ei_test).argmax(dim=1)      # inference on the test-period subgraph
        p = pred[test_mask].cpu().numpy()
        t = y[test_mask].cpu().numpy()
        per_seed[int(s)] = float(f1_score(t, p, pos_label=1, zero_division=0))
        preds.append(pred.cpu().numpy())

    med = int(np.argsort([per_seed[int(s)] for s in seeds])[len(list(seeds)) // 2])
    pred_med = preds[med]
    tm = test_mask.cpu().numpy()
    vals = np.array(list(per_seed.values()))
    return {
        "model": model,
        "n_features": int(x.size(1)),
        "f1_mean": float(vals.mean()),
        "f1_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "per_seed_f1": per_seed,
        "zones": zone_summary(step[tm], y_np[tm], pred_med[tm]),
        "per_step": per_step_f1(step[tm], y_np[tm], pred_med[tm]),
    }
