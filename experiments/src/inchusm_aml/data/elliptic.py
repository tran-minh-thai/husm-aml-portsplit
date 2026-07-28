"""Load the Elliptic dataset (49 disjoint snapshots) under the strict-inductive protocol.

Conventions:
- Standard split: train = step 1..34, test = step 35..49.
- Labels: class '1' = illicit -> 1, '2' = licit -> 0, 'unknown' -> -1.
- Features: 165 columns (drop the time-step column); Weber's "AF" = 166 because it folds in the time step.
- No cross-time-step edges (verified by directly counting all 234,355 edges).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_MAX_STEP = 34
N_FEATURES = 165


def _default_root() -> Path:
    return Path(__file__).resolve().parents[4] / "datasets" / "elliptic1"


@dataclass
class EllipticData:
    """Node table (txId, step, label, f0..f164) + edge table (src, dst, step)."""

    nodes: pd.DataFrame
    edges: pd.DataFrame

    @property
    def feature_cols(self) -> list[str]:
        return [f"f{i}" for i in range(N_FEATURES)]

    def labeled(self) -> pd.DataFrame:
        return self.nodes[self.nodes["label"] >= 0]

    def split_labeled(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(train, test) with labeled nodes only, standard temporal split."""
        lab = self.labeled()
        return (
            lab[lab["step"] <= TRAIN_MAX_STEP],
            lab[lab["step"] > TRAIN_MAX_STEP],
        )

    def snapshot(self, step: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(nodes, edges) of one snapshot  -  input to the miner (E1/E2)."""
        return (
            self.nodes[self.nodes["step"] == step],
            self.edges[self.edges["step"] == step],
        )


def load_elliptic(root: str | Path | None = None) -> EllipticData:
    root = Path(root) if root is not None else _default_root()

    feats = pd.read_csv(root / "elliptic_txs_features.csv", header=None)
    feats.columns = ["txId", "step"] + [f"f{i}" for i in range(feats.shape[1] - 2)]
    feats["step"] = feats["step"].astype(np.int16)
    for c in feats.columns[2:]:
        feats[c] = feats[c].astype(np.float32)

    classes = pd.read_csv(root / "elliptic_txs_classes.csv")
    label_map = {"1": 1, "2": 0, "unknown": -1}
    classes["label"] = classes["class"].astype(str).map(label_map).astype(np.int8)

    nodes = feats.merge(classes[["txId", "label"]], on="txId", how="left")
    nodes["label"] = nodes["label"].fillna(-1).astype(np.int8)

    edges = pd.read_csv(root / "elliptic_txs_edgelist.csv")
    edges.columns = ["src", "dst"]
    step_of = nodes.set_index("txId")["step"]
    edges["step"] = step_of.reindex(edges["src"]).to_numpy(dtype=np.int16)

    return EllipticData(nodes=nodes, edges=edges)


def sanity_check(data: EllipticData) -> dict:
    """Cross-check against Weber 2019 / Maganti 2026  -  run before every experiment."""
    lab = data.labeled()
    train, test = data.split_labeled()
    cross = (
        data.edges["step"]
        != data.nodes.set_index("txId")["step"].reindex(data.edges["dst"]).to_numpy()
    ).sum()
    return {
        "n_nodes": len(data.nodes),                 # expected 203_769
        "n_edges": len(data.edges),                 # expected 234_355
        "n_illicit": int((lab["label"] == 1).sum()),  # expected 4_545
        "n_licit": int((lab["label"] == 0).sum()),    # expected 42_019
        "n_train_labeled": len(train),              # expected 29_894 (Maganti B.2)
        "n_test_labeled": len(test),                # expected 16_670
        "n_cross_step_edges": int(cross),           # expected 0
    }
