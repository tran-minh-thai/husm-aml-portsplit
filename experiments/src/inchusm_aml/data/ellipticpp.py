"""Load the Elliptic++ actor (wallet) graph with real monetary features.

This is the journal-only data layer. Unlike the transaction-level Elliptic loader, it
reads the wallet-to-wallet graph together with per-wallet monetary attributes, so that
utility can be derived from real transaction value instead of a discretized-label proxy.

Files (under datasets/ellipticpp/):
- wallets_features.csv : per-wallet features incl. Time step and monetary columns
                         (btc_transacted_total, fees_as_share_mean, ...).
- wallets_classes.csv  : per-wallet class (1=illicit, 2=licit, 3/unknown).
- AddrAddr_edgelist.csv : directed wallet-to-wallet edges (the actor graph).

Convention kept from the earlier work: statistics used to bucket the monetary features are
fit on the training window only and then frozen; labels are used only at the evaluation
layer, never in mining. The standard temporal split is Time step 1..34 = train.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_MAX_STEP = 34

# Monetary columns kept for utility derivation (a focused subset of the 57 wallet features).
MONETARY_COLS = [
    "btc_transacted_total",
    "btc_sent_total",
    "btc_received_total",
    "fees_total",
    "fees_as_share_mean",
    "total_txs",
]


def _default_root() -> Path:
    return Path(__file__).resolve().parents[4] / "datasets" / "ellipticpp"


@dataclass
class EllipticppData:
    """Wallet node table (step, label, monetary features) and the wallet-to-wallet edges."""

    wallets: pd.DataFrame          # columns: address, step, label, + MONETARY_COLS
    edges: pd.DataFrame            # columns: src, dst (wallet addresses)

    def split_by_step(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(train, test) wallets under the standard temporal split (step <= 34 = train)."""
        w = self.wallets
        return w[w["step"] <= TRAIN_MAX_STEP], w[w["step"] > TRAIN_MAX_STEP]


def load_ellipticpp(root: str | Path | None = None) -> EllipticppData:
    """Load the actor graph and per-wallet monetary features.

    Building the utility from the monetary columns is done separately
    (see inchusm_aml.utility.monetary) so the fit-on-train / freeze step is explicit.
    """
    root = Path(root) if root is not None else _default_root()

    feats = pd.read_csv(
        root / "wallets_features.csv",
        usecols=["address", "Time step", *MONETARY_COLS],
    ).rename(columns={"Time step": "step"})
    feats["step"] = feats["step"].astype(np.int16)
    for c in MONETARY_COLS:
        feats[c] = feats[c].astype(np.float32)
    # a wallet can appear in several steps; keep its earliest appearance for a stable split
    feats = feats.sort_values("step").drop_duplicates("address", keep="first")

    classes = pd.read_csv(root / "wallets_classes.csv")
    label_map = {"1": 1, "2": 0}   # illicit -> 1, licit -> 0, everything else -> -1
    classes["label"] = classes["class"].astype(str).map(label_map).fillna(-1).astype(np.int8)

    wallets = feats.merge(classes[["address", "label"]], on="address", how="left")
    wallets["label"] = wallets["label"].fillna(-1).astype(np.int8)

    edges = pd.read_csv(
        root / "AddrAddr_edgelist.csv",
        usecols=["input_address", "output_address"],
    ).rename(columns={"input_address": "src", "output_address": "dst"})

    return EllipticppData(wallets=wallets, edges=edges)


def prefilter_actor_graph(edges: pd.DataFrame, labeled: set, hub_cap: int) -> pd.DataFrame:
    """Drop self-loops, edges touching unlabeled wallets, and high-degree hubs.

    High-degree wallets are typically exchanges or services that touch most of the graph
    and make mining explode; removing them keeps the mule-wallet structure that laundering
    relies on (same rationale as the motif-join prefilter).
    """
    e = edges[edges["src"] != edges["dst"]]
    e = e[e["src"].isin(labeled) & e["dst"].isin(labeled)]
    deg = pd.concat([e["src"], e["dst"]]).value_counts()
    hubs = set(deg[deg > hub_cap].index)
    return e[~e["src"].isin(hubs) & ~e["dst"].isin(hubs)]


def build_actor_stream(edges: pd.DataFrame, node_label: pd.Series, step_of: pd.Series):
    """Split the prefiltered actor graph into an add-only snapshot stream by Time step.

    An edge is placed in the snapshot of its later-appearing endpoint (max of the two
    first-appearance steps), so every edge appears exactly once and the union grows
    monotonically. Yields (step, node_labels, edges) for each non-empty step in order.

    Each snapshot is exported as its OWN graph (export_lg numbers vertices per snapshot), so
    wallet identity is NOT carried across snapshots: a wallet active in k periods becomes k
    unrelated actors. That is what makes the snapshots disjoint, so Lemma L1 and everything
    resting on it (exact maintenance, the sliding window) do apply to this stream.

    The cost is a modelling choice worth stating: the mined patterns are WITHIN-PERIOD
    structures, not lifetime structures, so a laundering campaign spread over several periods
    is split into per-period fragments rather than surfacing as one high-utility pattern.
    """
    s_src = edges["src"].map(step_of)
    s_dst = edges["dst"].map(step_of)
    edge_step = pd.concat([s_src, s_dst], axis=1).max(axis=1).astype("Int64")
    for t in sorted(x for x in edge_step.dropna().unique()):
        e_t = edges[edge_step == t]
        nodes = pd.Index(pd.concat([e_t["src"], e_t["dst"]]).unique())
        labels_t = node_label[node_label.index.isin(nodes)]
        yield int(t), labels_t, e_t
