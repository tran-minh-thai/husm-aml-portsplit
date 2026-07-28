"""Per-transaction AML motif features on the Elliptic graph (for the RF+/-motif branch, the motif-mining work).

Each transaction (node) gets a STRUCTURAL feature vector describing its role in the
local money-laundering typologies (fan-in/out, gather-scatter, cycle, smurf), computed
DIRECTLY from the graph of the snapshot that contains it.

No leakage: Elliptic has no cross-time-step edges, so every feature of
a node depends only on that node's snapshot  -  this is a deterministic per-node transform,
of the same nature as the 165 raw features (each node carries its own features), NOT a
statistic fit on the test set. A test node draws its features from its own neighborhood
structure at inference time  -  in the spirit of strict-inductive.

Typology mapping:
  fan_out / fan_in            -> dout / din and threshold flags
  gather-scatter (smurf T1)   -> gather_scatter (din>=2 & dout>=2)
  scatter-gather / smurf T2   -> pred_dout_max, succ_din_max (adjacent to a fan hub)
  cycle (money returning)     -> recip_flag (2-cycle), triangle_flag (3-cycle)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MOTIF_COLS = [
    "m_din", "m_dout", "m_deg",
    "m_fan_in", "m_fan_out", "m_gather_scatter",
    "m_pred_dout_max", "m_pred_dout_sum",
    "m_succ_din_max", "m_succ_din_sum",
    "m_two_hop_out", "m_two_hop_in",
    "m_recip", "m_triangle",
]


def compute_motif_features(edges: pd.DataFrame, node_ids: pd.Index,
                           fan_k: int = 2) -> pd.DataFrame:
    """Return a DataFrame index=txId (following node_ids), columns MOTIF_COLS.

    edges: [src, dst] directed (src -> dst = money flow), includes a step column but
           it is not needed since edges do not cross steps.
    """
    src = edges["src"].to_numpy()
    dst = edges["dst"].to_numpy()

    din = pd.Series(dst).value_counts()
    dout = pd.Series(src).value_counts()

    def deg(series, ids):
        return series.reindex(ids).fillna(0).to_numpy(dtype=np.int32)

    din_arr = deg(din, node_ids)
    dout_arr = deg(dout, node_ids)

    # successor / predecessor adjacency (dict node -> list)
    succ: dict[int, list[int]] = {}
    pred: dict[int, list[int]] = {}
    for s, d in zip(src, dst):
        succ.setdefault(s, []).append(d)
        pred.setdefault(d, []).append(s)

    din_map = din.to_dict()
    dout_map = dout.to_dict()

    pred_dout_max = np.zeros(len(node_ids), np.int32)
    pred_dout_sum = np.zeros(len(node_ids), np.int32)
    succ_din_max = np.zeros(len(node_ids), np.int32)
    succ_din_sum = np.zeros(len(node_ids), np.int32)
    two_hop_out = np.zeros(len(node_ids), np.int32)
    two_hop_in = np.zeros(len(node_ids), np.int32)
    recip = np.zeros(len(node_ids), np.int8)
    triangle = np.zeros(len(node_ids), np.int8)

    pos = {node: i for i, node in enumerate(node_ids)}
    for node, i in pos.items():
        ss = succ.get(node, ())
        ps = pred.get(node, ())
        if ps:
            douts = [dout_map.get(p, 0) for p in ps]
            pred_dout_max[i] = max(douts)
            pred_dout_sum[i] = sum(douts)
        if ss:
            dins = [din_map.get(s, 0) for s in ss]
            succ_din_max[i] = max(dins)
            succ_din_sum[i] = sum(dins)
            two_hop_out[i] = sum(dout_map.get(s, 0) for s in ss)
        if ps:
            two_hop_in[i] = sum(din_map.get(p, 0) for p in ps)
        # cycles through node: 2-cycle (v->u->v) and directed triangle (v->a->b->v)
        succ_set = set(ss)
        if succ_set:
            if any(node in succ.get(u, ()) for u in succ_set):
                recip[i] = 1
            found_tri = False
            for a in succ_set:
                for b in succ.get(a, ()):
                    if node in succ.get(b, ()):
                        found_tri = True
                        break
                if found_tri:
                    break
            triangle[i] = 1 if found_tri else 0

    out = pd.DataFrame(index=node_ids)
    out["m_din"] = din_arr
    out["m_dout"] = dout_arr
    out["m_deg"] = din_arr + dout_arr
    out["m_fan_in"] = (din_arr >= fan_k).astype(np.int8)
    out["m_fan_out"] = (dout_arr >= fan_k).astype(np.int8)
    out["m_gather_scatter"] = ((din_arr >= fan_k) & (dout_arr >= fan_k)).astype(np.int8)
    out["m_pred_dout_max"] = pred_dout_max
    out["m_pred_dout_sum"] = pred_dout_sum
    out["m_succ_din_max"] = succ_din_max
    out["m_succ_din_sum"] = succ_din_sum
    out["m_two_hop_out"] = two_hop_out
    out["m_two_hop_in"] = two_hop_in
    out["m_recip"] = recip
    out["m_triangle"] = triangle
    return out
