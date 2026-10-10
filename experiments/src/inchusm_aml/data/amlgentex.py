"""Load an AMLGentex transaction log (Östman et al. 2025; huggingface.co/datasets/AMLGentex).

Format of tx_log.csv: step (day), type (TRANSFER, CASH, INITALBALANCE), amount, nameOrig, nameDest,
bank and balance columns, isSAR, alertID, modelType. Account -2 is the simulator's income source and
-1 its spending sink; rows touching either are money entering or leaving the simulated network, not
transfers between accounts, and are dropped. A laundering attempt is the set of rows sharing an
alertID >= 0 with isSAR = 1; rows with isSAR = 1 and alertID = -1 are other activity of flagged
accounts. Background = isSAR = 0.

modelType codes of the laundering patterns, as in src/utils/pattern_types.py of the AMLGentex
repository and confirmed on sweden_100K_easy and sweden_100K_difficult by the structure of every
attempt (code 1 has a single sender, 2 a single receiver, 3 only pass-through accounts with equal
in- and out-degree, 4 no pass-through account, 7 one source and one sink around many intermediaries,
8 many sources and sinks around one centre):
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SAR_TYPES = {1: "FAN-OUT", 2: "FAN-IN", 3: "CYCLE", 4: "BIPARTITE", 5: "STACK", 6: "RANDOM",
             7: "SCATTER-GATHER", 8: "GATHER-SCATTER"}
SPECIAL_ACCOUNTS = (-2, -1)
COLUMNS = ["step", "type", "amount", "nameOrig", "nameDest", "isSAR", "alertID", "modelType"]


def load_amlgentex(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (edges, ptx) in the shapes the gate measurements use.

    edges: src, dst, amount, time (days), laundering (isSAR), for transfers between accounts.
    ptx:   attempt (alertID), type (name from SAR_TYPES), src, dst, time, amount, one row per
           transaction of every laundering attempt.
    attrs of edges: rows_total, rows_special (dropped), and sar_without_alert (isSAR = 1 rows with
    alertID = -1, kept in edges as laundering but in no attempt). Raises ValueError on an unknown
    laundering modelType, so a renumbered release cannot be read under the wrong names.
    """
    df = pd.read_csv(path, usecols=COLUMNS)
    special = df["nameOrig"].isin(SPECIAL_ACCOUNTS) | df["nameDest"].isin(SPECIAL_ACCOUNTS)
    t = df[~special]
    edges = pd.DataFrame({"src": t["nameOrig"].to_numpy(), "dst": t["nameDest"].to_numpy(),
                          "amount": t["amount"].to_numpy(dtype=float), "time": t["step"].to_numpy(),
                          "laundering": t["isSAR"].to_numpy(dtype="int8")})
    a = t[(t["isSAR"] == 1) & (t["alertID"] >= 0)]
    unknown = sorted(set(a["modelType"]) - set(SAR_TYPES))
    if unknown:
        raise ValueError(f"{path}: laundering modelType codes {unknown} are not in {sorted(SAR_TYPES)}")
    ptx = pd.DataFrame({"attempt": a["alertID"].to_numpy(), "type": a["modelType"].map(SAR_TYPES).to_numpy(),
                        "src": a["nameOrig"].to_numpy(), "dst": a["nameDest"].to_numpy(),
                        "time": a["step"].to_numpy(), "amount": a["amount"].to_numpy(dtype=float)})
    edges.attrs.update(rows_total=len(df), rows_special=int(special.sum()),
                       sar_without_alert=int(((t["isSAR"] == 1) & (t["alertID"] < 0)).sum()))
    return edges, ptx
