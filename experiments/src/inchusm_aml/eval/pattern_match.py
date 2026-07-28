"""Pattern-recovery rate (E3)  -  a metric that machine-learning baselines cannot provide.

Operational definition (declared in the paper): a laundering attempt (ground truth)
counts as RECOVERED if the miner returns a SAME-FAMILY motif where:
- FAN-OUT / FAN-IN / GATHER-SCATTER: motif center = attempt center;
- SCATTER-GATHER: motif pair (u, w) = attempt's endpoint pair;
- CYCLE: a mined cycle shares >= min(3, |attempt|) accounts with the attempt.
- RANDOM / BIPARTITE / STACK (journal extension, motif_ext): a mined structure of the same
  family shares >= min(3, |attempt|) accounts with the attempt. When their mined structures
  are not supplied, they report covered=False (the first-paper scope).
"""

from __future__ import annotations

import pandas as pd

from ..data.amlworld import Attempt


def recovery_table(attempts: list[Attempt],
                   fan_out_centers: set,
                   fan_in_centers: set,
                   gs_centers: set,
                   sg_pairs: set,
                   cycles: list[tuple],
                   random_structs: list[frozenset] | None = None,
                   bipartite_structs: list[frozenset] | None = None,
                   stack_structs: list[frozenset] | None = None) -> pd.DataFrame:
    cycle_sets = [set(c) for c in cycles]
    extra = {"RANDOM": random_structs, "BIPARTITE": bipartite_structs, "STACK": stack_structs}
    rows = []
    for a in attempts:
        covered, recovered = True, False
        if a.type == "FAN-OUT":
            recovered = a.fan_out_center() in fan_out_centers
        elif a.type == "FAN-IN":
            recovered = a.fan_in_center() in fan_in_centers
        elif a.type == "GATHER-SCATTER":
            # set-based, encoding-independent: recovered if ANY candidate center is captured
            recovered = bool(a.gather_scatter_candidates() & gs_centers)
        elif a.type == "SCATTER-GATHER":
            us, ws = a.scatter_gather_candidates()
            recovered = any((u, w) in sg_pairs for u in us for w in ws)
        elif a.type == "CYCLE":
            acc = a.accounts
            need = min(3, len(acc))
            recovered = any(len(cs & acc) >= need for cs in cycle_sets)
        elif a.type in extra:
            structs = extra[a.type]
            if structs is None:
                covered = False
            else:
                acc = a.accounts
                need = min(3, len(acc))
                recovered = any(len(s & acc) >= need for s in structs)
        else:
            covered = False
        rows.append({"type": a.type, "covered": covered, "recovered": recovered,
                     "n_accounts": len(a.accounts)})
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("type")
        .agg(n_attempts=("type", "size"),
             covered=("covered", "max"),
             recovered=("recovered", "sum"))
        .reset_index()
    )
    summary["recovery_rate"] = summary.apply(
        lambda r: r["recovered"] / r["n_attempts"] if r["covered"] else float("nan"), axis=1)
    return summary.sort_values("type").reset_index(drop=True)


def precision_summary(mined: pd.DataFrame, laund_col: str, label: str) -> dict:
    """Precision proxy: fraction of mined motifs with >= 1 laundering-labeled transaction
    in their window/structure (labels used only at the evaluation layer)."""
    if mined.empty:
        return {"family": label, "n_mined": 0, "precision_proxy": float("nan")}
    hit = (mined[laund_col] > 0).mean()
    return {"family": label, "n_mined": int(len(mined)), "precision_proxy": float(hit)}
