"""Naive baseline for E3: pick centers at random / by vertex degree.

Purpose: show that motif-join's recovery is MEANINGFUL,
i.e. clearly higher than a trivial heuristic at the SAME BUDGET (equal budget = same number
of flagged accounts). If top-degree matches it, the time-flow structure adds nothing;
if motif-join wins (higher recovery OR higher precision at the same recovery),
the velocity/flow constraint has value.

Compare 3 methods at the same budget N for each fan family:
  A. motif-join (windowed, with the deltaT constraint)  -  from baselines.motif_join
  B. top-degree (top-N by out/in/total degree)  -  no temporal constraint
  C. random (N random accounts, averaged over several seeds)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def all_accounts(edges: pd.DataFrame) -> np.ndarray:
    return pd.unique(pd.concat([edges["src"], edges["dst"]], ignore_index=True))


def top_degree_centers(edges: pd.DataFrame, n: int, direction: str) -> set:
    """The N highest-degree accounts  -  the 'degree-only, no temporal constraint' baseline.

    direction: 'out' (fan-out), 'in' (fan-in), 'both' (gather-scatter).
    """
    if direction == "out":
        deg = edges.groupby("src")["dst"].nunique()
    elif direction == "in":
        deg = edges.groupby("dst")["src"].nunique()
    else:  # both: min(in, out)  -  the gather-scatter signature (both gathering and scattering)
        din = edges.groupby("dst")["src"].nunique()
        dout = edges.groupby("src")["dst"].nunique()
        deg = pd.concat([din, dout], axis=1).fillna(0).min(axis=1)
    return set(deg.sort_values(ascending=False).head(n).index)


def random_centers(accounts: np.ndarray, n: int, seed: int) -> set:
    rng = np.random.default_rng(seed)
    n = min(n, len(accounts))
    return set(rng.choice(accounts, size=n, replace=False))


def recovery_of_centers(attempts, centers: set, typ: str) -> float:
    """Fraction of a typology's attempts whose center lies in the centers set."""
    from ..data.amlworld import Attempt  # noqa: F401

    hits = 0
    total = 0
    for a in attempts:
        if a.type != typ:
            continue
        total += 1
        if typ == "FAN-OUT":
            recovered = a.fan_out_center() in centers
        elif typ == "FAN-IN":
            recovered = a.fan_in_center() in centers
        elif typ == "GATHER-SCATTER":
            # set-based, encoding-independent (matches eval.pattern_match)
            recovered = bool(a.gather_scatter_candidates() & centers)
        else:
            continue
        if recovered:
            hits += 1
    return hits / total if total else float("nan")


def precision_proxy_of_centers(edges: pd.DataFrame, centers: set, direction: str) -> float:
    """Fraction of flagged centers with >=1 laundering transaction (out/in per family)."""
    if "laundering" not in edges:
        return float("nan")
    col = "src" if direction == "out" else "dst"
    laund = edges[edges["laundering"] == 1]
    flagged_laund = set(laund[col].unique())
    if not centers:
        return float("nan")
    return len(centers & flagged_laund) / len(centers)
