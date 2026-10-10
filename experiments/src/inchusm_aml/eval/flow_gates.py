"""Data properties that decide the design of a directed, time-respecting, amount-weighted pattern measure.

Three questions, each answered on labelled laundering attempts (one row per attempt transaction, as
returned by data.amlworld.load_pattern_transactions) and the background transactions:

  amount_gate     Do attempt transactions carry amounts that differ from background ones? Reported as
                  the probability that an attempt amount exceeds a background amount (ties count one
                  half; 0.5 = no difference), and as the background percentile of each attempt's
                  bottleneck (its smallest amount).
  order_gate      Does money enter a pass-through account before it leaves? A pass-through account of
                  an attempt is one that both receives and sends among that attempt's transactions.
                  For each such account, with t_in / t_out the times of its incoming / outgoing attempt
                  transactions:
                    all_in_before_out_le   max(t_in) <= min(t_out)
                    all_in_before_out_lt   max(t_in) <  min(t_out)
                    some_in_before_out_le  min(t_in) <= max(t_out)
                  An attempt satisfies a variant when every pass-through account of it does.
  reduction_gate  How much of the graph and of the attempts survives a degree cap or an amount
                  threshold? An attempt is intact when every one of its transactions survives.

Self-transfers (src == dst) are dropped everywhere. Every rate is reported with its denominator.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

ALL_LAUNDERING = "ALL (Is Laundering = 1)"


def _no_self(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["src"] != frame["dst"]]


def percentile_in(values, reference_sorted: np.ndarray) -> np.ndarray:
    """For each value, the fraction of the reference strictly below it plus half the fraction equal."""
    v = np.asarray(values, dtype=float)
    n = len(reference_sorted)
    if n == 0:
        raise ValueError("empty reference distribution")
    lo = np.searchsorted(reference_sorted, v, side="left")
    hi = np.searchsorted(reference_sorted, v, side="right")
    return (lo + 0.5 * (hi - lo)) / n


def amount_gate(edges: pd.DataFrame, ptx: pd.DataFrame) -> pd.DataFrame:
    """One row per typology, plus one for every transaction labelled as laundering."""
    e = _no_self(edges)
    background = np.sort(e.loc[e["laundering"] == 0, "amount"].to_numpy(dtype=float))
    p = _no_self(ptx)
    rows = []
    for typ, g in p.groupby("type", sort=True):
        bottleneck = g.groupby("attempt")["amount"].min()
        pct = percentile_in(bottleneck, background)
        rows.append({"type": typ, "n_transactions": len(g), "n_attempts": len(bottleneck),
                     "auc_vs_background": float(percentile_in(g["amount"], background).mean()),
                     "bottleneck_pct_q1": float(np.quantile(pct, 0.25)),
                     "bottleneck_pct_median": float(np.median(pct)),
                     "bottleneck_pct_q3": float(np.quantile(pct, 0.75))})
    lab = e.loc[e["laundering"] == 1, "amount"].to_numpy(dtype=float)
    rows.append({"type": ALL_LAUNDERING, "n_transactions": len(lab), "n_attempts": np.nan,
                 "auc_vs_background": float(percentile_in(lab, background).mean()) if len(lab) else np.nan,
                 "bottleneck_pct_q1": np.nan, "bottleneck_pct_median": np.nan, "bottleneck_pct_q3": np.nan})
    out = pd.DataFrame(rows)
    out.attrs["n_background"] = len(background)
    return out


def passthrough_accounts(ptx: pd.DataFrame) -> pd.DataFrame:
    """One row per (attempt, pass-through account) with the order statistics of its transactions."""
    p = _no_self(ptx)
    inc = p.groupby(["attempt", "dst"])["time"].agg(min_in="min", max_in="max")
    inc.index = inc.index.set_names(["attempt", "account"])
    out = p.groupby(["attempt", "src"])["time"].agg(min_out="min", max_out="max")
    out.index = out.index.set_names(["attempt", "account"])
    acc = inc.join(out, how="inner").reset_index()
    types = p.groupby("attempt")["type"].first()
    acc["type"] = acc["attempt"].map(types)
    acc["all_in_before_out_le"] = acc["max_in"] <= acc["min_out"]
    acc["all_in_before_out_lt"] = acc["max_in"] < acc["min_out"]
    acc["some_in_before_out_le"] = acc["min_in"] <= acc["max_out"]
    acc["tie"] = acc["max_in"] == acc["min_out"]
    acc["gap_min"] = acc["min_out"] - acc["max_in"]
    acc["span_min"] = acc["max_out"] - acc["min_in"]
    return acc


VARIANTS = ("all_in_before_out_le", "all_in_before_out_lt", "some_in_before_out_le")


def order_gate(ptx: pd.DataFrame) -> pd.DataFrame:
    """One row per typology: how many attempts have a pass-through account, and how many satisfy each variant."""
    p = _no_self(ptx)
    acc = passthrough_accounts(p)
    per_attempt = acc.groupby("attempt")[list(VARIANTS)].all()
    attempts_by_type = p.groupby("type")["attempt"].nunique()
    rows = []
    for typ, n_att in attempts_by_type.sort_index().items():
        a = acc[acc["type"] == typ]
        sat = per_attempt.loc[per_attempt.index.isin(a["attempt"].unique())]
        row = {"type": typ, "n_attempts": int(n_att), "n_with_passthrough": len(sat),
               "n_passthrough_accounts": len(a)}
        for v in VARIANTS:
            row[f"frac_attempts_{v}"] = float(sat[v].mean()) if len(sat) else np.nan
        row["frac_accounts_tie"] = float(a["tie"].mean()) if len(a) else np.nan
        for name in ("gap_min", "span_min"):
            for q, tag in ((0.25, "q1"), (0.5, "median"), (0.75, "q3")):
                row[f"{name}_{tag}"] = float(a[name].quantile(q)) if len(a) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _intact(ptx: pd.DataFrame, survives: np.ndarray) -> pd.Series:
    """Per attempt: True when every one of its transactions survives."""
    return pd.Series(survives, index=ptx.index).groupby(ptx["attempt"]).all()


def reduction_gate(edges: pd.DataFrame, ptx: pd.DataFrame, caps, removal_fractions,
                   apply_hub_cap: Callable[[pd.DataFrame, int], pd.DataFrame]) -> pd.DataFrame:
    """One row per reduction (none, each degree cap, each amount threshold)."""
    e = _no_self(edges).reset_index(drop=True)
    p = _no_self(ptx).reset_index(drop=True)
    types = sorted(p["type"].unique())
    att_type = p.groupby("attempt")["type"].first()
    amounts = np.sort(e["amount"].to_numpy(dtype=float))
    n_launder = int((e["laundering"] == 1).sum())

    def describe(kind, param, kept: pd.DataFrame, survives: np.ndarray, removed_frac):
        deg = pd.concat([kept["src"], kept["dst"]]).value_counts()
        intact = _intact(p, survives)
        row = {"reduction": kind, "parameter": param, "removed_fraction": removed_frac,
               "n_transactions": len(kept), "n_accounts": int(deg.size),
               "max_degree": int(deg.max()) if deg.size else 0,
               "n_attempts": int(intact.size), "frac_attempts_intact": float(intact.mean()),
               "frac_laundering_transactions_kept":
                   float((kept["laundering"] == 1).sum() / n_launder) if n_launder else np.nan}
        for t in types:
            it = intact[att_type.reindex(intact.index) == t]
            row[f"intact_{t}"] = float(it.mean()) if len(it) else np.nan
        return row

    rows = [describe("none", 0, e, np.ones(len(p), bool), 0.0)]
    for cap in caps:
        kept = apply_hub_cap(e, cap)
        deg = pd.concat([e["src"], e["dst"]]).value_counts()
        hot = set(deg[deg > cap].index)
        survives = ~(p["src"].isin(hot) | p["dst"].isin(hot)).to_numpy()
        rows.append(describe("degree_cap", cap, kept, survives, 1 - len(kept) / len(e)))
    for q in removal_fractions:
        tau = float(amounts[min(int(q * len(amounts)), len(amounts) - 1)])
        kept = e[e["amount"] >= tau]
        survives = (p["amount"] >= tau).to_numpy()
        row = describe("amount_threshold", tau, kept, survives, 1 - len(kept) / len(e))
        row["target_removed_fraction"] = q
        rows.append(row)
    return pd.DataFrame(rows)
