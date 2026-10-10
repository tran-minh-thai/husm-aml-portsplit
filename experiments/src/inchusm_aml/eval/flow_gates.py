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


# ------------------------------------------------------------------ second round
#
# Two hereditary time constraints on an occurrence (dropping transactions from a valid occurrence
# leaves a valid one, so a support counted on valid occurrences stays anti-monotone):
#   (a) order with tolerance k: at most k head-to-tail pairs (u->v at t1, v->w at t2) with t1 > t2;
#       variant a1 counts every pair, a2 skips back-and-forth pairs (w = u);
#   (b) window: max(t) - min(t) <= delta.
# A directed cycle forces at least one reversed pair, so cycles are reported but judged apart.

def head_to_tail_pairs(ptx: pd.DataFrame) -> pd.DataFrame:
    """Every ordered pair of distinct transactions of one attempt where the first ends where the second starts."""
    p = _no_self(ptx).reset_index(drop=True)
    p = p.assign(tx=np.arange(len(p)))
    first = p[["attempt", "type", "tx", "src", "dst", "time", "amount"]].rename(
        columns={"tx": "tx1", "src": "u", "dst": "v", "time": "t1", "amount": "a1"})
    second = p[["attempt", "tx", "src", "dst", "time", "amount"]].rename(
        columns={"tx": "tx2", "src": "v", "dst": "w", "time": "t2", "amount": "a2"})
    pairs = first.merge(second, on=["attempt", "v"])
    pairs = pairs[pairs["tx1"] != pairs["tx2"]].reset_index(drop=True)
    pairs["back_and_forth"] = pairs["w"] == pairs["u"]
    pairs["in_order"] = pairs["t1"] <= pairs["t2"]
    pairs["gap"] = pairs["t2"] - pairs["t1"]
    return pairs


KS = (0, 1, 2, 3)


def order_tolerance_gate(ptx: pd.DataFrame) -> pd.DataFrame:
    """Per typology: pair-level in-order rates (a1, a2) and attempt-level shares with at most k reversed pairs."""
    p = _no_self(ptx)
    pairs = head_to_tail_pairs(p)
    pairs["reversed_a1"] = ~pairs["in_order"]
    pairs["reversed_a2"] = ~pairs["in_order"] & ~pairs["back_and_forth"]
    v = pairs.groupby("attempt")[["reversed_a1", "reversed_a2"]].sum()
    attempts = p.groupby("attempt")["type"].first()
    v = v.reindex(attempts.index, fill_value=0)          # attempts without any pair have V = 0
    has_pair = pd.Series(attempts.index.isin(pairs["attempt"].unique()), index=attempts.index)
    rows = []
    for typ in sorted(attempts.unique()):
        ids = attempts.index[attempts == typ]
        pt = pairs[pairs["type"] == typ]
        pt2 = pt[~pt["back_and_forth"]]
        row = {"type": typ, "n_attempts": len(ids), "n_attempts_with_pair": int(has_pair[ids].sum()),
               "n_pairs_a1": len(pt), "in_order_a1": float(pt["in_order"].mean()) if len(pt) else np.nan,
               "n_pairs_a2": len(pt2), "in_order_a2": float(pt2["in_order"].mean()) if len(pt2) else np.nan,
               "n_back_and_forth": int(pt["back_and_forth"].sum())}
        sel = has_pair[ids]
        for var in ("a1", "a2"):
            vv = v.loc[ids[sel.to_numpy()], f"reversed_{var}"]
            for k in KS:
                row[f"attempts_V_le_{k}_{var}"] = float((vv <= k).mean()) if len(vv) else np.nan
            row[f"attempts_V_max_{var}"] = int(vv.max()) if len(vv) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def window_gate(ptx: pd.DataFrame, deltas, unit: float) -> pd.DataFrame:
    """Per typology: attempt durations and the share of attempts no longer than each delta.

    deltas are in days; unit is the number of time units per day (1440 for minutes, 1 for days)."""
    p = _no_self(ptx)
    dur = p.groupby("attempt")["time"].agg(lambda t: t.max() - t.min()) / unit
    attempts = p.groupby("attempt")["type"].first()
    pairs = head_to_tail_pairs(p)
    rows = []
    for typ in sorted(attempts.unique()):
        d = dur[attempts == typ]
        g = pairs.loc[pairs["type"] == typ, "gap"].abs() / unit
        row = {"type": typ, "n_attempts": len(d),
               "duration_days_q1": float(d.quantile(0.25)), "duration_days_median": float(d.median()),
               "duration_days_q3": float(d.quantile(0.75)), "duration_days_max": float(d.max()),
               "pair_gap_days_median": float(g.median()) if len(g) else np.nan}
        for delta in deltas:
            row[f"within_{delta}d"] = float((d <= delta).mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- third round
#
# Amount conservation along a head-to-tail pair (u->v of a1, v->w of a2, w != u): the deviation
# l = |ln(a2 / a1)| is 0 when the amount passes on unchanged. A pair keeps the amount within theta when
# (1 - theta) <= a2 / a1 <= (1 + theta). A per-pair predicate, so requiring it of every pair of an
# occurrence is hereditary. Background pairs are sampled: a uniformly random background transaction
# e2 = v->w, then a uniformly random background transaction e1 entering v, redrawn when v has none or
# when e1 = w->v (back-and-forth).

THETAS = (0.05, 0.10, 0.20, 0.50)
BACKGROUND_LABEL = "BACKGROUND (sampled pairs)"
ALL_PAIRS_LABEL = "ALL laundering pairs"


def deviation(a1, a2) -> np.ndarray:
    return np.abs(np.log(np.asarray(a2, dtype=float) / np.asarray(a1, dtype=float)))


def auc_lower(values, reference_sorted: np.ndarray) -> float:
    """P(value < reference) + 1/2 P(value = reference), averaged over values."""
    return float(np.mean(1.0 - percentile_in(values, reference_sorted)))


def sample_background_pairs(edges: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n head-to-tail pairs of background transactions; attrs report draws and rejections."""
    e = _no_self(edges)
    e = e[e["laundering"] == 0].reset_index(drop=True)
    src, dst = e["src"].to_numpy(), e["dst"].to_numpy()
    amount = e["amount"].to_numpy(dtype=float)
    by_dst = np.argsort(dst, kind="stable")
    dst_sorted = dst[by_dst]
    rng = np.random.default_rng(seed)
    first, second = [], []
    draws = no_incoming = back = 0
    got = 0
    while got < n:
        batch = max(2 * (n - got), 1024)
        e2 = rng.integers(0, len(e), size=batch)
        v = src[e2]
        lo = np.searchsorted(dst_sorted, v, side="left")
        hi = np.searchsorted(dst_sorted, v, side="right")
        cnt = hi - lo
        has = cnt > 0
        e1 = np.full(batch, -1)
        e1[has] = by_dst[lo[has] + np.floor(rng.random(has.sum()) * cnt[has]).astype(np.int64)]
        bf = has & (src[np.where(has, e1, 0)] == dst[e2])
        ok = has & ~bf
        draws += batch
        no_incoming += int((~has).sum())
        back += int(bf.sum())
        take = np.flatnonzero(ok)[: n - got]
        first.append(e1[take])
        second.append(e2[take])
        got += len(take)
    i1, i2 = np.concatenate(first), np.concatenate(second)
    out = pd.DataFrame({"u": src[i1], "v": dst[i1], "w": dst[i2], "a1": amount[i1], "a2": amount[i2],
                        "tx1": i1, "tx2": i2})
    out.attrs.update(draws=draws, rejected_no_incoming=no_incoming, rejected_back_and_forth=back,
                     n_background_transactions=len(e))
    return out


def _conservation_row(label, a1, a2, background_dev_sorted=None):
    a1, a2 = np.asarray(a1, dtype=float), np.asarray(a2, dtype=float)
    ok = (a1 > 0) & (a2 > 0)
    a1, a2 = a1[ok], a2[ok]
    r = a2 / a1
    d = np.abs(np.log(r))
    row = {"group": label, "n_pairs": int(len(r)), "n_nonpositive_excluded": int((~ok).sum()),
           "auc_conservation": auc_lower(d, background_dev_sorted)
           if background_dev_sorted is not None and len(d) else np.nan,
           "frac_decreasing": float((a2 <= a1).mean()) if len(r) else np.nan}
    for th in THETAS:
        row[f"within_{th:g}"] = float(((r >= 1 - th) & (r <= 1 + th)).mean()) if len(r) else np.nan
    for q, tag in ((0.25, "q1"), (0.5, "median"), (0.75, "q3")):
        row[f"ratio_{tag}"] = float(np.quantile(r, q)) if len(r) else np.nan
    return row


def conservation_gate(edges: pd.DataFrame, ptx: pd.DataFrame, n_background: int, seed: int) -> pd.DataFrame:
    """Rows: every typology, all laundering pairs pooled, and the sampled background pairs."""
    pairs = head_to_tail_pairs(ptx)
    pairs = pairs[~pairs["back_and_forth"]]
    bg = sample_background_pairs(edges, n_background, seed)
    bga1, bga2 = bg["a1"].to_numpy(), bg["a2"].to_numpy()
    bok = (bga1 > 0) & (bga2 > 0)
    ref = np.sort(deviation(bga1[bok], bga2[bok]))
    rows = [_conservation_row(t, g["a1"], g["a2"], ref) for t, g in pairs.groupby("type", sort=True)]
    rows.append(_conservation_row(ALL_PAIRS_LABEL, pairs["a1"], pairs["a2"], ref))
    rows.append(_conservation_row(BACKGROUND_LABEL, bga1, bga2))
    out = pd.DataFrame(rows)
    out.attrs.update(bg.attrs)
    return out
