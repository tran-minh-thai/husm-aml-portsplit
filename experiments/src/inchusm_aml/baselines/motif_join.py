"""E3  -  Starnini-style motif-join baseline.

Find AML motifs by sliding-window scan / relational join over the edge table, avoiding
subgraph isomorphism. For each account, scan every deltaT window and take the best
configuration; an account's activity need not all fit inside a single deltaT window.

Input: edge DataFrame [src, dst, amount, time, laundering] (time = minutes, int).
`laundering` (0/1, optional) is used only to EVALUATE precision  -  it takes no part in mining.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def _window_scan(times, partners, amounts, launds, delta_t):
    """Two pointers over one account: return the window configuration with the MOST
    distinct partners within deltaT: (k_best, amount_sum, laund_count) at that window."""
    best = (0, 0.0, 0)
    cnt: Counter = Counter()
    amt = 0.0
    laund = 0
    left = 0
    for right in range(len(times)):
        cnt[partners[right]] += 1
        amt += amounts[right]
        laund += launds[right]
        while times[right] - times[left] > delta_t:
            cnt[partners[left]] -= 1
            if cnt[partners[left]] == 0:
                del cnt[partners[left]]
            amt -= amounts[left]
            laund -= launds[left]
            left += 1
        if len(cnt) > best[0]:
            best = (len(cnt), amt, laund)
    return best


def windowed_fan(edges: pd.DataFrame, direction: str, delta_t: int) -> pd.DataFrame:
    """Fan-out (direction='out') / fan-in ('in'): for each center, the maximum number of
    distinct partners in any deltaT window. Return every center with its k_best
     -  the k_min threshold and k_max cap filter at the evaluation layer (enabling precision/recovery curves)."""
    center_col, partner_col = ("src", "dst") if direction == "out" else ("dst", "src")
    df = edges.sort_values([center_col, "time"], kind="stable")
    centers = df[center_col].to_numpy()
    partners = df[partner_col].to_numpy()
    times = df["time"].to_numpy()
    amounts = df["amount"].to_numpy()
    launds = (df["laundering"].to_numpy().astype(np.int64)
              if "laundering" in df else np.zeros(len(df), np.int64))

    rows = []
    bounds = np.flatnonzero(np.r_[True, centers[1:] != centers[:-1], True])
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], bounds[b + 1]
        k, amt, laund = _window_scan(times[lo:hi], partners[lo:hi], amounts[lo:hi],
                                     launds[lo:hi], delta_t)
        if k >= 2:
            rows.append({"center": centers[lo], "k": k, "window_amount": amt,
                         "laund_edges": laund})
    return pd.DataFrame(rows, columns=["center", "k", "window_amount", "laund_edges"])


def windowed_gather_scatter(edges: pd.DataFrame, delta_t: int) -> pd.DataFrame:
    """Smurf type-1 (SET of sources -> ONE middle -> SET of destinations): for each account, the deltaT
    window with (distinct-source count s, distinct-destination count t) maximizing min(s,t).
    Also record in/out amount in that window for post-filtering the flow ratio (Starnini)."""
    inc = edges[["dst", "src", "time", "amount"]].copy()
    inc.columns = ["acct", "partner", "time", "amount"]
    inc["dir"] = 0
    inc["laund"] = (edges["laundering"].to_numpy().astype(np.int64)
                    if "laundering" in edges else 0)
    out = edges[["src", "dst", "time", "amount"]].copy()
    out.columns = ["acct", "partner", "time", "amount"]
    out["dir"] = 1
    out["laund"] = inc["laund"].to_numpy()
    ev = pd.concat([inc, out], ignore_index=True).sort_values(["acct", "time"], kind="stable")

    accts = ev["acct"].to_numpy()
    partners = ev["partner"].to_numpy()
    times = ev["time"].to_numpy()
    amounts = ev["amount"].to_numpy()
    dirs = ev["dir"].to_numpy()
    launds = ev["laund"].to_numpy()

    rows = []
    bounds = np.flatnonzero(np.r_[True, accts[1:] != accts[:-1], True])
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], bounds[b + 1]
        best = (0, 0, 0, 0.0, 0.0, 0)  # (min(s,t), s, t, in_amt, out_amt, laund)
        cin: Counter = Counter()
        cout: Counter = Counter()
        in_amt = out_amt = 0.0
        laund = 0
        left = lo
        for right in range(lo, hi):
            if dirs[right] == 0:
                cin[partners[right]] += 1
                in_amt += amounts[right]
            else:
                cout[partners[right]] += 1
                out_amt += amounts[right]
            laund += launds[right]
            while times[right] - times[left] > delta_t:
                if dirs[left] == 0:
                    cin[partners[left]] -= 1
                    if cin[partners[left]] == 0:
                        del cin[partners[left]]
                    in_amt -= amounts[left]
                else:
                    cout[partners[left]] -= 1
                    if cout[partners[left]] == 0:
                        del cout[partners[left]]
                    out_amt -= amounts[left]
                laund -= launds[left]
                left += 1
            score = min(len(cin), len(cout))
            if score > best[0]:
                best = (score, len(cin), len(cout), in_amt, out_amt, laund)
        if best[0] >= 2:
            rows.append({"center": accts[lo], "s": best[1], "t": best[2],
                         "in_amount": best[3], "out_amount": best[4],
                         "laund_edges": best[5]})
    return pd.DataFrame(rows, columns=["center", "s", "t", "in_amount", "out_amount",
                                       "laund_edges"])


def scatter_gather_pairs(edges: pd.DataFrame, delta_t: int, hub_cap: int = 50,
                         mid_cap: int = 2500) -> pd.DataFrame:
    """Smurf type-2 (ONE source u -> m intermediaries -> ONE destination w): self-join the
    pair-level edge table on the intermediary key + an approximate temporal condition
    (the out phase does not end before the in phase; it starts within deltaT after the in phase).

    Pair-level approximation: use [t_min, t_max] of each pair (u,m)/(m,w) instead of individual
    transactions  -  fast, may slightly over-count (declared in the paper)."""
    pairs = (
        edges.groupby(["src", "dst"], sort=False)
        .agg(t_min=("time", "min"), t_max=("time", "max"),
             amount=("amount", "sum"),
             laund=("laundering", "max") if "laundering" in edges else ("amount", "size"))
        .reset_index()
    )
    out_deg = pairs.groupby("src", sort=False)["dst"].size()
    in_deg = pairs.groupby("dst", sort=False)["src"].size()
    hubs = set(out_deg[out_deg > hub_cap].index) | set(in_deg[in_deg > hub_cap].index)
    pairs = pairs[~pairs["src"].isin(hubs) & ~pairs["dst"].isin(hubs)]

    # cap join blow-up: drop intermediaries whose in_cnt*out_cnt is too large
    icnt = pairs.groupby("dst", sort=False)["src"].size().rename("icnt")
    ocnt = pairs.groupby("src", sort=False)["dst"].size().rename("ocnt")
    mid = pd.concat([icnt, ocnt], axis=1).dropna()
    ok_mid = set(mid[(mid["icnt"] * mid["ocnt"]) <= mid_cap].index)

    left = pairs[pairs["dst"].isin(ok_mid)].rename(
        columns={"src": "u", "dst": "m", "t_min": "in_min", "t_max": "in_max",
                 "amount": "in_amt", "laund": "l1"})
    right = pairs[pairs["src"].isin(ok_mid)].rename(
        columns={"src": "m", "dst": "w", "t_min": "out_min", "t_max": "out_max",
                 "amount": "out_amt", "laund": "l2"})
    j = left.merge(right, on="m")
    j = j[(j["u"] != j["w"])
          & (j["out_max"] >= j["in_min"])
          & (j["out_min"] <= j["in_max"] + delta_t)]
    if j.empty:
        return pd.DataFrame(columns=["u", "w", "m_count", "laund_any"])
    g = j.groupby(["u", "w"], sort=False).agg(
        m_count=("m", "nunique"),
        laund_any=("l1", "max"),
        laund_any2=("l2", "max"),
    ).reset_index()
    g["laund_any"] = (g["laund_any"].astype(int) | g["laund_any2"].astype(int))
    return g.drop(columns="laund_any2")


def temporal_cycles(edges: pd.DataFrame, delta_t: int, max_len: int = 5,
                    hub_cap: int = 20, per_start_cap: int = 20_000,
                    global_cap: int = 500_000) -> list[tuple]:
    """Simple cycles of 2..max_len vertices, non-decreasing time (pair-level approximation),
    total span <= deltaT. BOUNDED search (per_start_cap/global_cap + drop hubs)  - 
    a bounded baseline, not claimed to be exhaustive (declared in the paper)."""
    pairs = (
        edges.groupby(["src", "dst"], sort=False)
        .agg(t_min=("time", "min"), t_max=("time", "max"))
        .reset_index()
    )
    out_deg = pairs.groupby("src", sort=False)["dst"].size()
    hubs = set(out_deg[out_deg > hub_cap].index)
    pairs = pairs[~pairs["src"].isin(hubs) & ~pairs["dst"].isin(hubs)]

    adj: dict = {}
    for r in pairs.itertuples(index=False):
        adj.setdefault(r.src, []).append((r.dst, r.t_min, r.t_max))

    cycles: list[tuple] = []
    seen: set = set()
    budget = [0]

    def dfs(start, node, path, cursor, t_start, states):
        if states[0] <= 0 or budget[0] >= global_cap:
            return
        for nxt, tmin, tmax in adj.get(node, ()):
            states[0] -= 1
            if states[0] <= 0:
                return
            if tmax < cursor or tmin > t_start + delta_t:
                continue
            if nxt == start and len(path) >= 2:
                i = path.index(min(path))
                canon = tuple(path[i:] + path[:i])
                if canon not in seen:
                    seen.add(canon)
                    cycles.append(canon)
                    budget[0] += 1
            elif nxt not in path and len(path) < max_len:
                dfs(start, nxt, path + [nxt], max(cursor, tmin), t_start, states)

    for s in list(adj):
        if budget[0] >= global_cap:
            break
        states = [per_start_cap]
        for nxt, tmin, tmax in adj[s]:
            if nxt != s:
                dfs(s, nxt, [s, nxt], tmin, tmin, states)
    return cycles
