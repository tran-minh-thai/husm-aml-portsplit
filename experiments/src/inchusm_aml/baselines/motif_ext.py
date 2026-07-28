"""E3+ (journal): miners for the three typologies beyond the conference / first-paper scope.

The first paper covered five single-hub or short-path motifs (fan-out, fan-in,
gather-scatter, scatter-gather, cycle). This module adds the remaining three AMLworld
typologies so recovery is measured over all eight:

- RANDOM   : a single forward-time path of up to max_len accounts (not closing, unlike a cycle);
- STACK    : layered chains through intermediary accounts that receive then forward;
- BIPARTITE: a dense source-group to destination-group core.

Each miner returns the account SETS of the structures it finds (frozensets), so recovery
matching stays set-based and encoding-independent, consistent with the earlier metric: an
attempt is recovered when a mined structure shares enough accounts with it.

The three are structurally harder than the five single-hub motifs; the thresholds
(hub_cap, budgets) are the tuning knobs and are stated per run rather than hard-coded.
"""

from __future__ import annotations

import pandas as pd


def _forward_adjacency(edges: pd.DataFrame, hub_cap: int):
    """Pair-level forward adjacency: account -> list of (next, t_min, t_max), hubs dropped."""
    pairs = (
        edges.groupby(["src", "dst"], sort=False)
        .agg(t_min=("time", "min"), t_max=("time", "max"))
        .reset_index()
    )
    out_deg = pairs.groupby("src", sort=False)["dst"].size()
    in_deg = pairs.groupby("dst", sort=False)["src"].size()
    hubs = set(out_deg[out_deg > hub_cap].index) | set(in_deg[in_deg > hub_cap].index)
    pairs = pairs[~pairs["src"].isin(hubs) & ~pairs["dst"].isin(hubs)]
    adj: dict = {}
    for r in pairs.itertuples(index=False):
        adj.setdefault(r.src, []).append((r.dst, r.t_min, r.t_max))
    return adj, pairs


def random_paths(edges: pd.DataFrame, delta_t: int, max_len: int = 5,
                 hub_cap: int = 20, per_start_cap: int = 20_000,
                 global_cap: int = 500_000) -> list[frozenset]:
    """Forward-time simple paths of 3..max_len distinct accounts that do NOT return to the
    start (a random layering chain). Bounded depth-first search with the same budget guards
    as the cycle miner; each path is reported as its account set."""
    adj, _ = _forward_adjacency(edges, hub_cap)
    out: list[frozenset] = []
    seen: set = set()
    budget = [0]

    def dfs(node, path, cursor, t_start, states):
        if states[0] <= 0 or budget[0] >= global_cap:
            return
        if len(path) >= 3:
            key = frozenset(path)
            if key not in seen:
                seen.add(key)
                out.append(key)
                budget[0] += 1
        if len(path) >= max_len:
            return
        for nxt, tmin, tmax in adj.get(node, ()):
            states[0] -= 1
            if states[0] <= 0:
                return
            if tmax < cursor or tmin > t_start + delta_t or nxt in path:
                continue
            dfs(nxt, path + [nxt], max(cursor, tmin), t_start, states)

    for s in list(adj):
        if budget[0] >= global_cap:
            break
        states = [per_start_cap]
        # the window starts at EACH first edge (as in temporal_cycles): locking t_start to the
        # earliest edge out of s would wrongly discard chains that begin with a later edge
        for nxt, tmin, _ in adj[s]:
            if nxt != s:
                dfs(nxt, [s, nxt], tmin, tmin, states)
    return out


def stack_chains(edges: pd.DataFrame, delta_t: int, min_hops: int = 2,
                 hub_cap: int = 50, per_start_cap: int = 20_000,
                 global_cap: int = 500_000) -> list[frozenset]:
    """Layered chains: forward-time paths passing through at least one intermediary that both
    receives and forwards within the window (a stacked layering). Reported as account sets of
    chains with at least min_hops+1 accounts; the intermediaries are the identifying layer."""
    adj, pairs = _forward_adjacency(edges, hub_cap)
    mids = set(pairs["dst"]) & set(pairs["src"])   # accounts that both receive and forward
    out: list[frozenset] = []
    seen: set = set()
    budget = [0]

    def dfs(node, path, cursor, t_start, states):
        if states[0] <= 0 or budget[0] >= global_cap:
            return
        # a chain qualifies once it has enough hops and at least one intermediary
        if len(path) >= min_hops + 1 and any(a in mids for a in path[1:-1]):
            key = frozenset(path)
            if key not in seen:
                seen.add(key)
                out.append(key)
                budget[0] += 1
        for nxt, tmin, tmax in adj.get(node, ()):
            states[0] -= 1
            if states[0] <= 0:
                return
            if tmax < cursor or tmin > t_start + delta_t or nxt in path:
                continue
            dfs(nxt, path + [nxt], max(cursor, tmin), t_start, states)

    for s in list(adj):
        if budget[0] >= global_cap:
            break
        states = [per_start_cap]
        # the window starts at EACH first edge (as in temporal_cycles): locking t_start to the
        # earliest edge out of s would wrongly discard chains that begin with a later edge
        for nxt, tmin, _ in adj[s]:
            if nxt != s:
                dfs(nxt, [s, nxt], tmin, tmin, states)
    return out


def bipartite_cores(edges: pd.DataFrame, delta_t: int, min_side: int = 2,
                    hub_cap: int = 200) -> list[frozenset]:
    """Dense source-group to destination-group cores: sources with out-degree >= min_side and
    destinations with in-degree >= min_side, restricted to their induced bipartite subgraph;
    each connected component with both sides >= min_side is reported as its account set.

    Approximate (component-level, not exact bicliques); the min_side / hub_cap knobs trade
    recall against precision and are stated per run.
    """
    pairs = (
        edges.groupby(["src", "dst"], sort=False)
        .agg(t_min=("time", "min"), t_max=("time", "max"))
        .reset_index()
    )
    pairs = pairs[pairs["t_max"] - pairs["t_min"] <= delta_t]
    out_deg = pairs.groupby("src", sort=False)["dst"].size()
    in_deg = pairs.groupby("dst", sort=False)["src"].size()
    keep_src = set(out_deg[(out_deg >= min_side) & (out_deg <= hub_cap)].index)
    keep_dst = set(in_deg[(in_deg >= min_side) & (in_deg <= hub_cap)].index)
    core = pairs[pairs["src"].isin(keep_src) & pairs["dst"].isin(keep_dst)]
    if core.empty:
        return []

    # connected components over the bipartite graph (sources and destinations as nodes)
    adj: dict = {}
    for r in core.itertuples(index=False):
        adj.setdefault(("s", r.src), set()).add(("d", r.dst))
        adj.setdefault(("d", r.dst), set()).add(("s", r.src))
    seen: set = set()
    out: list[frozenset] = []
    for node in adj:
        if node in seen:
            continue
        stack, comp = [node], []
        seen.add(node)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        srcs = {a for side, a in comp if side == "s"}
        dsts = {a for side, a in comp if side == "d"}
        if len(srcs) >= min_side and len(dsts) >= min_side:
            out.append(frozenset(srcs | dsts))
    return out
