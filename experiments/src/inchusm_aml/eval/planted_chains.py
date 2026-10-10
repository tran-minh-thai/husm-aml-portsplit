"""Planted laundering chains, background chains, and the amount rules a chain occurrence may obey.

Planted chain (open, length m): m consecutive transactions of one laundering attempt,
e_1 .. e_m with target(e_i) = source(e_{i+1}), pairwise distinct accounts, and every intermediate
account a chain node of the attempt (exactly one incoming and one outgoing transaction inside it).
In a CYCLE attempt the origin (sender of the earliest transaction) is never an intermediate account
of an open chain: what arrives there is the money coming back, not money being forwarded.
Planted closed chain: a CYCLE attempt read from its origin, all of its accounts chain nodes.

Rules for a sequence of amounts a_1 .. a_m (consecutive transactions of a chain):
  pairwise(theta): (1 - theta) a_i <= a_{i+1} <= (1 + theta) a_i for every i  (the miner's rule)
  anchored(alpha): (1 - alpha) a_1 <= a_i <= (1 + alpha) a_1 for every i      (the cycle detectors' rule)
  none: always valid.
For a closed chain the sequence starts at the origin, so the pair at the origin is not constrained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pairwise(amounts, theta: float) -> bool:
    a = list(amounts)
    return all((1 - theta) * x <= y <= (1 + theta) * x for x, y in zip(a, a[1:]))


def anchored(amounts, alpha: float) -> bool:
    a = list(amounts)
    return all((1 - alpha) * a[0] <= y <= (1 + alpha) * a[0] for y in a[1:])


def planted_chains(ptx: pd.DataFrame, max_length: int) -> pd.DataFrame:
    """One row per planted chain: attempt, type, kind (open/closed), length, accounts, amounts."""
    rows = []
    p = ptx[ptx["src"] != ptx["dst"]]
    for attempt, g in p.groupby("attempt", sort=True):
        typ = g["type"].iat[0]
        g = g.sort_values("time", kind="stable")
        src, dst, amt = g["src"].to_numpy(), g["dst"].to_numpy(), g["amount"].to_numpy(dtype=float)
        k_in = pd.Series(dst).value_counts()
        k_out = pd.Series(src).value_counts()
        chain_node = {v for v in set(src) | set(dst) if k_in.get(v, 0) == 1 and k_out.get(v, 0) == 1}
        origin = src[0] if typ == "CYCLE" else None
        out_of = {}
        for i, s in enumerate(src):
            out_of.setdefault(s, []).append(i)
        for start in range(len(src)):
            accounts, amounts = [src[start], dst[start]], [amt[start]]
            i = start
            while True:
                v = dst[i]
                if v not in chain_node or v == origin:
                    break
                (i,) = out_of[v]
                nxt = dst[i]
                if nxt == accounts[0]:
                    if typ == "CYCLE" and accounts[0] == origin and len(amounts) + 1 <= max_length:
                        rows.append((attempt, typ, "closed", len(amounts) + 1, tuple(accounts), tuple(amounts + [amt[i]])))
                    break
                if nxt in accounts or len(amounts) + 1 > max_length:
                    break
                accounts.append(nxt)
                amounts.append(amt[i])
                rows.append((attempt, typ, "open", len(amounts), tuple(accounts), tuple(amounts)))
    return pd.DataFrame(rows, columns=["attempt", "type", "kind", "length", "accounts", "amounts"])


def background_chains(edges: pd.DataFrame, length: int, n: int, seed: int):
    """n random open chains of the given length made of background transactions, and the number of
    walks discarded (dead end or repeated account). Walk: a uniform background transaction, then at
    each step a uniform background transaction out of the current account."""
    e = edges[(edges["laundering"] == 0) & (edges["src"] != edges["dst"])]
    src, dst, amt = e["src"].to_numpy(), e["dst"].to_numpy(), e["amount"].to_numpy(dtype=float)
    order = np.argsort(src, kind="stable")
    accounts = np.unique(np.concatenate([src, dst]))
    pos = np.searchsorted(accounts, src[order])
    start = np.zeros(len(accounts) + 1, dtype=np.int64)
    np.add.at(start, pos + 1, 1)
    start = np.cumsum(start)
    rng = np.random.default_rng(seed)
    out, discarded = [], 0
    while len(out) < n:
        i = int(rng.integers(len(src)))
        acc, amounts = [src[i], dst[i]], [amt[i]]
        ok = True
        for _ in range(length - 1):
            k = int(np.searchsorted(accounts, acc[-1]))
            lo, hi = start[k], start[k + 1]
            if hi == lo:
                ok = False
                break
            j = order[int(rng.integers(lo, hi))]
            if dst[j] in acc:
                ok = False
                break
            acc.append(dst[j])
            amounts.append(amt[j])
        if ok:
            out.append(tuple(amounts))
        else:
            discarded += 1
    return out, discarded
