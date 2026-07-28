"""Does the bottleneck-flow measure escape the support domination that Proposition 4 predicts?

Proposition 4 says HUSM's utility is bounded by the utility TABLE's spread R, because
u = CMNI x MEAN of a few bucket values, and a mean of four numbers cannot vary much.
The paper measures the consequence: even with real money, support keeps 67% of the
score variance. u_flow replaces that mean with B(P), the best-funded occurrence's
weakest edge -- a raw amount, so it inherits the full heavy tail of the money.

This measures, on real AMLworld transactions with real amounts, how the variance of
each score splits between its support term and its utility term, and whether ranking
by u_flow picks different patterns than ranking by frequency alone.

Edge weights on the port-split graph:
  transaction edge a->b : the amount moved
  internal edge a::in-a::out : min(inflow(a), outflow(a)), the pass-through capacity
The internal-edge convention matters for the proof: every edge must carry a finite
positive weight, otherwise a pattern of only structural edges has an undefined
bottleneck and adding a transaction edge could RAISE it, breaking anti-monotonicity.
Capacity is also the right AML semantics -- a mule's throughput limits the chain.
"""
from __future__ import annotations

import math
import random
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments/scripts")
sys.path.insert(0, "experiments/src")
from e9_portsplit_recovery import DATA, load_patterns, load_transactions  # noqa: E402

NM = nx.algorithms.isomorphism.categorical_node_match("label", -1)
EM = nx.algorithms.isomorphism.categorical_edge_match("elabel", -1)


def build_port_split(edges: pd.DataFrame, n_buckets: int = 4):
    """Port-split graph carrying real money on every edge, plus the bucket table."""
    paid = edges.groupby("src")["amount"].sum()
    recv = edges.groupby("dst")["amount"].sum()
    signal = np.log1p(paid.add(recv, fill_value=0.0))
    qs = np.quantile(signal.to_numpy(), np.linspace(0, 1, n_buckets + 1)[1:-1])
    bucket = pd.Series(np.searchsorted(qs, signal.to_numpy()), index=signal.index)
    util = {b: float(signal[bucket == b].mean()) for b in sorted(bucket.unique())}

    G = nx.Graph()
    inflow = edges.groupby("dst")["amount"].sum()
    outflow = edges.groupby("src")["amount"].sum()

    def port(a, side):
        return f"{a}::{side}"

    for a, b, amt in edges[["src", "dst", "amount"]].itertuples(index=False):
        u, v = port(a, "o"), port(b, "i")
        G.add_node(u, label=2 * int(bucket[a]) + 1)
        G.add_node(v, label=2 * int(bucket[b]))
        w = float(amt)
        if G.has_edge(u, v):                       # parallel transactions: keep the largest
            G[u][v]["w"] = max(G[u][v]["w"], w)
        else:
            G.add_edge(u, v, elabel=0, w=w)
    for a in set(inflow.index) & set(outflow.index):
        u, v = port(a, "i"), port(a, "o")
        if u in G and v in G:
            G.add_edge(u, v, elabel=1, w=float(min(inflow[a], outflow[a])))
    return G, util, bucket


def occurrences(G, P, cap: int):
    gm = nx.algorithms.isomorphism.GraphMatcher(G, P, node_match=NM, edge_match=EM)
    for i, m in enumerate(gm.subgraph_monomorphisms_iter()):
        if i >= cap:
            return
        yield {p: g for g, p in m.items()}


def score(G, P, util, cap: int = 60_000):
    """Return (CMNI, B, ubar) -- support, bottleneck flow, and HUSM's mean vertex utility."""
    imgs = {v: set() for v in P}
    best = 0.0
    seen = False
    for phi in occurrences(G, P, cap):
        seen = True
        for v, x in phi.items():
            imgs[v].add(x)
        m = min(G[phi[a]][phi[b]]["w"] for a, b in P.edges())
        if m > best:
            best = m
    if not seen:
        return 0, 0.0, 0.0
    cm = min(len(s) for s in imgs.values())
    ubar = sum(util[P.nodes[v]["label"] // 2] for v in P) / P.number_of_nodes()
    return cm, best, ubar


def sample_patterns(G, rng, n_want: int, max_edges: int):
    """Connected sub-patterns lifted from the data, so every one of them occurs."""
    edges = list(G.edges())
    seen, out = set(), []
    for _ in range(n_want * 40):
        if len(out) >= n_want:
            break
        k = rng.randint(2, max_edges)
        chosen = {rng.choice(edges)}
        for _ in range(k - 1):
            front = {v for e in chosen for v in e}
            cand = [e for e in edges if e not in chosen and (e[0] in front or e[1] in front)]
            if not cand:
                break
            chosen.add(rng.choice(cand))
        P = nx.Graph()
        for a, b in chosen:
            P.add_edge(a, b, elabel=G[a][b]["elabel"])
        for v in P:
            P.nodes[v]["label"] = G.nodes[v]["label"]
        key = nx.weisfeiler_lehman_graph_hash(P, node_attr="label", edge_attr="elabel",
                                              iterations=3)
        if key in seen:
            continue
        seen.add(key)
        out.append(nx.convert_node_labels_to_integers(P))
    return out


def variance_share(sup, utl):
    """Fraction of Var(log score) carried by the utility term (the E8 decomposition)."""
    ls, lu = np.log(np.asarray(sup, float)), np.log(np.asarray(utl, float))
    tot = np.var(ls + lu)
    return float(np.var(lu) / tot) if tot > 0 else float("nan")


def main() -> None:
    rng = random.Random(20260724)
    t0 = time.time()
    edges, code = load_transactions(DATA / "HI-Small_Trans.csv")
    attempts = load_patterns(DATA / "HI-Small_Patterns.txt", code)
    truth = set()
    for a in attempts:
        truth |= a.accounts
    sub = edges[edges.src.isin(truth) & edges.dst.isin(truth)]     # attempts only: tractable
    print(f"  laundering-only subgraph: {len(sub):,} transactions "
          f"({time.time()-t0:.0f}s to load)")

    G, util, _ = build_port_split(sub)
    print(f"  port-split: {G.number_of_nodes():,} vertices, {G.number_of_edges():,} edges")
    amt = [d["w"] for _, _, d in G.edges(data=True) if d["elabel"] == 0]
    print(f"  amounts: min {min(amt):,.2f}  max {max(amt):,.2f}  ratio {max(amt)/min(amt):,.0f}x")
    print(f"  bucket utility table: { ({k: round(v,2) for k,v in util.items()}) }; "
          f"R = {max(util.values())/min(util.values()):.2f}")

    pats = sample_patterns(G, rng, n_want=300, max_edges=3)
    rows = []
    for P in pats:
        cm, B, ub = score(G, P, util)
        if cm > 0 and B > 0:
            rows.append({"cmni": cm, "B": B, "ubar": ub,
                         "u_flow": cm * B, "u_husm": cm * ub})
    d = pd.DataFrame(rows)
    print(f"  scored {len(d)} patterns ({time.time()-t0:.0f}s total)\n")

    f_share = variance_share(d.cmni, d.B)
    h_share = variance_share(d.cmni, d.ubar)
    print(f"  variance carried by the UTILITY term")
    print(f"    HUSM  (CMNI x mean bucket utility): {h_share:6.1%}   "
          f"=> support keeps {1-h_share:.1%}")
    print(f"    u_flow(CMNI x bottleneck flow)    : {f_share:6.1%}   "
          f"=> support keeps {1-f_share:.1%}")

    for k in (10, 20, 50):
        if len(d) < k:
            continue
        top_flow = set(d.u_flow.nlargest(k).index)
        top_husm = set(d.u_husm.nlargest(k).index)
        top_freq = set(d.cmni.nlargest(k).index)
        j = lambda a, b: len(a & b) / len(a | b)
        print(f"  top-{k:<3} Jaccard vs pure frequency: "
              f"HUSM {j(top_husm, top_freq):.3f}   u_flow {j(top_flow, top_freq):.3f}")
    print(f"\n  rank correlation with support: HUSM {d.u_husm.corr(d.cmni, method='spearman'):.3f}"
          f"   u_flow {d.u_flow.corr(d.cmni, method='spearman'):.3f}")


if __name__ == "__main__":
    main()
