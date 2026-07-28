"""Probe: is CMNI x bottleneck-flow downward closed, and does it escape the R-ceiling?

The claim under test (the bottleneck measure). With w(e) the money on a transaction edge,

    B(P)      = max over occurrences phi of min over e in E(P) of w(phi(e))
    u_flow(P) = CMNI(P) * B(P)

is ANTI-MONOTONE, i.e. u_flow(P') <= u_flow(P) whenever P' extends P. If that holds,
mining needs no upper bound at all: the measure prunes itself, unlike HUSM's
CMNI x mean(vertex utility), whose non-closure is exactly why CEU/EU/IVU exist.

This file tries to REFUTE that by brute force on random graphs, because the claim is
my own derivation and a derivation is not evidence. A single counterexample kills it.

Occurrences are subgraph monomorphisms (injective, non-induced), matching the MNI
convention the miner uses.
"""
from __future__ import annotations

import itertools
import random

import networkx as nx

NM = nx.algorithms.isomorphism.categorical_node_match("label", -1)


def occurrences(G: nx.Graph, P: nx.Graph):
    """All injective label-preserving mappings V(P) -> V(G) that carry E(P) into E(G)."""
    gm = nx.algorithms.isomorphism.GraphMatcher(G, P, node_match=NM)
    # GraphMatcher maps G-nodes -> P-nodes; invert to phi: P -> G.
    for m in gm.subgraph_monomorphisms_iter():
        yield {p: g for g, p in m.items()}


def cmni(G: nx.Graph, P: nx.Graph) -> int:
    """min over pattern vertices of the number of distinct images that vertex takes."""
    imgs = {v: set() for v in P}
    for phi in occurrences(G, P):
        for v, x in phi.items():
            imgs[v].add(x)
    return min((len(s) for s in imgs.values()), default=0)


def bottleneck(G: nx.Graph, P: nx.Graph) -> float:
    """B(P): the best-funded occurrence's weakest edge. 0 when P does not occur."""
    best = 0.0
    for phi in occurrences(G, P):
        m = min(G[phi[a]][phi[b]]["w"] for a, b in P.edges())
        if m > best:
            best = m
    return best


def u_flow(G: nx.Graph, P: nx.Graph) -> float:
    return cmni(G, P) * bottleneck(G, P)


def u_husm(G: nx.Graph, P: nx.Graph, ext: dict[int, float]) -> float:
    """HUSM Def. 8 for contrast: support x MEAN of vertex-label external utility."""
    n = P.number_of_nodes()
    if n == 0:
        return 0.0
    mean_u = sum(ext[P.nodes[v]["label"]] for v in P) / n
    return cmni(G, P) * mean_u


def random_graph(rng: random.Random, n: int, m: int, n_labels: int) -> nx.Graph:
    G = nx.gnm_random_graph(n, m, seed=rng.randint(0, 10**9))
    for v in G:
        G.nodes[v]["label"] = rng.randrange(n_labels)
    for a, b in G.edges():
        # heavy-tailed amounts, like real transaction values
        G[a][b]["w"] = round(10 ** rng.uniform(0, 4), 2)
    return G


def connected_patterns(G: nx.Graph, max_edges: int, rng: random.Random, cap: int):
    """Connected edge-induced sub-patterns of G, relabelled 0..k-1, as pattern graphs."""
    edges = list(G.edges())
    seen, out = set(), []
    for _ in range(cap * 6):
        if len(out) >= cap:
            break
        k = rng.randint(1, max_edges)
        start = rng.choice(edges)
        chosen = {start}
        for _ in range(k - 1):
            frontier = {v for e in chosen for v in e}
            cand = [e for e in edges if e not in chosen and (e[0] in frontier or e[1] in frontier)]
            if not cand:
                break
            chosen.add(rng.choice(cand))
        P = nx.Graph()
        for a, b in chosen:
            P.add_edge(a, b)
        for v in P:
            P.nodes[v]["label"] = G.nodes[v]["label"]
        key = nx.weisfeiler_lehman_graph_hash(P, node_attr="label", iterations=3)
        if key in seen:
            continue
        seen.add(key)
        out.append(nx.convert_node_labels_to_integers(P))
    return out


def one_edge_extensions(G: nx.Graph, P: nx.Graph, rng: random.Random, cap: int = 12):
    """Patterns P' with exactly one more edge that still occur somewhere in G."""
    out = []
    labels = {G.nodes[v]["label"] for v in G}
    for _ in range(cap):
        Q = P.copy()
        v = rng.choice(list(Q.nodes()))
        if rng.random() < 0.5 and Q.number_of_nodes() > 1:
            u = rng.choice([x for x in Q.nodes() if x != v])   # backward edge
            if Q.has_edge(u, v):
                continue
            Q.add_edge(u, v)
        else:
            new = max(Q.nodes()) + 1                            # forward edge
            Q.add_node(new, label=rng.choice(sorted(labels)))
            Q.add_edge(v, new)
        out.append(Q)
    return out


def main() -> None:
    rng = random.Random(20260724)
    ext = {lab: round(10 ** rng.uniform(0, 1), 3) for lab in range(4)}  # HUSM-style table

    checked = flow_viol = husm_viol = 0
    worst_flow = worst_husm = 0.0
    for trial in range(40):
        G = random_graph(rng, n=rng.randint(8, 14), m=rng.randint(12, 26), n_labels=4)
        for P in connected_patterns(G, max_edges=3, rng=rng, cap=10):
            uf, uh = u_flow(G, P), u_husm(G, P, ext)
            for Q in one_edge_extensions(G, P, rng):
                qf, qh = u_flow(G, Q), u_husm(G, Q, ext)
                checked += 1
                if qf > uf + 1e-9:                      # would refute the claim
                    flow_viol += 1
                    worst_flow = max(worst_flow, qf / max(uf, 1e-9))
                if qh > uh + 1e-9:                      # HUSM: expected to happen
                    husm_viol += 1
                    worst_husm = max(worst_husm, qh / max(uh, 1e-9))

    print(f"  parent/child pairs tested          : {checked:,}")
    print(f"  u_flow  violations (claim refuted?): {flow_viol}"
          f"{'' if not flow_viol else f'  worst ratio {worst_flow:.3f}'}")
    print(f"  u_HUSM  violations (control)       : {husm_viol}"
          f"  worst ratio {worst_husm:.3f}")
    print()
    if flow_viol == 0 and husm_viol > 0:
        print("  => u_flow held on every pair; the HUSM control DID break, so the test")
        print("     is live rather than vacuously passing.")
    elif flow_viol:
        print("  => CLAIM REFUTED. the bottleneck measure dies here; do not build on it.")
    else:
        print("  => INCONCLUSIVE: the control never broke, so the probe proves nothing.")


if __name__ == "__main__":
    main()
