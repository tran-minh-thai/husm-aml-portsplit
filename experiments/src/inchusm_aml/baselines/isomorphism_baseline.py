"""Speed baseline for E3: full subgraph isomorphism (ISMAGS) vs database-join.

E3's technical contribution is using a relational JOIN to avoid subgraph isomorphism
(NP-complete). Starnini's selling point is "database-join is >40x faster than ISMAGS". Here
that comparison is reproduced on AMLworld data: both find the scatter-gather motif
(smurf type-2, template 1->{m1,m2}->1), measuring the wall-clock of:
  A. ISMAGS (networkx  -  the very tool Starnini used) counting template monomorphisms
  B. relational join (baselines.motif_join.scatter_gather_pairs)
over subgraphs of increasing size (after Starnini-style pre-filtering) -> a speed curve.

ISMAGS blows up with graph size so it runs on a small subset only (declared in the paper);
the join runs on the full graph.
"""

from __future__ import annotations

import time

import networkx as nx
import pandas as pd


def gather_scatter_template() -> nx.DiGraph:
    """Gather-scatter (smurf type-1): 2 sources -> 1 center -> 2 destinations. 5 vertices, 4 edges:
    a->c, b->c, c->d, c->e. Center c has in>=2 & out>=2  -  matching the center of
    motif_rich_subgraph, so VF2 counts REAL motifs. The same motif that
    windowed_gather_scatter (join) finds => an apples-to-apples comparison."""
    t = nx.DiGraph()
    t.add_edges_from([("a", "c"), ("b", "c"), ("c", "d"), ("c", "e")])
    return t


def build_digraph(edges: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(zip(edges["src"], edges["dst"]))
    return g


def count_subgraph_iso(edges: pd.DataFrame, timeout_s: float = 120.0) -> tuple[int, float, bool]:
    """Count monomorphisms of the smurf type-2 template with VF2 (networkx DiGraphMatcher)  - 
    full subgraph isomorphism, NP-complete, same family as Starnini's ISMAGS.
    Return (count mappings, seconds, completed). It cannot be interrupted mid-run so it runs on a
    small subset only; if it exceeds the timeout at a checkpoint it is marked not completed.
    """
    g = build_digraph(edges)
    template = gather_scatter_template()
    t0 = time.perf_counter()
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(g, template)
    count = 0
    completed = True
    for _ in matcher.subgraph_monomorphisms_iter():
        count += 1
        if count % 500 == 0 and time.perf_counter() - t0 > timeout_s:
            completed = False
            break
    return count, time.perf_counter() - t0, completed


def count_join(edges: pd.DataFrame, delta_t: int) -> tuple[int, float]:
    """Count gather-scatter centers via relational join (the same motif VF2 finds). Return (count, seconds)."""
    from .motif_join import windowed_gather_scatter

    t0 = time.perf_counter()
    centers = windowed_gather_scatter(edges, delta_t)
    return len(centers), time.perf_counter() - t0


def prefilter_starnini(edges: pd.DataFrame, max_degree: int = 50) -> pd.DataFrame:
    """Drop vertices with degree > max_degree (large enterprises)  -  Starnini-style pre-filter."""
    deg = pd.concat([edges["src"], edges["dst"]]).value_counts()
    hubs = set(deg[deg > max_degree].index)
    return edges[~edges["src"].isin(hubs) & ~edges["dst"].isin(hubs)]


def motif_rich_subgraph(edges: pd.DataFrame, n_centers: int,
                        min_in: int = 2, min_out: int = 2) -> pd.DataFrame:
    """Motif-RICH subgraph, DETERMINISTIC: take the n_centers gather-scatter centers (in>=min_in
    & out>=min_out) with the HIGHEST total degree (sorted descending, tie-break by id ->
    deterministic, hash-independent), then keep ALL edges incident to those centers. Each center is a
    complete gather-scatter template => VF2 counts REAL motifs. The edge count grows monotonically
    with n_centers.
    """
    din = edges.groupby("dst")["src"].nunique()
    dout = edges.groupby("src")["dst"].nunique()
    both = sorted(set(din[din >= min_in].index) & set(dout[dout >= min_out].index))
    deg = {c: int(din.get(c, 0)) + int(dout.get(c, 0)) for c in both}
    centers = set(sorted(both, key=lambda c: (-deg[c], c))[:n_centers])
    return edges[edges["src"].isin(centers) | edges["dst"].isin(centers)].reset_index(drop=True)
