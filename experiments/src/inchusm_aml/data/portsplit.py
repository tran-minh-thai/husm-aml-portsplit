"""Port-splitting: encode money-flow DIRECTION for an undirected subgraph miner.

The miner (HUSM/gSpan canonical form) works on undirected graphs, and all eight AMLworld
typologies are defined by flow direction, so on the plain account graph fan-in and fan-out
collapse into the same star and the typologies are unrepresentable -- not hard, unrepresentable.
This transform lifts direction into LABELS, which the miner does understand:

    account a  ->  two vertices a_in, a_out
    tx a->b    ->  undirected edge {a_out, b_in}         (edge label 0)
    internal   ->  undirected edge {a_in, a_out}         (edge label 1),
                   only when a has both incoming and outgoing transactions

    label(a_in)  = 2 * bucket(a)        (even = in-port)
    label(a_out) = 2 * bucket(a) + 1    (odd  = out-port)

Under the split, gather and scatter become DIFFERENT stars (their centres carry different
port labels), a pass-through account is a path crossing its internal edge, and a directed
cycle differs from an undirected one. The cost, stated in the paper: vertex count at most
doubles, and a k-hop path costs k-1 extra internal edges, so maxEdges must grow accordingly.

Utility semantics: both ports of an account carry the account's bucket utility, so ubar(P)
remains an average of bucket utilities and the ceiling proposition (R = e_max/e_min of the
BUCKET table) is unchanged by the transform.
"""

from __future__ import annotations

import pandas as pd


def split_labels(bucket_label: pd.Series) -> pd.Series:
    """Port-node labels for every account: index 'a::i'/'a::o', value 2b / 2b+1.

    Every account gets both ports here; export keeps only ports that carry an edge.
    """
    b = bucket_label.astype(int)
    lab_in = pd.Series((2 * b).to_numpy(), index=[f"{a}::i" for a in b.index])
    lab_out = pd.Series((2 * b + 1).to_numpy(), index=[f"{a}::o" for a in b.index])
    return pd.concat([lab_in, lab_out])


def split_utility(bucket_utility: dict[int, float]) -> dict[int, float]:
    """Both ports of bucket b carry u(b): {2b: u, 2b+1: u}. R over labels equals R over buckets."""
    out: dict[int, float] = {}
    for b, u in bucket_utility.items():
        out[2 * int(b)] = float(u)
        out[2 * int(b) + 1] = float(u)
    return out


def port_split(edges: pd.DataFrame, bucket_label: pd.Series,
               internal_edges: bool = True, weighted: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    """Transform a directed account graph into the port-split undirected graph.

    edges: columns [src, dst] (+ 'amount' when weighted), one row per directed edge.
    bucket_label: index = account id, value = int bucket.
    Returns (node_labels, edges) ready for miner_bridge.export_lg: transaction edges carry
    elabel 0, internal edges elabel 1. Only ports incident to some edge are kept, so an
    account with only outgoing money contributes a single vertex, exactly like before.

    weighted=True adds a 'weight' column carrying the money on each edge, for the FLOW
    measure: a transaction edge carries its amount (parallel transactions keep the largest,
    matching the miner's single-edge model), and an internal edge a_in-a_out carries the
    pass-through capacity min(inflow(a), outflow(a)). Every edge is thus finitely weighted,
    which the anti-monotonicity of u_flow requires.
    """
    e = edges[["src", "dst"]].astype(str)
    tx = pd.DataFrame({"src": e["src"] + "::o", "dst": e["dst"] + "::i", "elabel": 0})
    if weighted:
        amt = pd.to_numeric(edges["amount"], errors="coerce").fillna(0.0).to_numpy()
        tx["weight"] = amt
        # parallel transactions collapse to one undirected edge; keep the largest amount
        tx = (tx.sort_values("weight", ascending=False)
                .drop_duplicates(subset=["src", "dst"], keep="first")
                .reset_index(drop=True))
    parts = [tx]
    if internal_edges:
        both = sorted(set(e["src"]) & set(e["dst"]))
        if both:
            internal = pd.DataFrame({
                "src": [f"{a}::i" for a in both],
                "dst": [f"{a}::o" for a in both],
                "elabel": 1,
            })
            if weighted:
                inflow = pd.to_numeric(edges["amount"], errors="coerce").fillna(0.0).groupby(e["dst"]).sum()
                outflow = pd.to_numeric(edges["amount"], errors="coerce").fillna(0.0).groupby(e["src"]).sum()
                internal["weight"] = [min(float(inflow.get(a, 0.0)), float(outflow.get(a, 0.0)))
                                      for a in both]
            parts.append(internal)
    all_edges = pd.concat(parts, ignore_index=True)
    labels = split_labels(bucket_label.astype(int))
    used = pd.Index(pd.unique(pd.concat([all_edges["src"], all_edges["dst"]])))
    return labels.loc[used], all_edges
