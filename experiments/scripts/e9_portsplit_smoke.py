"""E9 smoke test: does port-splitting make direction visible to the undirected miner?

Four decisive checks on a toy directed graph holding one fan-in (u1,u2,u3 -> m1), one
fan-out (m2 -> w1,w2,w3) and one pass-through path (p1 -> q -> p2), with two monetary
buckets (0: u=1, 1: u=5; the centres m1, m2, q are bucket 1):

  A. UNSPLIT baseline: the undirected miner sees exactly ONE 1-edge pattern between the two
     buckets -- gather edges, scatter edges and path hops are indistinguishable. This is the
     unrepresentability the paper proves; here it is shown by running the actual miner.
  B. SPLIT: the same edges separate into (out0 - in1) and (out1 - in0) patterns, so
     gather and scatter stop being the same object.
  C. Stars: the 3-edge star centred on an IN-port (fan-in) and the star centred on an
     OUT-port (fan-out) appear as distinct patterns.
  D. Pass-through: with internal edges the miner can cross an account (a pattern containing
     edge label 1 exists); with internal_edges=False no pattern joins the two hops.

Run:  python scripts/e9_portsplit_smoke.py     (needs the jar; graph is tiny, runs in seconds)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inchusm_aml import miner_bridge                       # noqa: E402
from inchusm_aml.data.portsplit import port_split, split_utility  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
TUPLE = re.compile(r"\((-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+)\)")

BUCKET_UTILITY = {0: 1.0, 1: 5.0}


def toy_graph() -> tuple[pd.DataFrame, pd.Series]:
    edges = pd.DataFrame(
        [("u1", "m1"), ("u2", "m1"), ("u3", "m1"),      # fan-in
         ("m2", "w1"), ("m2", "w2"), ("m2", "w3"),      # fan-out
         ("p1", "q"), ("q", "p2")],                     # pass-through
        columns=["src", "dst"])
    labels = pd.Series({"u1": 0, "u2": 0, "u3": 0, "m1": 1,
                        "m2": 1, "w1": 0, "w2": 0, "w3": 0,
                        "p1": 0, "q": 1, "p2": 0})
    return edges, labels


def mine(tag: str, node_labels: pd.Series, edges: pd.DataFrame,
         utility: dict[int, float], max_edges: int = 3) -> pd.DataFrame:
    """MER=3, not the project's usual 1: HUSM only counts embeddings inside the MER-ball
    around the image of the DFS-code ROOT, and the minimal code of a star roots at a LEAF
    whenever a leaf label sorts below the centre label -- the opposite leaf then sits at
    distance 2, and a port-split pass-through chain reaches distance 3. This is a real cost
    of port-splitting the paper must state: the transform stretches pattern diameter as seen
    from the code root, so MER has to grow with maxEdges."""
    lg = RESULTS / f"e9_smoke_{tag}.lg"
    tsv = RESULTS / f"e9_smoke_{tag}.util.tsv"
    miner_bridge.export_lg(node_labels, edges, lg)
    miner_bridge.export_utilities(utility, tsv)
    hus, _ = miner_bridge.run_husm(lg, tsv, min_util=1.0, mer=3, max_edges=max_edges,
                                   out_jsonl=RESULTS / f"e9_smoke_{tag}.jsonl")
    return hus


def edge_types(frame: pd.DataFrame) -> set[tuple[int, int, int]]:
    """1-edge patterns as (min vertex label, edge label, max vertex label)."""
    out = set()
    for code in frame[frame.edges == 1].code:
        (_, _, li, lij, lj), = (tuple(map(int, m.groups())) for m in TUPLE.finditer(code))
        out.add((min(li, lj), lij, max(li, lj)))
    return out


def star_centres(frame: pd.DataFrame) -> set[int]:
    """Centre labels of 3-edge stars, read off the minimum DFS code.

    The minimal code of a star need not start at the centre: gSpan picks the lexicographically
    smallest first tuple, so when a leaf label is smaller than the centre label the code walks
    leaf -> centre first. Identify the star structurally instead: the vertex of degree 3.
    """
    out = set()
    for code in frame[frame.edges == 3].code:
        t = [tuple(map(int, m.groups())) for m in TUPLE.finditer(code)]
        deg: dict[int, int] = {}
        lab: dict[int, int] = {}
        for i, j, li, _lij, lj in t:
            deg[i] = deg.get(i, 0) + 1
            deg[j] = deg.get(j, 0) + 1
            lab[i], lab[j] = li, lj
        centres = [v for v, d in deg.items() if d == 3]
        if len(centres) == 1 and len(deg) == 4:
            out.add(lab[centres[0]])
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    edges, labels = toy_graph()
    checks: list[tuple[str, bool, str]] = []

    plain = mine("plain", labels, edges.assign(elabel=0), BUCKET_UTILITY)
    t_plain = edge_types(plain)
    cross = {t for t in t_plain if t[0] != t[2]}
    checks.append(("A unsplit: gather, scatter and hop are ONE single-edge pattern",
                   cross == {(0, 0, 1)}, f"cross-bucket edge types: {sorted(cross)}"))

    nl, ed = port_split(edges, labels)
    util2 = split_utility(BUCKET_UTILITY)
    split = mine("split", nl, ed, util2, max_edges=3)
    t_split = edge_types(split)
    # labels: in0=0, out0=1, in1=2, out1=3
    fan_in_edge, fan_out_edge = (1, 0, 2), (0, 0, 3)
    checks.append(("B split: fan-in and fan-out are TWO different edge types",
                   fan_in_edge in t_split and fan_out_edge in t_split,
                   f"transaction edge types: {sorted(t for t in t_split if t[1] == 0)}"))

    centres = star_centres(split)
    checks.append(("C split: 3-edge stars centred on an in-port and on an out-port are two patterns",
                   2 in centres and 3 in centres, f"star centres found: {sorted(centres)}"))

    def crosses_account(frame: pd.DataFrame) -> bool:
        """A pattern that really passes THROUGH an account: >=2 edges mixing a transaction
        edge (elabel 0) with an internal port edge (elabel 1)."""
        for _, row in frame[frame.edges >= 2].iterrows():
            els = {int(m.group(4)) for m in TUPLE.finditer(row.code)}
            if 0 in els and 1 in els:
                return True
        return False

    no_int = mine("split_noint", *port_split(edges, labels, internal_edges=False), util2)
    checks.append(("D pass-through: the internal edge lets a pattern pass THROUGH an account",
                   crosses_account(split) and not crosses_account(no_int),
                   f"pattern through an account with internal edges: {crosses_account(split)}; "
                   f"without: {crosses_account(no_int)}"))

    print("\n" + "=" * 74)
    ok_all = True
    for name, ok, detail in checks:
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}")
    print("=" * 74)
    print(f"{'4/4: port-splitting works with the current engine, no Java change' if ok_all else 'A CHECK FAILED'}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
