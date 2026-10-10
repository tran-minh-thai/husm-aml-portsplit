"""Typology recovery on AMLworld under port-splitting.

The eight AMLworld laundering typologies are defined by the direction of money flow, but
the miner works on undirected graphs, where gathering and scattering are the same star.
Port-splitting removes that obstacle: every account becomes an in-port and an out-port,
a transaction becomes an edge from the payer's out-port to the payee's in-port, and the
direction now lives in the vertex labels the miner already reads. See data/portsplit.py
for the transform, and e9_portsplit_smoke.py for a four-case check that it works.

The degree-cap sweep then measures one cost in isolation. Before port-splitting, a missed
fan had two possible causes at once: the pattern was not representable at all, and the
degree cap needed for tractability had dropped the fan centre. Port-splitting removes the
first, so the recovery-against-cap curve measures the second alone.

Utility is money: a per-account signal (total amount in and out, in US Dollar), bucketed
by quantile. Both ports of an account carry that bucket, so the mean utility of a pattern is
unchanged by the transform and the utility-table ceiling still applies unchanged.

  --utility-scale raw   uses amounts directly.
  --utility-scale log   buckets log(1 + amount). It is monotone, so the labels and the
                        graph are identical, but it narrows the table and with it the
                        share of pattern pairs whose order utility can change.

Every mine reports the table spread R, the share of pattern pairs the ceiling locks, the
share utility actually reverses relative to CMNI, and the variance decomposition of the
score with its covariance (see inchusm_aml.eval.score_decomposition). It stops if a locked
pair is reversed, and warns if utility reverses no pair at all.

Scale. The full port-split HI-Small graph holds about 1.03M vertices and 5.48M edges and
needs 64 GB of RAM. --mode truth-neighborhood mines the subgraph of every edge touching a
ground-truth account, so each labelled attempt is fully present, plus its one-hop
background: about 16k vertices, which runs on a laptop.

Thresholds. --min-util is absolute while pattern utility grows with the graph, so it must
be scaled: 200 on the truth-neighbourhood subgraph corresponds to 12000 on the full graph,
which is 57 times larger. The driver warns when the threshold is out of scale. Do not set
--min-util 1: every pattern clears it and the search space explodes. A --max-edges of 4 or
more also explodes, not finishing within an hour even on 64 GB; such a run is timed out
and recorded as timed_out=1 rather than left hanging.

Run:  python scripts/e9_portsplit_recovery.py --dataset HI-Small --mode truth-neighborhood \
          --hub-caps 0 200 50 --mer 2 --max-edges 3 --min-util 200 --utility-scale raw
      run_e9.bat runs the full set on Windows.
Output: results/e9_portsplit_recovery_{dataset}_{mode}[_raw].csv
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inchusm_aml import miner_bridge                                  # noqa: E402
from inchusm_aml.data.amlworld import load_patterns, load_transactions  # noqa: E402
from inchusm_aml.data.portsplit import port_split, split_utility     # noqa: E402
from inchusm_aml.utility.monetary import assign_bucket               # noqa: E402
from inchusm_aml.eval.score_decomposition import order_statistics, variance_decomposition  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parents[1] / "results"
# Everything this script writes goes to OUT: the record directory when run through
# experiments/provenance/run_and_record.py, so a new run never overwrites a committed
# artifact; RESULTS otherwise.
OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else RESULTS
DATA = ROOT / "datasets" / "amlworld"
JAR = ROOT / "inchusm-miner" / "target" / "inchusm-miner-0.1.0-SNAPSHOT.jar"


def account_signal(edges: pd.DataFrame, scale: str = "log") -> pd.Series:
    """Per-account monetary signal: the total amount flowing in and out of the account.

    scale='log' applies log1p, scale='raw' does not. The choice does NOT move any account
    between buckets -- log1p is monotone and the buckets are quantiles -- so labels, graph
    and every structural result are identical either way. What it changes is the SPREAD of
    the utility table, which bounds how many pattern pairs utility can reorder at all (the
    support-dominance ceiling): the log scale narrows it to a ratio of a few, the raw scale
    keeps the heavy tail of the amounts.
    """
    paid = edges.groupby("src")["amount"].sum()
    recv = edges.groupby("dst")["amount"].sum()
    total = paid.add(recv, fill_value=0.0)
    vals = total.to_numpy(dtype=np.float64)
    return pd.Series(np.log1p(vals) if scale == "log" else vals, index=total.index)


def utility_diagnostics(hus: pd.DataFrame) -> dict[str, float]:
    """How much of the score is actually the UTILITY dimension, and how much is just support.

    A miner called 'high-utility' can silently be doing frequent mining: u = CMNI * ubar, and
    when the utility table is narrow, ubar is nearly constant so u is a monotone function of
    CMNI. The support-dominance ceiling makes that precise -- pattern pairs further apart in support than
    R = max(ubar)/min(ubar) are order-locked whatever the utilities are. This is reported on
    EVERY mine so the claim can never again rest on an unmeasured assumption.
    """
    d = hus[(hus["cmni"] > 0) & (hus["utility"] > 0)].copy()
    if len(d) < 3:
        return {"R": float("nan"), "utility_share": float("nan"), "locked_pairs": float("nan"),
                "reversed_pairs": float("nan"), "order_violations": 0, "n_patterns": len(d)}
    d["ubar"] = d["utility"] / d["cmni"]
    R = float(d["ubar"].max() / d["ubar"].min())
    dec = variance_decomposition(d["cmni"], d["ubar"])
    order = order_statistics(d["cmni"], d["ubar"], R)
    # utility_share is the Shapley share (with the covariance); locked/reversed are over the
    # pairs with different CMNI. Earlier versions wrote Var(log ubar)/Var(log u) here and the
    # locked share over ALL pairs, including equal-CMNI pairs.
    return {"R": R, "utility_share": dec["share_weight"], "cov_log": dec["cov_log"],
            "ratio_utility_no_cov": dec["ratio_weight_no_cov"],
            "locked_pairs": order["locked_frac"], "reversed_pairs": order["reversed_frac"],
            "comparable_pairs": order["comparable_pairs"],
            "order_violations": order["violations"], "n_patterns": len(d)}


def fit_buckets(signal: pd.Series, n_buckets: int) -> tuple[list[float], dict[int, float]]:
    """Quantile bucket edges over ALL accounts, plus each bucket's mean-signal utility.

    AMLworld has no train/test time split for this graph-level task, so buckets are fit over
    the whole account population; the fit is frozen before mining, exactly as elsewhere.
    """
    qs = np.linspace(0.0, 1.0, n_buckets + 1)[1:-1]
    edges_q = list(np.quantile(signal.to_numpy(), qs))
    lab = assign_bucket(signal, edges_q)
    util = {int(b): float(signal[lab == b].mean()) for b in sorted(lab.unique())}
    return edges_q, util


def truth_neighborhood(edges: pd.DataFrame, attempts) -> pd.DataFrame:
    """Every edge incident to a ground-truth account: each attempt is fully present, plus a
    one-hop background of the accounts those attempts transact with."""
    truth = set()
    for a in attempts:
        truth |= a.accounts
    keep = edges["src"].isin(truth) | edges["dst"].isin(truth)
    return edges[keep].reset_index(drop=True)


def apply_hub_cap(edges: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Drop accounts whose total degree exceeds cap (0 = no cap). Mirrors the mining pipeline's
    prefilter; the point of the sweep is to watch recovery fall as the cap removes fan centres."""
    if cap <= 0:
        return edges
    deg = pd.concat([edges["src"], edges["dst"]]).value_counts()
    hot = set(deg[deg > cap].index)
    keep = ~(edges["src"].isin(hot) | edges["dst"].isin(hot))
    return edges[keep].reset_index(drop=True)


def mine_and_match(edges: pd.DataFrame, labels: pd.Series, util: dict[int, float],
                   tag: str, mer: int, max_edges: int, min_util: float, heap: str = "8g",
                   timeout_s: int = 900) -> tuple[list[str] | None, dict]:
    """Port-split the graph and mine HUS; return the mined DFS codes, or None if the mine did not
    finish within timeout_s. Raising max_edges from 3 to 4 explodes the pattern space (a 16k-vertex
    graph did not finish in an hour at min_util=200), so the sweep must survive a config that never
    completes: None flows through to a 'timed out' recovery row rather than crashing the whole run.
    That a size bound large enough to represent the >3-edge typologies makes mining intractable at
    this threshold is itself a reportable cost, not a failure to hide."""
    node_labels, split_edges = port_split(edges, labels)
    split_util = split_utility(util)
    lg = OUT / f"e9_{tag}.lg"
    tsv = OUT / f"e9_{tag}.util.tsv"
    miner_bridge.export_lg(node_labels, split_edges, lg)
    miner_bridge.export_utilities(split_util, tsv)

    # min_util is ABSOLUTE while u = CMNI * mean-utility and CMNI grows with the graph, so a
    # threshold calibrated on a small graph silently collapses on a big one and the lattice
    # explodes. CALIB_V is the vertex count min-util=200 was calibrated on; report what this
    # threshold is worth here so a mis-scaled run is visible before it hangs, not after.
    CALIB_V = 16_117
    eff = min_util * CALIB_V / max(1, len(node_labels))
    if eff < 50:
        print(f"  [{tag}] WARNING: min-util={min_util:g} on {len(node_labels):,} vertices is worth "
              f"~{eff:.1f} on the {CALIB_V:,}-vertex calibration graph, where <50 was already "
              f"measured to explode. Scale it by ~{len(node_labels)/CALIB_V:.0f}x.", flush=True)
    print(f"  [{tag}] split graph: {len(node_labels):,} vertices, {len(split_edges):,} edges; "
          f"mining (mer={mer}, max-edges={max_edges}, min-util={min_util})...", flush=True)
    t0 = time.time()
    try:
        hus, _ = miner_bridge.run_husm(lg, tsv, min_util=min_util, mer=mer, max_edges=max_edges,
                                       out_jsonl=OUT / f"e9_{tag}.hus.jsonl", timeout_s=timeout_s,
                                       heap=heap)
    except subprocess.TimeoutExpired:
        print(f"  [{tag}] TIMED OUT after {timeout_s}s -- pattern space too large at this "
              f"(max_edges, min_util); recorded as timed out.")
        return None, {"R": float("nan"), "utility_share": float("nan"),
                      "locked_pairs": float("nan"), "reversed_pairs": float("nan"),
                      "order_violations": 0, "n_patterns": 0}
    diag = utility_diagnostics(hus)
    print(f"  [{tag}] {len(hus)} patterns ({time.time()-t0:.0f}s mine); "
          f"R={diag['R']:.2f}, utility share {diag['utility_share']:.1%} (Shapley), "
          f"{diag['locked_pairs']:.1%} of pairs locked, {diag['reversed_pairs']:.1%} reversed "
          f"by utility", flush=True)
    if diag["order_violations"]:
        raise RuntimeError(f"[{tag}] {diag['order_violations']} pair(s) are locked by the ceiling "
                           f"yet reversed by utility: the score is not CMNI x mean utility, or R "
                           f"is wrong. Do not use this mine.")
    if diag["reversed_pairs"] == 0:
        print(f"  [{tag}] WARNING: utility reverses no pair relative to CMNI, so this mine ranks "
              f"exactly as FREQUENT subgraph mining would.", flush=True)
    return (list(hus.code) if len(hus) else []), diag


TUPLE = re.compile(r"\((-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+)\)")


def pattern_graph(code: str) -> nx.Graph:
    """A mined DFS code -> small labelled nx graph (node attr 'label', edge attr 'elabel')."""
    g = nx.Graph()
    for i, j, li, lij, lj in (tuple(map(int, m.groups())) for m in TUPLE.finditer(code)):
        g.add_node(i, label=li)
        g.add_node(j, label=lj)
        g.add_edge(i, j, elabel=lij)
    return g


def attempt_portsplit_graph(att, labels: pd.Series) -> tuple[nx.Graph, dict]:
    """The attempt's OWN edges, port-split and labelled the same way as the mining graph.
    Returns (graph, node->account) so an occurrence's distinct-account span can be counted."""
    g = nx.Graph()
    acct_of: dict = {}

    def port(acc, side):
        name = f"{acc}::{side}"
        if name not in g:
            b = int(labels.get(acc, 0))
            g.add_node(name, label=2 * b + (1 if side == "o" else 0))
            acct_of[name] = acc
        return name

    both = set(att.srcs) & set(att.dsts)
    for u, v in zip(att.srcs, att.dsts):
        g.add_edge(port(u, "o"), port(v, "i"), elabel=0)
    for a in both:
        g.add_edge(port(a, "i"), port(a, "o"), elabel=1)
    return g, acct_of


def occurrence_recovery(attempts, codes: list[str], labels: pd.Series) -> pd.DataFrame:
    """Honest, occurrence-level recovery: an attempt is recovered when some mined pattern OCCURS
    as a labelled subgraph inside the attempt's own port-split graph, spanning at least
    min(3, |attempt|) DISTINCT accounts. Subgraph monomorphism matches vertex and edge labels;
    the >=3-account span rules out the trivial one-edge patterns whose image-support union covered
    the whole graph (the inflated measure this replaces)."""
    pats = [g for g in (pattern_graph(c) for c in codes) if g.number_of_edges() >= 2]
    nm = nx.algorithms.isomorphism.categorical_node_match("label", -1)
    em = nx.algorithms.isomorphism.categorical_edge_match("elabel", -1)
    per_att = []
    for att in attempts:
        need = min(3, len(set(att.accounts)))
        G, acct_of = attempt_portsplit_graph(att, labels)
        rec = False
        for pg in pats:
            if pg.number_of_nodes() > G.number_of_nodes() or rec:
                continue
            gm = nx.algorithms.isomorphism.GraphMatcher(G, pg, node_match=nm, edge_match=em)
            for mapping in gm.subgraph_monomorphisms_iter():
                if len({acct_of[n] for n in mapping}) >= need:
                    rec = True
                    break
        per_att.append({"type": att.type, "recovered": int(rec)})
    df = pd.DataFrame(per_att)
    g = df.groupby("type").agg(n_attempts=("recovered", "size"),
                               n_recovered=("recovered", "sum")).reset_index()
    g["recovery_rate"] = g["n_recovered"] / g["n_attempts"]
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HI-Small")
    ap.add_argument("--mode", default="truth-neighborhood",
                    choices=["truth-neighborhood", "full"])
    ap.add_argument("--hub-caps", type=int, nargs="+", default=[0, 200, 50],
                    help="degree caps to sweep; 0 = no cap")
    ap.add_argument("--mer", type=int, default=2)
    ap.add_argument("--max-edges", type=int, nargs="+", default=[3],
                    help="one or more values; each is mined and tagged, so the bipartite/cycle "
                         "sensitivity to the size bound is one run")
    ap.add_argument("--min-util", type=float, default=200.0)
    ap.add_argument("--n-buckets", type=int, default=4)
    ap.add_argument("--utility-scale", default="log", choices=["log", "raw"],
                    help="'log' reproduces the published runs; 'raw' keeps the heavy tail of the "
                         "amounts so the utility dimension is not flattened (see account_signal). "
                         "Labels are identical either way, so the two arms differ only in utility.")
    ap.add_argument("--timeout", type=int, default=900, help="per-mine timeout (s)")
    ap.add_argument("--heap", default="8g", help="JVM max heap for the miner (e.g. 8g, 32g)")
    ap.add_argument("--out-suffix", default="", help="appended to the output CSV name "
                    "so a separate run (e.g. a me=4 probe) does not clobber the main one")
    args = ap.parse_args()
    if isinstance(args.max_edges, int):
        args.max_edges = [args.max_edges]

    if not JAR.exists():
        raise SystemExit(f"missing {JAR} -- build it: mvn -f inchusm-miner/pom.xml package")
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Loading AMLworld {args.dataset}...", flush=True)
    edges, code = load_transactions(DATA / f"{args.dataset}_Trans.csv")
    attempts = load_patterns(DATA / f"{args.dataset}_Patterns.txt", code)
    print(f"  {len(edges):,} transactions, {len(code):,} accounts, {len(attempts)} attempts")

    if args.mode == "truth-neighborhood":
        edges = truth_neighborhood(edges, attempts)
        print(f"  truth-neighborhood subgraph: {len(edges):,} transactions")

    signal = account_signal(edges, scale=args.utility_scale)
    bucket_edges, util = fit_buckets(signal, args.n_buckets)
    labels = assign_bucket(signal, bucket_edges)
    print(f"  monetary buckets: {dict(labels.value_counts().sort_index())}")
    print(f"  utility table ({args.utility_scale}): "
          f"{ {k: float(f'{v:.4g}') for k, v in util.items()} }; "
          f"R = {max(util.values())/min(util.values()):,.2f}")

    # min-util is absolute and u = CMNI * ubar, so widening the table inflates every score.
    # Rescale by the table's mean so the SAME --min-util keeps the same meaning across arms
    # and the two arms stay comparable instead of one of them mining a different-sized set.
    min_util = args.min_util
    if args.utility_scale == "raw":
        ref = fit_buckets(account_signal(edges, scale="log"), args.n_buckets)[1]
        factor = (sum(util.values()) / len(util)) / (sum(ref.values()) / len(ref))
        min_util = args.min_util * factor
        print(f"  min-util rescaled for the wider table: {args.min_util:g} -> {min_util:,.0f} "
              f"(x{factor:,.0f}), so both arms mine at a comparable depth")

    rows = []
    for cap in args.hub_caps:
        ecap = apply_hub_cap(edges, cap)
        lab_cap = labels.loc[list(set(ecap['src']) | set(ecap['dst']))]
        att_present = [a for a in attempts if a.accounts <= (set(ecap['src']) | set(ecap['dst']))]
        for me in args.max_edges:
            tag = f"{args.dataset}_{args.mode}_{args.utility_scale}_cap{cap}_me{me}"
            codes, diag = mine_and_match(ecap, lab_cap, util, tag, args.mer, me, min_util,
                                         heap=args.heap, timeout_s=args.timeout)
            if codes is None:
                # timed out: record every typology as not-computed, keep the sweep going
                types = sorted({a.type for a in attempts})
                tbl = pd.DataFrame({"type": types, "n_attempts": 0, "n_recovered": 0,
                                    "recovery_rate": float("nan"), "timed_out": 1})
            else:
                tbl = occurrence_recovery(attempts, codes, lab_cap)
                tbl["timed_out"] = 0
            tbl["hub_cap"] = cap
            tbl["max_edges"] = me
            tbl["attempts_present"] = len(att_present)
            tbl["utility_scale"] = args.utility_scale
            for k, v in diag.items():
                tbl[k] = v
            rows.append(tbl)
            print(f"  cap={cap} max-edges={me}: "
                  f"{len(att_present)}/{len(attempts)} attempts fully present"
                  + ("  [TIMED OUT]" if codes is None else ""))
            if codes is not None:
                print(tbl[["type", "recovery_rate"]].to_string(index=False))

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / f"e9_portsplit_recovery_{args.dataset}_{args.mode}_{args.utility_scale}{args.out_suffix}.csv", index=False)
    print(f"\nDone ({time.time()-t0:.0f}s) -> "
          f"{OUT / f'e9_portsplit_recovery_{args.dataset}_{args.mode}_{args.utility_scale}{args.out_suffix}.csv'}")


if __name__ == "__main__":
    main()
