"""E5a: high-utility subgraph mining on the Elliptic++ actor graph with real monetary utility.

Flagship experiment for the journal paper. The vertex utility is derived from real
transaction value (btc transacted, fee share) instead of a discretized-label proxy, so
this is high-utility subgraph mining on genuine monetary utility rather than on weights
drawn from a synthetic distribution.

Pipeline:
  load actor graph + wallet monetary features
  -> fit utility buckets on the training window (Time step <= 34) and freeze them
  -> label every wallet by its frozen bucket
  -> prefilter the actor graph (drop self-loops and high-degree hubs)
  -> export the labeled graph (.lg) and the {label -> external utility} table
  -> run the Java miner with measure=utility
  -> report the high-utility subgraphs.

Build the jar first:  mvn -f inchusm-miner/pom.xml package
Run:  python scripts/e5a_ellipticpp_utility.py [--n-buckets 4] [--hub-cap 50] [--min-util AUTO]
Output: results/e5a_ellipticpp_hus.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inchusm_aml.data.ellipticpp import (  # noqa: E402
    TRAIN_MAX_STEP,
    load_ellipticpp,
    prefilter_actor_graph,
)
from inchusm_aml.utility.monetary import (  # noqa: E402
    assign_bucket,
    fit_monetary_utility,
    monetary_signal,
)
from inchusm_aml import miner_bridge  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-buckets", type=int, default=4)
    ap.add_argument("--hub-cap", type=int, default=50, help="drop wallets with degree above this")
    ap.add_argument("--min-util", default="AUTO", help="absolute utility threshold, or AUTO")
    ap.add_argument("--bound", default="ceu", choices=["ivu", "eu", "ceu"])
    ap.add_argument("--max-edges", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    print("Loading Elliptic++ actor graph and monetary features...", flush=True)
    data = load_ellipticpp()
    print(f"  {len(data.wallets):,} wallets, {len(data.edges):,} AddrAddr edges", flush=True)

    # 1) fit monetary-utility buckets on the training window, then freeze
    train_mask = data.wallets["step"] <= TRAIN_MAX_STEP
    buckets = fit_monetary_utility(data.wallets, train_mask, n_buckets=args.n_buckets)
    print(f"  utility buckets (fit on step <= {TRAIN_MAX_STEP}): {buckets.label_utility}", flush=True)

    # 2) label every wallet by its frozen bucket
    signal = monetary_signal(data.wallets)
    labels = assign_bucket(signal, buckets.edges)
    node_label = pd.Series(labels.to_numpy(), index=data.wallets["address"])
    node_label = node_label[~node_label.index.duplicated(keep="first")]

    # 3) prefilter the actor graph, then keep only labeled wallets that survive as edges
    edges = prefilter_actor_graph(data.edges, set(node_label.index), args.hub_cap)
    used = pd.Index(pd.concat([edges["src"], edges["dst"]]).unique())
    node_labels = node_label[node_label.index.isin(used)]
    print(f"  after prefilter: {len(node_labels):,} wallets, {len(edges):,} edges", flush=True)

    # 4) export the labeled graph and the utility table
    RESULTS.mkdir(exist_ok=True)
    graph_lg = RESULTS / "e5a_ellipticpp.lg"
    util_tsv = RESULTS / "e5a_ellipticpp.util.tsv"
    miner_bridge.export_lg(node_labels, edges, graph_lg)
    miner_bridge.export_utilities(buckets.label_utility, util_tsv)

    # 5) pick a threshold. AUTO uses a fraction of the largest single-label utility times a
    #    modest support, as a starting point; calibrate per the printed utility scale.
    max_lab_util = max(buckets.label_utility.values())
    min_util = float(args.min_util) if args.min_util != "AUTO" else round(5 * max_lab_util, 2)
    print(f"  running miner: measure=utility bound={args.bound} minutil={min_util}", flush=True)

    hus, stdout = miner_bridge.run_husm(
        graph_lg, util_tsv, min_util=min_util, bound=args.bound, max_edges=args.max_edges
    )
    print(stdout.strip())

    # 6) report
    out = RESULTS / "e5a_ellipticpp_hus.csv"
    hus.to_csv(out, index=False)
    print(f"\n{len(hus)} high-utility subgraphs (monetary utility).")
    if len(hus):
        top = hus.sort_values("utility", ascending=False).head(10) if "utility" in hus else hus.head(10)
        print(top.to_string(index=False))
    print(f"\nDone ({time.time() - t0:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
