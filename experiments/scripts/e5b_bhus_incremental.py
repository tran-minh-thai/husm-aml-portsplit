"""E5b: incremental Boundary High-Utility Subgraphs on the Elliptic++ actor stream.

Builds an add-only snapshot stream from the wallet Time step, runs IncHUSM with real
monetary utility (measure=utility), and maintains the compact BHUS boundaries per snapshot
(BHUSInc, on the Java side). The point of the experiment is twofold:
  - compactness: the boundary (MaxBHUS + MinBHUS) is much smaller than the full high set,
    so it is a compact summary of what has become high-utility so far;
  - incrementality: the boundary is updated from N_t each snapshot in O(|N_t|), rather than
    recomputed from the full HUS set (BHUS.extract) at every snapshot.

This extends the conference algorithm in two ways at once: the utility is real monetary
value (not a proxy), and the maintained object is the compact boundary, not the full set.

Build the jar first:  mvn -f inchusm-miner/pom.xml package
Run:  python scripts/e5b_bhus_incremental.py [--n-buckets 4] [--hub-cap 50] [--min-value AUTO]
Output: results/e5b_bhus_stream.csv
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
    build_actor_stream,
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
    ap.add_argument("--hub-cap", type=int, default=50)
    ap.add_argument("--min-value", default="AUTO", help="absolute utility threshold, or AUTO")
    ap.add_argument("--bound", default="ceu", choices=["ivu", "eu", "ceu"])
    ap.add_argument("--max-edges", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    print("Loading Elliptic++ actor graph and monetary features...", flush=True)
    data = load_ellipticpp()

    # utility buckets fit on the training window, then frozen (same as e5a)
    train_mask = data.wallets["step"] <= TRAIN_MAX_STEP
    buckets = fit_monetary_utility(data.wallets, train_mask, n_buckets=args.n_buckets)
    signal = monetary_signal(data.wallets)
    labels = assign_bucket(signal, buckets.edges)
    node_label = pd.Series(labels.to_numpy(), index=data.wallets["address"])
    node_label = node_label[~node_label.index.duplicated(keep="first")]
    step_of = data.wallets.drop_duplicates("address").set_index("address")["step"]

    # prefilter, then split into an add-only snapshot stream by Time step
    edges = prefilter_actor_graph(data.edges, set(node_label.index), args.hub_cap)
    stream = list(build_actor_stream(edges, node_label, step_of))
    print(f"  {len(stream)} snapshots; "
          f"{sum(len(e) for _, _, e in stream):,} edges after prefilter", flush=True)

    # export the .lg sequence + the frozen utility table
    RESULTS.mkdir(exist_ok=True)
    snap_dir = RESULTS / "e5b_snapshots"
    miner_bridge.export_snapshot_stream([(lab, e) for _, lab, e in stream], snap_dir)
    util_tsv = RESULTS / "e5b_ellipticpp.util.tsv"
    miner_bridge.export_utilities(buckets.label_utility, util_tsv)

    max_lab_util = max(buckets.label_utility.values())
    min_value = float(args.min_value) if args.min_value != "AUTO" else round(5 * max_lab_util, 2)
    print(f"  running IncHUSM stream: measure=utility bound={args.bound} minvalue={min_value}",
          flush=True)

    # IncHUSM over the stream. When the miner is wired with --bhus-inc, the telemetry gains
    # max_bhus / min_bhus columns (BHUSInc maintained per snapshot); until then this reports
    # the tracked/high counts, which already show the high set growing monotonically.
    df, stdout = miner_bridge.run_inchusm_stream(
        snap_dir, util_tsv, min_value=min_value, measure="utility",
        bound=args.bound, max_edges=args.max_edges, bhus_inc=True,
    )
    print(stdout.strip())

    out = RESULTS / "e5b_bhus_stream.csv"
    df.to_csv(out, index=False)
    if {"max_bhus", "min_bhus", "high"}.issubset(df.columns):
        last = df.iloc[-1]
        print(f"\nAt t={len(df)}: full high set = {int(last['high'])}, "
              f"boundary = {int(last['max_bhus']) + int(last['min_bhus'])} "
              f"(Max {int(last['max_bhus'])} + Min {int(last['min_bhus'])})")
    print(f"\nDone ({time.time() - t0:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
