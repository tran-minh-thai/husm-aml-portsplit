"""E5c: IncHUSM-mu, relative (data-scaled) utility threshold with a pre-large safety band.

The conference algorithm uses a fixed absolute threshold. Here the threshold is a fraction
mu of the data scale, so the effective absolute threshold grows as the actor stream
accumulates. That breaks the "once high, always high" monotonicity, which the pre-large band
is meant to restore: patterns are reported as LARGE only at or above mu_upper * scale,
tracked as PRE-LARGE down to mu_lower * scale, and exactness holds while the scale added
since the last full mine stays under the safety bound; beyond it, exactness needs a re-mine.

This script reports a NEGATIVE result, which the paper states. The bound only helps when each
increment is small relative to the accumulated data -- the situation pre-large was designed
for, where one transaction joins a database of millions. A 49-snapshot stream is not that
situation: a snapshot adds a median 3.2% of the accumulated scale (46% at the head of the
stream) while the bound at mu=(0.02, 0.01) absorbs 1.02% of it. An exact run would therefore
re-mine at 46 of 49 snapshots, which is what an incremental miner exists to avoid, and no
parameter choice rescues it: widening the band enough means raising mu_upper until the
reported set is empty. remine_report() prints that schedule. The miner run below is kept
because it produces the state figures, but note that the engine does NOT enforce the bound
(see RelativeThreshold), so its reported set carries no exactness guarantee here.

Build the jar first:  mvn -f inchusm-miner/pom.xml package
Run:  python scripts/e5c_inchusm_mu.py [--mu-upper 0.02] [--mu-lower 0.01] [--n-buckets 4]
Output: results/e5c_inchusm_mu.csv
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


def per_snapshot_scale(wallets: pd.DataFrame, labels: pd.Series, label_utility: dict) -> pd.DataFrame:
    """Cumulative external-utility weight of the data by Time step (the relative-threshold base).

    Each wallet contributes the external utility of its bucket once, at its first-appearance
    step; the cumulative sum over steps is the data scale used for mu * scale.
    """
    util = labels.map(label_utility).to_numpy()
    by_step = pd.Series(util, index=wallets["step"].to_numpy()).groupby(level=0).sum()
    scale = by_step.sort_index().cumsum()
    return scale.rename("scale").rename_axis("step").reset_index()


def remine_schedule(scale: pd.Series, mu_upper: float, mu_lower: float) -> list[int]:
    """Snapshots at which an EXACT mu-run would have to re-mine from scratch.

    The safety bound says an answer stays exact while the scale added since the last full
    mine stays under (mu_upper - mu_lower) * s0 / (1 - mu_upper). Whenever the accumulated
    delta passes that, exactness requires a full re-mine and the reference scale resets.
    """
    bound = lambda s: (mu_upper - mu_lower) * s / (1.0 - mu_upper)
    s0 = scale.iloc[0]
    out = []
    for step, s in scale.items():
        if s - s0 > bound(s0):
            out.append(int(step))
            s0 = s
    return out


def remine_report(sched: pd.DataFrame, mu_upper: float, mu_lower: float) -> None:
    """Report what the safety bound actually costs on this stream.

    This is the honest measure of the pre-large technique: its whole point is to avoid
    re-mining, so if the bound forces a re-mine at nearly every snapshot it buys nothing.
    """
    s = sched.set_index("step")["scale"]
    ratio = (s.diff() / s.shift()).dropna()
    band = (mu_upper - mu_lower) / (1.0 - mu_upper)
    print(f"\n  --- what the safety bound costs (mu_upper={mu_upper}, mu_lower={mu_lower}) ---")
    print(f"  a snapshot adds this share of the accumulated scale: "
          f"head {ratio.iloc[0]*100:.1f}% | median {ratio.median()*100:.1f}% | "
          f"tail {ratio.iloc[-1]*100:.2f}%")
    print(f"  the bound absorbs only {band*100:.2f}% of the scale, so it is exceeded whenever "
          f"a snapshot is larger than that")
    for mu, ml in [(mu_upper, mu_lower), (mu_upper, mu_lower / 10),
                   (0.05, 0.01), (0.10, 0.01), (0.20, 0.01), (0.50, 0.01)]:
        n = len(remine_schedule(s, mu, ml))
        b = (mu - ml) / (1.0 - mu)
        print(f"    mu=({mu:<4}, {ml:<6}) bound={b*100:5.2f}% of s -> {n:2d}/{len(s)} re-mines"
              f"   (reporting floor at t=last: {mu*s.iloc[-1]:,.0f})")
    print("  Reducing re-mines needs a wide band, but mu_upper IS the reporting threshold: at")
    print("  mu_upper=0.5 a pattern must carry half of ALL the data's utility to be reported.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu-upper", type=float, default=0.02, help="large fraction (reported)")
    ap.add_argument("--mu-lower", type=float, default=0.01, help="pre-large fraction (buffer)")
    ap.add_argument("--n-buckets", type=int, default=4)
    ap.add_argument("--hub-cap", type=int, default=50)
    ap.add_argument("--bound", default="ceu", choices=["ivu", "eu", "ceu"])
    ap.add_argument("--max-edges", type=int, default=5)
    args = ap.parse_args()
    assert 0 < args.mu_lower <= args.mu_upper <= 1, "require 0 < mu_lower <= mu_upper <= 1"

    t0 = time.time()
    print("Loading Elliptic++ actor graph and monetary features...", flush=True)
    data = load_ellipticpp()

    train_mask = data.wallets["step"] <= TRAIN_MAX_STEP
    buckets = fit_monetary_utility(data.wallets, train_mask, n_buckets=args.n_buckets)
    signal = monetary_signal(data.wallets)
    labels = assign_bucket(signal, buckets.edges)

    # relative thresholds per snapshot from the data scale (runs without the miner)
    scale = per_snapshot_scale(data.wallets, labels, buckets.label_utility)
    scale["theta_upper"] = args.mu_upper * scale["scale"]
    scale["theta_lower"] = args.mu_lower * scale["scale"]
    scale["safety_bound"] = (args.mu_upper - args.mu_lower) * scale["scale"] / (1.0 - args.mu_upper)
    RESULTS.mkdir(exist_ok=True)
    scale.to_csv(RESULTS / "e5c_mu_schedule.csv", index=False)
    print("  data scale and relative thresholds by step:")
    print(scale.head().to_string(index=False))
    print("  ...")
    print(scale.tail(3).to_string(index=False))

    remine_report(scale, args.mu_upper, args.mu_lower)

    # Drive the miner in mu mode: IncHUSM tracks patterns down to theta_lower(scale_t) and the
    # driver reports those at or above theta_upper(scale_t) as LARGE (miner column "large").
    node_label = pd.Series(labels.to_numpy(), index=data.wallets["address"])
    node_label = node_label[~node_label.index.duplicated(keep="first")]
    step_of = data.wallets.drop_duplicates("address").set_index("address")["step"]
    edges = prefilter_actor_graph(data.edges, set(node_label.index), args.hub_cap)
    stream = list(build_actor_stream(edges, node_label, step_of))
    print(f"\n  prefiltered actor graph: {len(node_label):,} labeled wallets, "
          f"{len(edges):,} edges, {len(stream)} snapshots")

    snap_dir = RESULTS / "e5c_snapshots"
    miner_bridge.export_snapshot_stream([(lab, e) for _, lab, e in stream], snap_dir)
    util_tsv = RESULTS / "e5c_ellipticpp.util.tsv"
    miner_bridge.export_utilities(buckets.label_utility, util_tsv)

    print(f"  running IncHUSM-mu: mu_upper={args.mu_upper} mu_lower={args.mu_lower} "
          f"bound={args.bound}", flush=True)
    df, stdout = miner_bridge.run_inchusm_stream(
        snap_dir, util_tsv, min_value=0.0, measure="utility",
        bound=args.bound, max_edges=args.max_edges,
        mu_upper=args.mu_upper, mu_lower=args.mu_lower,
    )
    print(stdout.strip())
    df.to_csv(RESULTS / "e5c_inchusm_mu.csv", index=False)
    if "large" in df.columns:
        print(f"\n  large (>= theta_upper) by snapshot: {df['large'].tolist()}")
        print(f"  tracked (pre-large + large):        {df['tracked'].tolist()}")

    print(f"\nDone ({time.time() - t0:.0f}s) -> {RESULTS / 'e5c_inchusm_mu.csv'}")


if __name__ == "__main__":
    main()
