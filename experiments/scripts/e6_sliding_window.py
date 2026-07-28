"""E6 - sliding-window IncHUSM: exactness and bounded state, on both streams.

The add-only miner never forgets, so its state grows with the stream. This experiment runs
the same stream under a sliding window and measures the two claims of the theory:

  (exactness)      with --window W, the maintained high set must equal static HUSM on the
                   WINDOW union after every snapshot (match='OK'). This is Theorem T1' --
                   the add-only run checks the same thing against the cumulative union, so
                   both modes are validated by the same column.
  (bounded state)  'tracked' under a window must flatten, while the add-only run keeps
                   growing. This is the Proposition: state is bounded by the window
                   content, not by the stream length.

Two streams, because the paper claims the window and the monetary utility together and they
have to be shown together:

  --stream elliptic  the 49 Elliptic time steps with the conference protocol (vertex label =
                     feature f0 in K quantile buckets fit on step<=34 and frozen; external
                     utility of label i is i+1). A proxy utility, but directly comparable to
                     the conference streaming numbers.
  --stream actor     the Elliptic++ actor stream carrying REAL monetary utility (buckets fit
                     on the training window and frozen, external utility = mean monetary
                     signal). Snapshots are disjoint because wallet identity is not carried
                     across them, so T1' applies here too.

Note the window makes its own verification affordable: static re-mining is bounded to the W
snapshots of the window rather than growing with the accumulated union, which is what makes
--compare-static usable on the actor stream at all.

Run:  python scripts/e6_sliding_window.py                       # elliptic, windows 5,10
      python scripts/e6_sliding_window.py --stream actor --windows 5
      python scripts/e6_sliding_window.py --max-steps 8 --windows 3   # quick check
Output: results/e6_sliding_window_<stream>.csv + a summary table.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inchusm_aml.miner_bridge import (  # noqa: E402
    export_snapshot_stream, export_utilities, run_inchusm_stream,
)
from inchusm_aml.data.ellipticpp import (  # noqa: E402
    build_actor_stream, load_ellipticpp, prefilter_actor_graph,
)
from inchusm_aml.utility.monetary import (  # noqa: E402
    assign_bucket, fit_monetary_utility, monetary_signal,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parents[1] / "results"
ELLIPTIC = ROOT / "datasets" / "elliptic1"
TRAIN_MAX_STEP = 34


def build_actor_snapshots(k: int, max_steps: int, hub_cap: int):
    """(snapshots, utility table) for the Elliptic++ actor stream with REAL monetary utility.

    Same construction as the monetary experiments: buckets fit on the training window and
    frozen, external utility = the bucket's mean monetary signal. Each snapshot is its own
    graph, so the snapshots are disjoint and the window theory applies.
    """
    data = load_ellipticpp()
    train_mask = data.wallets["step"] <= TRAIN_MAX_STEP
    buckets = fit_monetary_utility(data.wallets, train_mask, n_buckets=k)
    signal = monetary_signal(data.wallets)
    labels = assign_bucket(signal, buckets.edges)
    node_label = pd.Series(labels.to_numpy(), index=data.wallets["address"])
    node_label = node_label[~node_label.index.duplicated(keep="first")]
    step_of = data.wallets.drop_duplicates("address").set_index("address")["step"]

    edges = prefilter_actor_graph(data.edges, set(node_label.index), hub_cap)
    stream = [(lab, e) for t, lab, e in build_actor_stream(edges, node_label, step_of)
              if t <= max_steps]
    return stream, buckets.label_utility


def build_snapshots(k: int, max_steps: int):
    """(snapshots, n_labels); label = quantile bucket of f0, edges fit on train then frozen."""
    df = pd.read_csv(
        ELLIPTIC / "elliptic_txs_features.csv", header=None, usecols=[0, 1, 2],
        names=["txId", "step", "f0"],
        dtype={0: np.int64, 1: np.int16, 2: np.float32},
    )
    train_f0 = df.loc[df.step <= TRAIN_MAX_STEP, "f0"]
    _, bins = pd.qcut(train_f0, k, retbins=True, duplicates="drop")
    bins = bins.copy()
    bins[0], bins[-1] = -np.inf, np.inf
    df["label"] = pd.cut(df["f0"], bins=bins, labels=False, include_lowest=True).astype(np.int32)
    n_labels = int(df["label"].nunique())

    edges = pd.read_csv(ELLIPTIC / "elliptic_txs_edgelist.csv")
    edges.columns = ["src", "dst"]
    step_of = df.set_index("txId")["step"]
    edges["step"] = step_of.reindex(edges["src"]).to_numpy(dtype=np.int16)

    steps = sorted(df["step"].unique())[:max_steps]
    snaps = []
    for s in steps:
        nd = df[df.step == s]
        snaps.append((nd.set_index("txId")["label"], edges.loc[edges.step == s, ["src", "dst"]]))
    return snaps, n_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=["elliptic", "actor"], default="elliptic",
                    help="elliptic = proxy utility on the 49 time steps; "
                         "actor = REAL monetary utility on the Elliptic++ actor stream")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-value", type=float, default=None,
                    help="utility threshold (default 500 for elliptic, 100 for actor)")
    ap.add_argument("--hub-cap", type=int, default=50, help="actor stream only")
    ap.add_argument("--bound", choices=["ivu", "eu", "ceu"], default="ceu")
    ap.add_argument("--max-edges", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=49)
    ap.add_argument("--windows", type=int, nargs="+", default=[5, 10],
                    help="sliding-window lengths to run (add-only is always run first)")
    ap.add_argument("--no-add-only", action="store_true",
                    help="skip the add-only mode; on the actor stream its static re-run "
                         "grows with the accumulated union and is far slower than a window's")
    ap.add_argument("--no-compare", action="store_true",
                    help="skip the static re-run (drops the exactness column)")
    args = ap.parse_args()
    min_value = args.min_value if args.min_value is not None \
        else (500.0 if args.stream == "elliptic" else 100.0)

    work = RESULTS / f"e6_snapshots_{args.stream}"
    work.mkdir(parents=True, exist_ok=True)
    print(f"[E6] building {args.stream} snapshots (k={args.k}, max_steps={args.max_steps}) ...",
          flush=True)
    t0 = time.time()
    if args.stream == "actor":
        snaps, util_tbl = build_actor_snapshots(args.k, args.max_steps, args.hub_cap)
        desc = f"{len(snaps)} snapshots, monetary utility {util_tbl}"
    else:
        snaps, n_labels = build_snapshots(args.k, args.max_steps)
        util_tbl = {i: float(i + 1) for i in range(n_labels)}
        desc = f"{len(snaps)} snapshots, {n_labels} labels, proxy utility i+1"
    # The miner reads EVERY .lg in this directory, so snapshots left by an earlier run with
    # a larger --max-steps would silently be mined as well; clear them first.
    stale = sorted(work.glob("snap_*.lg"))
    for f in stale:
        f.unlink()
    if stale:
        print(f"[E6] removed {len(stale)} snapshot file(s) from a previous run", flush=True)
    export_snapshot_stream(snaps, work)
    util_path = work / "utilities.tsv"
    export_utilities(util_tbl, util_path)
    print(f"[E6] {desc} ({time.time()-t0:.1f}s)\n", flush=True)

    RESULTS.mkdir(exist_ok=True)
    frames = []
    modes = ([] if args.no_add_only else [0]) + list(args.windows)
    for window in modes:
        tag = "add-only" if window == 0 else f"W={window}"
        print(f"{'='*64}\n[E6] running {tag} ({args.stream})\n{'='*64}", flush=True)
        df, _ = run_inchusm_stream(
            work, util_path, min_value, measure="utility", bound=args.bound,
            max_edges=args.max_edges, compare_static=not args.no_compare,
            window=window, out_csv=RESULTS / f"e6_raw_{args.stream}_w{window}.csv",
        )
        df["mode"] = tag
        df["window"] = window
        frames.append(df)
        n_ok = int((df["match"] == "OK").sum()) if not args.no_compare else -1
        print(f"  -> tracked: first={int(df.tracked.iloc[0])} "
              f"max={int(df.tracked.max())} final={int(df.tracked.iloc[-1])}"
              f" | high final={int(df.high_total.iloc[-1])}"
              f" | inc total={int(df.inc_ms.sum())}ms"
              + (f" | exact {n_ok}/{len(df)}" if n_ok >= 0 else ""), flush=True)

    all_df = pd.concat(frames, ignore_index=True)
    out = RESULTS / f"e6_sliding_window_{args.stream}.csv"
    all_df.to_csv(out, index=False)

    print(f"\n{'='*72}\nE6 SUMMARY - sliding window, stream={args.stream}, {args.max_steps} steps\n{'='*72}")
    print(f"{'mode':<10}{'exact':>10}{'tracked max':>14}{'tracked final':>15}"
          f"{'high final':>12}{'inc ms':>10}")
    for df in frames:
        tag = df["mode"].iloc[0]
        ok = (f"{int((df['match']=='OK').sum())}/{len(df)}"
              if not args.no_compare else "-")
        print(f"{tag:<10}{ok:>10}{int(df.tracked.max()):>14}"
              f"{int(df.tracked.iloc[-1]):>15}{int(df.high_total.iloc[-1]):>12}"
              f"{int(df.inc_ms.sum()):>10}")

    base = frames[0]
    print("\nstate growth (tracked patterns) over the stream:")
    marks = [i for i in (0, len(base) // 4, len(base) // 2, 3 * len(base) // 4, len(base) - 1)]
    header = "  t=" + "".join(f"{int(base.snapshot.iloc[i]):>8}" for i in marks)
    print(header)
    for df in frames:
        row = "".join(f"{int(df.tracked.iloc[i]):>8}" for i in marks)
        print(f"  {df['mode'].iloc[0]:<8}" + row)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
