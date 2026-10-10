"""Second-round gate measurements: which hereditary time constraint fits laundering attempts.

Pre-registered before any value was seen. For one set, writes round2_order.csv (pair-level in-order
rates and attempt-level shares with at most k reversed head-to-tail pairs, counting every pair (a1)
or skipping back-and-forth pairs (a2)), round2_window.csv (attempt durations and shares within each
window) and prints the two hypotheses with the values they are decided on. The decision combines
two sets and is not made here. On AMLGentex it also writes round2_amount.csv, the amount comparison
of the first round, which was not pre-registered for that source and is printed as exploratory.

Run: python experiments/probes/round2_measurements.py --source amlworld --dataset HI-Small
     python experiments/probes/round2_measurements.py --source amlgentex --dataset sweden_100K_easy
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))

from inchusm_aml.eval import flow_gates as fg  # noqa: E402

ROOT = HERE.parents[2]
OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else HERE.parents[1] / "results"
DELTAS = (1, 2, 3, 5, 7, 10)
H4_TYPES = ("GATHER-SCATTER", "SCATTER-GATHER", "STACK", "RANDOM")


def load(source: str, dataset: str):
    """(edges, ptx, time units per day)."""
    if source == "amlworld":
        from inchusm_aml.data.amlworld import load_pattern_transactions, load_transactions
        d = ROOT / "datasets" / "amlworld"
        edges, code = load_transactions(d / f"{dataset}_Trans.csv")
        ptx = load_pattern_transactions(d / f"{dataset}_Patterns.txt", code, edges.attrs["usd_rates"])
        return edges, ptx, 1440
    if source == "amlgentex":
        from inchusm_aml.data.amlgentex import load_amlgentex
        edges, ptx = load_amlgentex(ROOT / "datasets" / "amlgentex" / dataset / "tx_log.csv")
        print(f"  rows {edges.attrs['rows_total']:,}; dropped {edges.attrs['rows_special']:,} rows with the income "
              f"source or spending sink; {edges.attrs['sar_without_alert']:,} flagged rows outside any attempt")
        return edges, ptx, 1
    raise SystemExit(f"unknown source {source!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    print(f"{args.source} {args.dataset}")
    edges, ptx, unit = load(args.source, args.dataset)
    span = (edges["time"].max() - edges["time"].min()) / unit
    print(f"  {len(edges):,} transactions spanning {span:.2f} days; {ptx['attempt'].nunique()} attempts, "
          f"{len(ptx):,} attempt transactions")

    order = fg.order_tolerance_gate(ptx)
    order.to_csv(OUT / "round2_order.csv", index=False)
    print("\n[order] head-to-tail pairs (t1 <= t2 is in order)")
    print(order.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    h4 = order.set_index("type")
    parts, counted = [], []
    for t in H4_TYPES:
        if t not in h4.index or h4.loc[t, "n_pairs_a2"] < 30:
            parts.append(f"{t}: {0 if t not in h4.index else int(h4.loc[t, 'n_pairs_a2'])} pairs < 30, not counted")
            continue
        counted.append(t)
        parts.append(f"{t}: {h4.loc[t, 'in_order_a2']:.3f} of {int(h4.loc[t, 'n_pairs_a2'])}")
    holds4 = bool(counted) and all(h4.loc[t, "in_order_a2"] >= 0.90 for t in counted)
    print("    H4 on this set (a2 pair-level in-order >= 0.90 each): " + "; ".join(parts) + f"; holds here: {holds4}")
    for t in h4.index:
        if h4.loc[t, "n_attempts_with_pair"] == 0:
            print(f"    {t}: no head-to-tail pair in any attempt")
            continue
        ks = [k for k in fg.KS if h4.loc[t, f"attempts_V_le_{k}_a2"] >= 0.90]
        print(f"    {t}: smallest k with >= 90% of attempts at V <= k (a2): {ks[0] if ks else '> 3'}")

    window = fg.window_gate(ptx, DELTAS, unit)
    window.to_csv(OUT / "round2_window.csv", index=False)
    print("\n[window] attempt durations in days")
    print(window.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    ok = [d for d in DELTAS if (window[f"within_{d}d"] >= 0.95).all() and d <= span / 2]
    print(f"    H5 on this set (>= 95% of every typology within delta, delta <= {span / 2:.2f} days): "
          f"deltas {ok or 'none'}; holds here: {bool(ok)}")

    if args.source == "amlgentex":
        amount = fg.amount_gate(edges, ptx)
        amount.to_csv(OUT / "round2_amount.csv", index=False)
        print(f"\n[amount, exploratory: not pre-registered for this source] vs "
              f"{amount.attrs['n_background']:,} background transactions")
        print(amount.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
