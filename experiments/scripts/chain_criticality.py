"""mu(theta): mean number of conserving continuations of a background transaction.

For every background transaction e = (u -> v) (not flagged, not a self-transaction), count the
background transactions f = (v -> w) with w != u and (1 - theta) a(e) <= a(f) <= (1 + theta) a(e);
mu is the mean of these counts over all background transactions. It is the mean offspring number of
the walk that keeps conserving the amount through background accounts: below 1 the number of
conserving background chains through an account shrinks geometrically with their length.

Counting is exact: amounts are replaced by their rank among all distinct amounts, so every range
query compares the same binary64 values as the conservation predicate.

Run: python experiments/scripts/chain_criticality.py --source amlworld --dataset HI-Small
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "probes"))

from round2_measurements import load  # noqa: E402

OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"
THETAS = (0.05, 0.1, 0.2, 0.5)


def continuation_counts(src: np.ndarray, dst: np.ndarray, amt: np.ndarray, theta: float) -> np.ndarray:
    """For every transaction e, the number of transactions f with s(f) = t(e), t(f) != s(e), cons(e, f)."""
    values = np.unique(amt)
    rank = np.searchsorted(values, amt)
    big = np.int64(len(values) + 1)
    order = np.lexsort((rank, src))
    key = src[order].astype(np.int64) * big + rank[order]
    lo_rank = np.searchsorted(values, (1 - theta) * amt, side="left")            # first value >= lower bound
    hi_rank = np.searchsorted(values, (1 + theta) * amt, side="right") - 1       # last value <= upper bound
    base = dst.astype(np.int64) * big
    count = np.searchsorted(key, base + hi_rank, side="right") - np.searchsorted(key, base + lo_rank, side="left")
    count = np.where(hi_rank >= lo_rank, count, 0)
    # remove the continuations that go straight back to the sender (back-and-forth)
    e = pd.DataFrame({"i": np.arange(len(src)), "u": src, "v": dst, "a": amt})
    f = pd.DataFrame({"v": src, "w": dst, "b": amt})
    back = e.merge(f, left_on=["v", "u"], right_on=["v", "w"])
    back = back[(back["b"] >= (1 - theta) * back["a"]) & (back["b"] <= (1 + theta) * back["a"])]
    np.subtract.at(count, back["i"].to_numpy(), 1)
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    edges, _, _ = load(args.source, args.dataset)
    e = edges[(edges["laundering"] == 0) & (edges["src"] != edges["dst"])]
    src, dst, amt = e["src"].to_numpy(), e["dst"].to_numpy(), e["amount"].to_numpy(dtype=float)
    rows = []
    for theta in THETAS:
        c = continuation_counts(src, dst, amt, theta)
        rows.append({"dataset": args.dataset, "theta": theta, "background_transactions": len(c), "mu": float(c.mean()),
                     "share_with_a_continuation": float((c > 0).mean()), "median": float(np.median(c)),
                     "p99": float(np.percentile(c, 99))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "criticality.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
