"""How long the planted laundering chains are: the evidence for the maximum length of the mining.

For one data set, every laundering attempt's longest planted open chain and every planted closed chain (a CYCLE
attempt read from its origin), with lengths counted in transactions (planted_chains, no length cap below 64).
Writes planted_lengths.csv: kind, length, number of attempts whose longest chain of that kind has that length,
and the cumulative share of attempts with a longest chain of at most that length.

Run: python experiments/scripts/planted_lengths.py --source amlworld --dataset HI-Small
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "probes"))

from inchusm_aml.eval import planted_chains as pc  # noqa: E402
from round2_measurements import load  # noqa: E402

CAP = 64
OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    _, ptx, _ = load(args.source, args.dataset)
    chains = pc.planted_chains(ptx, CAP)
    if chains.empty:
        sys.exit(f"{args.dataset}: no planted chains")
    if int(chains["length"].max()) >= CAP:
        sys.exit(f"{args.dataset}: a planted chain reaches the cap of {CAP} transactions; raise CAP")
    longest = chains.groupby(["kind", "attempt"])["length"].max().reset_index()
    rows = []
    for kind, g in longest.groupby("kind"):
        counts = g["length"].value_counts().sort_index()
        total, cum = int(counts.sum()), 0
        for length, n in counts.items():
            cum += int(n)
            rows.append({"dataset": args.dataset, "kind": kind, "length": int(length), "attempts": int(n),
                         "share_at_most": cum / total, "attempts_of_kind": total})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "planted_lengths.csv", index=False)
    print(f"{args.source} {args.dataset}: longest planted chain per attempt (transactions)")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
