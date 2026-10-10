"""Exploratory (not pre-registered): node-level flow conservation inside laundering attempts.

For each pass-through account of an attempt (it both receives and sends among the attempt's
transactions), the ratio of the amount it sends to the amount it receives within that attempt.
A generator that splits a received total among several outgoing transfers conserves flow at this
level while a single incoming/outgoing pair need not keep the amount. Prints quartiles per
typology and the share of accounts with a ratio in [0.8, 1.0] and in [0.85, 0.95]; writes
round3_node_flow.csv. No background comparison: this only checks what the generator does.

Run: python experiments/probes/round3_node_flow.py --source amlgentex --dataset sweden_100K_easy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent))

from round2_measurements import OUT, load  # noqa: E402


def node_flow(ptx: pd.DataFrame) -> pd.DataFrame:
    p = ptx[ptx["src"] != ptx["dst"]]
    inc = p.groupby(["attempt", "dst"])["amount"].sum().rename("amount_in")
    inc.index = inc.index.set_names(["attempt", "account"])
    out = p.groupby(["attempt", "src"])["amount"].sum().rename("amount_out")
    out.index = out.index.set_names(["attempt", "account"])
    acc = pd.concat([inc, out], axis=1, join="inner").reset_index()
    acc["type"] = acc["attempt"].map(p.groupby("attempt")["type"].first())
    acc["ratio_out_in"] = acc["amount_out"] / acc["amount_in"]
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    edges, ptx, _ = load(args.source, args.dataset)
    acc = node_flow(ptx)
    acc.to_csv(OUT / "round3_node_flow.csv", index=False)
    rows = []
    for t, g in acc.groupby("type"):
        r = g["ratio_out_in"].to_numpy()
        rows.append({"type": t, "n_accounts": len(r), "ratio_q1": np.quantile(r, 0.25),
                     "ratio_median": np.median(r), "ratio_q3": np.quantile(r, 0.75),
                     "within_0.8_1.0": float(((r >= 0.8) & (r <= 1.0)).mean()),
                     "within_0.85_0.95": float(((r >= 0.85) & (r <= 0.95)).mean())})
    print(f"{args.source} {args.dataset} (exploratory): sent / received at pass-through accounts")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
