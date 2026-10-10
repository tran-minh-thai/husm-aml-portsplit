"""Exploratory: is the pair at the account that starts a laundering cycle conserving?

A cycle attempt starts at the account that sends its earliest transaction (the origin) and ends
when money comes back to it. For chain-node pairs (the account receives one and sends one
transaction inside the attempt) of CYCLE attempts, this splits the ratio a_out / a_in into the
origin pair (incoming = the transaction that returns, outgoing = the first transaction) and the
interior pairs, by cycle length. Not pre-registered; it decides whether a cycle pattern must also
constrain the pair that closes the cycle.

Run: python experiments/probes/cycle_origin_pairs.py --source amlgentex --dataset sweden_100K_easy
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

from inchusm_aml.eval import flow_gates as fg  # noqa: E402
from round2_measurements import OUT, load  # noqa: E402

THETAS = (0.1, 0.2, 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    _, ptx, _ = load(args.source, args.dataset)
    cyc = ptx[(ptx["type"] == "CYCLE") & (ptx["src"] != ptx["dst"])]
    first = cyc.sort_values(["attempt", "time"], kind="stable").groupby("attempt").head(1)
    origin = first.set_index("attempt")["src"]
    n_tx = cyc.groupby("attempt").size()
    ties = cyc.groupby("attempt")["time"].apply(lambda t: int((t == t.min()).sum()) > 1)
    cp = fg.chain_pairs(cyc)
    cp = cp[(cp["a1"] > 0) & (cp["a2"] > 0)].copy()
    cp["at_origin"] = cp["v"].to_numpy() == cp["attempt"].map(origin).to_numpy()
    cp["length"] = cp["attempt"].map(n_tx)
    cp["origin_time_tie"] = cp["attempt"].map(ties)
    cp["ratio"] = cp["a2"] / cp["a1"]
    rows = []
    for (at_origin, length), g in cp.groupby(["at_origin", "length"]):
        row = {"at_origin": bool(at_origin), "cycle_length": int(length), "n_pairs": len(g),
               "n_attempts": g["attempt"].nunique(), "median_ratio": float(g["ratio"].median()),
               "n_origin_time_ties": int(g["origin_time_tie"].sum())}
        for th in THETAS:
            row[f"within_{th:g}"] = float(((g["ratio"] >= 1 - th) & (g["ratio"] <= 1 + th)).mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "cycle_origin_pairs.csv", index=False)
    print(f"{args.source} {args.dataset}: {cyc['attempt'].nunique()} CYCLE attempts, {len(cp)} chain-node pairs, "
          f"{int(ties.sum())} attempts whose earliest time is shared by more than one transaction")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for at_origin in (True, False):
        g = cp[cp["at_origin"] == at_origin]
        if len(g):
            print(f"  {'origin' if at_origin else 'interior'} pairs: {len(g)}, within 0.2: "
                  f"{float(((g['ratio'] >= 0.8) & (g['ratio'] <= 1.2)).mean()):.3f}, "
                  f"quartiles of ratio: {np.quantile(g['ratio'], [0.25, 0.5, 0.75]).round(3).tolist()}")


if __name__ == "__main__":
    main()
