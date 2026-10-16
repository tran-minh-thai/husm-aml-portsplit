"""How high a retention lower bound built from transaction-disjoint chains could ever reach (a feasibility probe).

Chains through an account v that share no transaction conserve independently when amounts are independent of the
graph, so Pr[v retained] >= 1 - (1 - p_m)^D(v), D(v) the number of such chains. Every chain with v at an intermediate
position uses one transaction into v and one out of v, so D(v) <= c(v) = min(in-degree, out-degree). This probe
computes, over the background intermediate accounts of the unconstrained mining (theta = inf, labeling S),
ceiling_m = mean of 1 - (1 - p_m)^c(v): no bound of that form can exceed it. It is set against the measured
background retention r_B: a ceiling below r_B by far means disjoint chains cannot explain the retention, and a ceiling
below 1 / (1 + eps) means such a bound can never certify saturation.

Run: python experiments/probes/disjoint_bound_ceiling.py --dataset sweden_100K_easy --effect <run id>
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from saturation_check import ROOT, p_conserving  # noqa: E402

OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else None


def edges(dataset: str) -> pd.DataFrame:
    path = ROOT / "datasets" / "chain" / f"{dataset}_S.tg"
    frame = pd.read_csv(path, sep=" ", header=None, comment="#", names=["kind", "s", "t", "l", "amount"],
                        dtype={"kind": "category", "s": np.float64, "t": np.float64, "l": np.float64,
                               "amount": np.float64}, engine="c")
    e = frame.loc[frame["kind"] == "e"]
    e = e.loc[e["s"] != e["t"]]                       # the miner drops self-transactions
    return pd.DataFrame({"s": e["s"].astype(np.int64), "t": e["t"].astype(np.int64), "amount": e["amount"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--effect", required=True, help="run id under results-invariant/chain-conservation-effect (labeling S)")
    ap.add_argument("--theta", type=float, default=0.2)
    args = ap.parse_args()
    rec = ROOT / "results-invariant" / "chain-conservation-effect" / args.effect
    effect = {int(r["length"]): r for r in csv.DictReader(open(rec / "conservation_effect.csv"))}
    e = edges(args.dataset)
    p = p_conserving(e["amount"].to_numpy(), args.theta, 6)
    d_out = e["s"].value_counts()
    d_in = e["t"].value_counts()
    laundering = {int(x) for x in (ROOT / "datasets" / "chain" / f"{args.dataset}_laundering_accounts.txt").read_text().split()}
    accounts = pd.read_csv(gzip.open(rec / "accounts_theta_inf.tsv.gz", "rt"), sep="\t")
    rows = []
    for m in range(2, 6):
        ids = accounts.loc[(accounts["kind"] == "open") & (accounts["length"] == m)
                           & (accounts["role"] == "intermediate"), "account"].astype(np.int64)
        ids = ids[~ids.isin(laundering)].to_numpy()
        c = np.minimum(d_in.reindex(ids, fill_value=0).to_numpy(), d_out.reindex(ids, fill_value=0).to_numpy())
        ceiling = float(np.mean(1 - (1 - p[m - 1]) ** c))
        r = effect[m]
        r_b = (float(r["I_0.2"]) - float(r["marked_0.2"])) / (float(r["I_inf"]) - float(r["marked_inf"]))
        rows.append({"dataset": args.dataset, "length": m, "background_intermediates": len(ids),
                     "zero_degree_side": int((c == 0).sum()), "median_c": float(np.median(c)), "p_m": p[m - 1],
                     "ceiling": ceiling, "r_B": r_b, "ceiling_over_r_B": ceiling / r_b})
    for row in rows:
        print(", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    if OUT is not None:
        with open(OUT / "disjoint_bound_ceiling.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
