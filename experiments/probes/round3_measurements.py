"""Third-round gate measurement: is the amount passed on unchanged along laundering head-to-tail pairs?

Pre-registered before any value was seen. For one set, compares the deviation |ln(a2 / a1)| of
laundering head-to-tail pairs (both transactions in one attempt, back-and-forth pairs skipped) with
200,000 sampled background pairs (seed 20261010), writes round3_conservation.csv and prints the
hypothesis with the value it is decided on. The decision combines four sets and is not made here.

Run: python experiments/probes/round3_measurements.py --source amlworld --dataset HI-Small
     python experiments/probes/round3_measurements.py --source amlgentex --dataset sweden_100K_easy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent))

from inchusm_aml.eval import flow_gates as fg  # noqa: E402
from round2_measurements import OUT, load  # noqa: E402

N_BACKGROUND = 200_000
SEED = 20261010


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    print(f"{args.source} {args.dataset}")
    edges, ptx, _ = load(args.source, args.dataset)
    g = fg.conservation_gate(edges, ptx, N_BACKGROUND, SEED)
    g.to_csv(OUT / "round3_conservation.csv", index=False)
    a = g.attrs
    print(f"  background: {a['n_background_transactions']:,} transactions; {N_BACKGROUND:,} pairs from {a['draws']:,} "
          f"draws ({a['rejected_no_incoming']:,} without an incoming transaction, "
          f"{a['rejected_back_and_forth']:,} back-and-forth)")
    print(g.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    pooled = g.set_index("group").loc[fg.ALL_PAIRS_LABEL]
    print(f"    H6 on this set (pooled AUC >= 0.70): {pooled['auc_conservation']:.3f} over "
          f"{int(pooled['n_pairs'])} laundering pairs; holds here: {pooled['auc_conservation'] >= 0.70}")
    for th in fg.THETAS:
        print(f"    theta {th:g}: laundering pairs within {pooled[f'within_{th:g}']:.3f}, background "
              f"{g.set_index('group').loc[fg.BACKGROUND_LABEL, f'within_{th:g}']:.3f}")


if __name__ == "__main__":
    main()
