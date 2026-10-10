"""Fourth-round gate measurement: conservation by the role an account plays inside an attempt.

Pre-registered before any value was seen; decision sets and confirmation sets run from one commit.
For one set, writes round4_chain.csv (pair conservation at chain nodes, against 200,000 sampled
background pairs, seed 20261010) and round4_split_merge.csv (total conservation at split/merge
nodes, against 10 background replicates per node with the same numbers of incoming and outgoing
transactions, seed 20261011), and prints both hypotheses with the values they are decided on.

Run: python experiments/probes/round4_measurements.py --source amlworld --dataset HI-Small
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

N_BACKGROUND, SEED_PAIRS = 200_000, 20261010
REPLICATES, SEED_MATCHED = 10, 20261011
MIN_UNITS = 30


def verdict(tab: pd.DataFrame, n_col: str, auc_col: str, label: str) -> None:
    ok_rows = tab[tab[n_col] >= MIN_UNITS]
    small = tab[tab[n_col] < MIN_UNITS]
    parts = [f"{t} {a:.3f} ({int(n)})" for t, a, n in zip(ok_rows["type"], ok_rows[auc_col], ok_rows[n_col])]
    holds = bool(len(ok_rows)) and bool((ok_rows[auc_col] >= 0.70).all())
    print(f"    {label} on this set (every typology with >= {MIN_UNITS} units at AUC >= 0.70): "
          f"{'; '.join(parts) or 'no typology with enough units'}"
          + (f"; not counted: {', '.join(f'{t} ({int(n)})' for t, n in zip(small['type'], small[n_col]))}" if len(small) else "")
          + f"; holds here: {holds}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    print(f"{args.source} {args.dataset}")
    edges, ptx, _ = load(args.source, args.dataset)
    print(f"  {len(edges):,} transactions; {ptx['attempt'].nunique()} attempts; typologies "
          f"{sorted(ptx['type'].unique())}")
    tab_i, tab_ii = fg.role_gate(edges, ptx, N_BACKGROUND, SEED_PAIRS, REPLICATES, SEED_MATCHED)
    tab_i.to_csv(OUT / "round4_chain.csv", index=False)
    tab_ii.to_csv(OUT / "round4_split_merge.csv", index=False)
    print(f"\n[i] chain nodes (background accounts with one background transaction in and one out: "
          f"{tab_i.attrs['n_one_in_one_out_background_accounts']:,})")
    print(tab_i.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    verdict(tab_i, "n_chain_pairs", "auc_i", "H7")
    print(f"\n[ii] split/merge nodes ({tab_ii.attrs['unmatched_nodes']} nodes without an eligible background account)")
    print(tab_ii.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    verdict(tab_ii, "n_split_merge_nodes", "auc_ii", "H8")


if __name__ == "__main__":
    main()
