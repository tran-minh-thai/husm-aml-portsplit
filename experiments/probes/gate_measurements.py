"""Three gate measurements on one AMLworld set, as pre-registered before any value was seen.

  1. amount   do laundering-attempt amounts differ from background amounts?
  2. order    does money enter a pass-through account before it leaves?
  3. reduce   does an amount threshold keep more intact attempts than a degree cap, at no larger size?

Amounts are in US Dollar. Self-transfers are dropped. Writes gate1_amount.csv, gate2_order.csv,
gate2_passthrough_accounts.csv and gate3_reduction.csv to the record directory (or
experiments/results when run outside the recording wrapper), and prints each hypothesis with the
values it is decided on. The decision itself combines two sets and is not made here.

Run: python experiments/probes/gate_measurements.py --dataset HI-Small
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

from e9_portsplit_recovery import DATA, apply_hub_cap  # noqa: E402
from inchusm_aml.data.amlworld import load_pattern_transactions, load_transactions  # noqa: E402
from inchusm_aml.eval import flow_gates as fg  # noqa: E402

OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else HERE.parents[1] / "results"
CAPS = (50, 200)
REMOVAL = (0.50, 0.75, 0.90, 0.95, 0.99)
H2_TYPES = ("GATHER-SCATTER", "SCATTER-GATHER", "STACK", "RANDOM")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HI-Small")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    edges, code = load_transactions(DATA / f"{args.dataset}_Trans.csv")
    ptx = load_pattern_transactions(DATA / f"{args.dataset}_Patterns.txt", code, edges.attrs["usd_rates"])
    n_self = int((ptx["src"] == ptx["dst"]).sum())
    n_unknown = int(((ptx["src"] < 0) | (ptx["dst"] < 0)).sum())
    print(f"{args.dataset}: {len(edges):,} transactions ({int((edges['src'] == edges['dst']).sum()):,} "
          f"self-transfers dropped), {ptx['attempt'].nunique()} attempts with {len(ptx):,} transactions "
          f"({n_self} self-transfers dropped, {n_unknown} with an account absent from the transactions)")

    # ---------------------------------------------------------------- 1. amount
    g1 = fg.amount_gate(edges, ptx)
    g1.to_csv(OUT / "gate1_amount.csv", index=False)
    print(f"\n[1] amounts vs {g1.attrs['n_background']:,} background transactions")
    print(g1.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    typed = g1[g1["type"] != fg.ALL_LAUNDERING]
    strong = typed[(typed["auc_vs_background"] - 0.5).abs() >= 0.10]
    print(f"    H1 on this set: {len(strong)} of {len(typed)} typologies with |AUC - 0.5| >= 0.10 "
          f"({', '.join(f'{t} {a:.3f}' for t, a in zip(strong['type'], strong['auc_vs_background'])) or 'none'}); "
          f"holds here: {len(strong) >= 4}")
    below = typed[typed["auc_vs_background"] < 0.5]["type"].tolist()
    print(f"    typologies whose amounts sit BELOW background (AUC < 0.5): {below or 'none'}")

    # ----------------------------------------------------------------- 2. order
    acc = fg.passthrough_accounts(ptx)
    acc.to_csv(OUT / "gate2_passthrough_accounts.csv", index=False)
    g2 = fg.order_gate(ptx)
    g2.to_csv(OUT / "gate2_order.csv", index=False)
    print("\n[2] order at pass-through accounts (times in minutes)")
    print(g2.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    h2 = g2[g2["type"].isin(H2_TYPES)].set_index("type")
    verdicts = []
    for t in H2_TYPES:
        if t not in h2.index or h2.loc[t, "n_with_passthrough"] < 10:
            verdicts.append(f"{t}: denominator {0 if t not in h2.index else int(h2.loc[t, 'n_with_passthrough'])} < 10, not counted")
            continue
        f = h2.loc[t, "frac_attempts_all_in_before_out_le"]
        verdicts.append(f"{t}: {f:.3f} of {int(h2.loc[t, 'n_with_passthrough'])}")
    counted = [t for t in H2_TYPES if t in h2.index and h2.loc[t, "n_with_passthrough"] >= 10]
    holds = bool(counted) and all(h2.loc[t, "frac_attempts_all_in_before_out_le"] >= 0.90 for t in counted)
    print("    H2 on this set (all-in-before-out, ties allowed, >= 0.90 each): " + "; ".join(verdicts)
          + f"; holds here: {holds}")

    # ---------------------------------------------------------------- 3. reduce
    g3 = fg.reduction_gate(edges, ptx, CAPS, REMOVAL, apply_hub_cap)
    g3.to_csv(OUT / "gate3_reduction.csv", index=False)
    print("\n[3] graph reductions on the whole graph")
    cols = ["reduction", "parameter", "removed_fraction", "n_transactions", "n_accounts", "max_degree",
            "n_attempts", "frac_attempts_intact", "frac_laundering_transactions_kept"]
    print(g3[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    per_type = [c for c in g3.columns if c.startswith("intact_")]
    print(g3[["reduction", "parameter"] + per_type].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    cap50 = g3[(g3["reduction"] == "degree_cap") & (g3["parameter"] == 50)].iloc[0]
    ok = g3[(g3["reduction"] == "amount_threshold") & (g3["n_transactions"] <= cap50["n_transactions"])
            & (g3["frac_attempts_intact"] >= cap50["frac_attempts_intact"])]
    print(f"    degree cap 50: {int(cap50['n_transactions']):,} transactions, "
          f"{cap50['frac_attempts_intact']:.3f} of {int(cap50['n_attempts'])} attempts intact")
    print(f"    H3 on this set: thresholds at most that size with at least that many intact attempts: "
          f"{len(ok)} ({', '.join(f'tau {p:.4g}' for p in ok['parameter']) or 'none'}); holds here: {len(ok) > 0}")


if __name__ == "__main__":
    main()
