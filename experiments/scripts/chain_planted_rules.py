"""Which amount rule keeps the planted laundering chains, and how many background chains it keeps.

For one data set: every planted open chain of length 2..5 and every planted closed chain (a CYCLE
attempt read from its origin), and 200,000 random background open chains per length; the share
valid under the pairwise rule (theta = 0.2), the rule anchored on the first transaction
(alpha = 0.2), by kind, length and typology. Writes planted_rules.csv and background_rules.csv.

Run: python experiments/scripts/chain_planted_rules.py --source amlgentex --dataset sweden_100K_easy
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

THETA = ALPHA = 0.2
MAX_LENGTH, CLOSED_MAX = 5, 64
N_BACKGROUND, SEED = 200_000, 20261012
OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"


def shares(chains: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    c = chains.assign(pairwise=chains["amounts"].map(lambda a: pc.pairwise(a, THETA)),
                      anchored=chains["amounts"].map(lambda a: pc.anchored(a, ALPHA)))
    g = c.groupby(by)
    return g.agg(n=("pairwise", "size"), pairwise=("pairwise", "mean"), anchored=("anchored", "mean")).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    edges, ptx, _ = load(args.source, args.dataset)
    planted = pc.planted_chains(ptx, CLOSED_MAX)
    planted = planted[(planted["kind"] == "closed") | (planted["length"] <= MAX_LENGTH)]
    pooled = shares(planted, ["kind", "length"]).assign(type="ALL")
    by_type = shares(planted, ["kind", "length", "type"])
    table = pd.concat([pooled, by_type], ignore_index=True)
    table.to_csv(OUT / "planted_rules.csv", index=False)

    rows = []
    for m in range(2, MAX_LENGTH + 1):
        chains, discarded = pc.background_chains(edges, m, N_BACKGROUND, SEED + m)
        bg = pd.DataFrame({"amounts": chains})
        rows.append({"length": m, "n": len(bg), "discarded_walks": discarded,
                     "pairwise": float(bg["amounts"].map(lambda a: pc.pairwise(a, THETA)).mean()),
                     "anchored": float(bg["amounts"].map(lambda a: pc.anchored(a, ALPHA)).mean())})
    background = pd.DataFrame(rows)
    background.to_csv(OUT / "background_rules.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"{args.source} {args.dataset}: {planted['attempt'].nunique()} attempts with a planted chain, "
          f"{len(planted)} planted chains (open up to length {MAX_LENGTH}, closed of any length)")
    print("\nplanted chains, pooled over typologies (share valid under each rule):")
    print(pooled.drop(columns="type").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nplanted chains by typology:")
    print(by_type.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nbackground open chains ({N_BACKGROUND} per length, seed {SEED} + length):")
    print(background.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
