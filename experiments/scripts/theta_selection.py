"""The quantity the tolerance theta is chosen by, over a grid of theta.

For one data set and every theta of the grid: the share of planted open chains of length 2 that conserve
under the pairwise rule, and the same share for random background chains of length 2. A planted open chain
of length 2 is one (incoming, outgoing) pair at an intermediate account of a laundering attempt, the origin
of a CYCLE attempt excluded (planted_chains), so its share is exactly the share of the pairs the mining
constraint applies to. The background chains are the walks of chain_planted_rules.py for length 2, same seed.

Selection rule, fixed before mining (fourth round): the smallest theta of the grid whose planted share is at
least 0.90 on both decision sets (HI-Small, sweden_100K_easy). The rule is applied by the paper's generator to
the records, not here. Writes theta_selection.csv.

Run: python experiments/scripts/theta_selection.py --source amlworld --dataset HI-Small
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

THETAS = (0.05, 0.1, 0.2, 0.5)
N_BACKGROUND, SEED = 200_000, 20261012          # as in chain_planted_rules.py, length 2 uses SEED + 2
OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"


def within(amounts: tuple, theta: float) -> bool:
    """One pair (a_in, a_out): written out here rather than through pc.pairwise, so the two can be compared."""
    a_in, a_out = amounts
    return (1 - theta) * a_in <= a_out <= (1 + theta) * a_in


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    edges, ptx, _ = load(args.source, args.dataset)
    planted = pc.planted_chains(ptx, 2)
    pairs = planted[(planted["kind"] == "open") & (planted["length"] == 2)]["amounts"].map(tuple)
    background, discarded = pc.background_chains(edges, 2, N_BACKGROUND, SEED + 2)
    background = pd.Series([tuple(a) for a in background])
    if len(pairs) == 0 or len(background) == 0:
        sys.exit(f"{args.dataset}: {len(pairs)} planted pairs, {len(background)} background pairs")
    rows = []
    for theta in THETAS:
        mine = pairs.map(lambda a: within(a, theta))
        ref = pairs.map(lambda a: pc.pairwise(a, theta))
        if not mine.equals(ref):
            sys.exit(f"theta {theta}: the pair test and pc.pairwise disagree on {int((mine != ref).sum())} pairs")
        rows.append({"dataset": args.dataset, "theta": theta, "planted_pairs": len(pairs),
                     "planted_within": float(mine.mean()), "background_pairs": len(background),
                     "background_within": float(background.map(lambda a: within(a, theta)).mean())})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "theta_selection.csv", index=False)
    print(f"{args.source} {args.dataset}: pairs at intermediate accounts within theta "
          f"(planted: open chains of length 2; background: {len(background)} walks, {discarded} discarded)")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
