"""Choose theta for the chain-node conservation constraint, as the fourth round pre-registered.

For one set, the share of chain-node pairs (pooled over typologies) whose ratio a_out / a_in lies in
[1 - theta, 1 + theta], for each theta of the grid, and per typology; writes round4_theta.csv. The
rule picks the smallest theta reaching 0.90 on both decision sets; the choice is made from the two
records, not here.

Run: python experiments/probes/round4_theta.py --source amlworld --dataset HI-Small
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    _, ptx, _ = load(args.source, args.dataset)
    cp = fg.chain_pairs(ptx)
    cp = cp[(cp["a1"] > 0) & (cp["a2"] > 0)]
    r = cp["a2"] / cp["a1"]
    rows = []
    for label, sel in [("ALL chain pairs", slice(None))] + [(t, cp["type"] == t) for t in sorted(cp["type"].unique())]:
        rr = r[sel]
        row = {"group": label, "n_pairs": len(rr)}
        for th in fg.THETAS:
            row[f"within_{th:g}"] = float(((rr >= 1 - th) & (rr <= 1 + th)).mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "round4_theta.csv", index=False)
    print(f"{args.source} {args.dataset}: chain-node pairs within theta")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
