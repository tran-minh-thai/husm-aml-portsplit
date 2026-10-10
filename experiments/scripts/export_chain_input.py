"""Write the chain miner's input for one data set and one labeling (S or A).

Output (under datasets/chain/, not committed): <dataset>_<labeling>.tg, <dataset>_<labeling>.json
(counts and label names), <dataset>_laundering_accounts.txt (accounts with a flagged transaction,
used only to count them in the miner's output).

Run: python experiments/scripts/export_chain_input.py --source amlworld --dataset HI-Small --labeling A
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "src"))

from inchusm_aml.data.chain_input import LABELINGS, build  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labeling", choices=LABELINGS, required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "datasets" / "chain"))
    args = ap.parse_args()
    info = build(args.source, args.dataset, args.labeling, ROOT, Path(args.out_dir))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
