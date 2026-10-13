"""Rebuild one chain input into a temporary directory and compare it, line by line, with the copy in datasets/chain.

The copy in datasets/chain may have been built on another machine. Equal sha256 means the two machines build
byte-identical inputs; otherwise this prints the line counts and the first differing lines, which show whether the
difference is in account numbering, line order or the printed amounts.

Run: python experiments/probes/compare_chain_input.py --source amlworld --dataset HI-Small --labeling S [--show 5]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "src"))

from inchusm_aml.data.chain_input import build  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def first_differences(a: Path, b: Path, show: int) -> tuple[int, int, list[tuple[int, str, str]]]:
    """Line counts of both files and up to `show` differing lines (1-based number, line in a, line in b)."""
    diffs, na, nb = [], 0, 0
    with open(a, "rb") as fa, open(b, "rb") as fb:
        for n, (la, lb) in enumerate(itertools.zip_longest(fa, fb), 1):
            na += la is not None
            nb += lb is not None
            if la != lb and len(diffs) < show:
                diffs.append((n, repr((la or b"<end of file>").decode(errors="replace")),
                              repr((lb or b"<end of file>").decode(errors="replace"))))
    return na, nb, diffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labeling", choices=["S", "A"], required=True)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()
    existing = ROOT / "datasets" / "chain" / f"{args.dataset}_{args.labeling}.tg"
    if not existing.is_file():
        sys.exit(f"missing: {existing.relative_to(ROOT).as_posix()}")
    with tempfile.TemporaryDirectory() as tmp:
        build(args.source, args.dataset, args.labeling, ROOT, Path(tmp))
        rebuilt = Path(tmp) / existing.name
        sa, sb = sha256(existing), sha256(rebuilt)
        print(f"{args.dataset} {args.labeling}: copy in datasets/chain {sa}, rebuilt here {sb}")
        if sa == sb:
            print("identical: yes")
            return
        na, nb, diffs = first_differences(existing, rebuilt, args.show)
        print(f"identical: no; lines {na} (copy) vs {nb} (rebuilt)")
        for n, la, lb in diffs:
            print(f"  line {n}:\n    copy:    {la}\n    rebuilt: {lb}")


if __name__ == "__main__":
    main()
