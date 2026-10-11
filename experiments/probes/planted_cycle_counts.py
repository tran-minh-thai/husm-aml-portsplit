"""Accounts of planted AMLworld cycles by number of transactions, read from a Patterns.txt file alone.

Uses the same chain construction as the experiments (planted_chains), with accounts identified by
bank and account number instead of the integer codes of the transaction file, so it needs only the
pattern file. A closed chain counts when its number of transactions is at most --max-length.

Run: python experiments/probes/planted_cycle_counts.py PATH/HI-Large_Patterns.txt [more files]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "src"))

from inchusm_aml.eval import planted_chains as pc  # noqa: E402


def read_patterns(path: Path) -> pd.DataFrame:
    rows, attempt, typ = [], -1, None
    for line in path.read_text().splitlines():
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            attempt += 1
            typ = line.split(" - ", 1)[1].split(":")[0].strip()
        elif line and not line.startswith("END"):
            tok = line.split(",")
            rows.append((attempt, typ, f"{tok[1]}_{tok[2]}", f"{tok[3]}_{tok[4]}",
                         pd.Timestamp(tok[0].replace("/", "-")), float(tok[7])))
    return pd.DataFrame(rows, columns=["attempt", "type", "src", "dst", "time", "amount"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--max-length", type=int, default=5)
    args = ap.parse_args()
    out = []
    for f in args.files:
        ptx = read_patterns(Path(f))
        planted = pc.planted_chains(ptx, args.max_length)
        closed = planted[planted["kind"] == "closed"]
        row = {"file": Path(f).name, "attempts": ptx["attempt"].nunique(),
               "cycle_attempts": ptx.loc[ptx["type"] == "CYCLE", "attempt"].nunique()}
        for m in range(2, args.max_length + 1):
            g = closed[closed["length"] == m]
            row[f"cycle_accounts_{m}"] = len(set(x for acc in g["accounts"] for x in acc))
        out.append(row)
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
