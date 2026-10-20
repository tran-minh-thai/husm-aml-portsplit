"""Compare CycleMembership with the chain miner on recorded closed-chain runs at theta = inf.

For every given record of chain_cycle_effect.py (run with the miner at theta = inf), runs CycleMembership on
the same graph with the same threshold and lengths and compares, row by row, the closed-chain rows of the two
account files (kind, length, role, account). Exits non-zero on any difference, when a record is not a miner
run at theta = inf, or when no row was compared.

Run: python experiments/scripts/cycle_membership_check.py results-invariant/chain-cycle-effect-final/<run> ...
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chain_cycle_effect as cce  # noqa: E402

OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"


def closed_rows(path: Path) -> set[tuple[str, str, str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as h:
        return {(r["kind"], r["length"], r["role"], r["account"]) for r in csv.DictReader(h, delimiter="\t")
                if r["kind"] == "closed"}


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("give one or more record directories of chain_cycle_effect.py")
    OUT.mkdir(parents=True, exist_ok=True)
    compared, failures = 0, []
    for rec in map(Path, sys.argv[1:]):
        meta = json.loads((rec / "run_meta.json").read_text(encoding="utf-8"))
        argv = meta["command"]
        if "chain_cycle_effect.py" not in " ".join(argv) or "--inf-engine" in argv:
            failures.append(f"{rec.name}: not a chain_cycle_effect.py run with the miner at theta = inf")
            continue
        dataset = argv[argv.index("--dataset") + 1]
        miner = rec / "accounts_theta_inf.tsv.gz"
        if not miner.is_file():
            miner = rec / "accounts_theta_inf.tsv"
        ours = OUT / f"cycle_accounts_{dataset}.tsv"
        subprocess.run(["java", "-Xmx24g", "-cp", str(cce.JAR), "edu.huflit.inchusm.cycles.CycleMembership",
                        "--graph", str(cce.CHAIN / f"{dataset}_S.tg"), "--min-support", str(cce.TAU),
                        "--min-length", str(cce.MIN_CYCLE), "--max-length", str(cce.MAX_LENGTH), "--out", str(ours)],
                       check=True)
        a, b = closed_rows(ours), closed_rows(miner)
        by_length = {}
        for kind, length, role, _ in b:
            by_length[length] = by_length.get(length, 0) + 1
        print(f"{dataset} (record {rec.name}, miner commit {meta.get('git_commit', '')[:7]}): miner rows by length "
              f"{dict(sorted(by_length.items()))}; cycle membership {len(a)} rows; only cycle membership {len(a - b)}, "
              f"only miner {len(b - a)}", flush=True)
        compared += len(a | b)
        if a != b:
            failures.append(f"{dataset}: {len(a - b)} rows only from cycle membership, {len(b - a)} only from the miner")
        ours.unlink()
    print(f"rows compared: {compared}; records: {len(sys.argv) - 1}; differences: {len(failures)}")
    if compared == 0:
        sys.exit("no row compared")
    if failures:
        sys.exit("\n".join(failures))


if __name__ == "__main__":
    main()
