"""The conservation effect at one theta, against the no-conservation run already recorded.

Mines the open chains of the structure-only input (labeling S, tau = 10, length <= 5) at the given
theta and computes, for every length m, OR(m) = odds(P with theta) / odds(P without conservation),
where P is the share of laundering accounts among the intermediate accounts of the reported chains,
and R(m), the share of planted-chain intermediate accounts covered. The run without conservation
does not depend on theta, so it is read from the conservation-effect record given, not mined again.

Run: python experiments/scripts/chain_theta_grid.py --source amlworld --dataset HI-Small --theta 0.1 \\
         --effect-record results-invariant/chain-conservation-effect/<run of HI-Small, labeling S>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "probes"))

from inchusm_aml.eval import planted_chains as pc  # noqa: E402
from round2_measurements import load  # noqa: E402

JAR = ROOT / "inchusm-miner" / "target" / "inchusm-miner-0.1.0-SNAPSHOT.jar"
CHAIN = ROOT / "datasets" / "chain"
OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"
TAU, MAX_LENGTH = 10, 5


def account_file(record: Path, theta: str) -> Path:
    """The account-role file of a run, as written or as stored gzip-compressed."""
    plain = record / f"accounts_theta_{theta}.tsv"
    return plain if plain.is_file() else plain.with_name(plain.name + ".gz")


def odds(p: float) -> float:
    return p / (1 - p) if 0 <= p < 1 else float("inf")


def intermediates(path: Path) -> dict[int, set]:
    r = pd.read_csv(path, sep="\t")
    r = r[(r["kind"] == "open") & (r["role"] == "intermediate")]
    return {m: set(g["account"].tolist()) for m, g in r.groupby("length")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--theta", required=True)
    ap.add_argument("--effect-record", required=True)
    ap.add_argument("--heap", default="16g")
    args = ap.parse_args()
    rec = ROOT / args.effect_record
    cmd = " ".join(json.loads((rec / "run_meta.json").read_text())["command"])
    if f"--dataset {args.dataset}" not in cmd or "--labeling S" not in cmd:
        sys.exit(f"{rec}: not the conservation-effect run of {args.dataset} with labeling S")
    without = intermediates(account_file(rec, "inf"))

    roles = OUT / f"accounts_theta_{args.theta}.tsv"
    subprocess.run(["java", f"-Xmx{args.heap}", "-cp", str(JAR), "edu.huflit.inchusm.chain.ChainMain", "--graph",
                    str(CHAIN / f"{args.dataset}_S.tg"), "--theta", args.theta, "--min-support", str(TAU),
                    "--max-length", str(MAX_LENGTH), "--no-closed", "--out", str(OUT / f"patterns_theta_{args.theta}.tsv"),
                    "--accounts-out", str(roles)], check=True)
    with_c = intermediates(roles)

    edges, ptx, _ = load(args.source, args.dataset)
    flagged = edges["laundering"].to_numpy() == 1
    marked = set(np.union1d(edges["src"].to_numpy()[flagged], edges["dst"].to_numpy()[flagged]).tolist())
    planted = pc.planted_chains(ptx, MAX_LENGTH)
    planted = planted[planted["kind"] == "open"]
    mid = {m: set(a for acc in g["accounts"] for a in acc[1:-1]) for m, g in planted.groupby("length")}
    rows = []
    for m in range(2, MAX_LENGTH + 1):
        s1, s0, c = with_c.get(m, set()), without.get(m, set()), mid.get(m, set())
        p1 = len(s1 & marked) / len(s1) if s1 else float("nan")
        p0 = len(s0 & marked) / len(s0) if s0 else float("nan")
        rows.append({"dataset": args.dataset, "theta": float(args.theta), "length": m, "I_theta": len(s1),
                     "I_inf": len(s0), "P_theta": p1, "P_inf": p0,
                     "OR": odds(p1) / odds(p0) if s1 and s0 and 0 < p0 < 1 else float("nan"),
                     "R_theta": len(s1 & c) / len(c) if c else float("nan"), "planted_intermediates": len(c)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "theta_grid.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
