"""What pairwise conservation changes for closed chains (rooted cycles), labeling S.

The miner runs with closed chains of 3 to 5 transactions (the pair at the origin is not
constrained), at theta = 0.2 and with no amount constraint (theta = inf). With --inf-engine cycles the
sets at theta = inf come instead from CycleMembership, which computes them directly: with one account
label and one transaction label and no amount constraint, the accounts of the reported closed chains of
length m are the accounts on a simple cycle of m transactions, when there are at least tau of them
(cycle_membership_check.py compares the two on recorded miner runs). For a closed-chain length m:

  I_c(m)  accounts at any position (origin or intermediate) of a reported closed chain of length m
  C_c(m)  accounts of the planted cycles (typology CYCLE) with exactly m transactions
  P(m)    |I_c(m) & M| / |I_c(m)|, M the laundering accounts
  R(m)    |I_c(m) & C_c(m)| / |C_c(m)|
  OR(m)   odds of P at theta 0.2 over odds of P at theta inf

Run: python experiments/scripts/chain_cycle_effect.py --source amlworld --dataset HI-Small [--inf-engine cycles]
"""
from __future__ import annotations

import argparse
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
TAU, MAX_LENGTH, MIN_CYCLE, LENGTHS, MIN_PLANTED = 10, 5, 3, (3, 4, 5), 30


def closed_sets(path: Path) -> dict[int, set]:
    """Accounts at any position of reported closed chains, by length."""
    r = pd.read_csv(path, sep="\t")
    r = r[r["kind"] == "closed"]
    return {m: set(g["account"].tolist()) for m, g in r.groupby("length")}


def planted_cycle_accounts(planted: pd.DataFrame) -> dict[int, set]:
    c = planted[planted["kind"] == "closed"]
    return {m: set(x for acc in g["accounts"] for x in acc) for m, g in c.groupby("length")}


def odds(p: float) -> float:
    return p / (1 - p) if p < 1 else float("inf")


def effect(sets: dict, marked: set, planted: dict) -> pd.DataFrame:
    rows = []
    for m in LENGTHS:
        c = planted.get(m, set())
        row = {"length": m, "planted_accounts": len(c)}
        for theta in ("0.2", "inf"):
            s = sets[theta].get(m, set())
            row[f"I_{theta}"] = len(s)
            row[f"P_{theta}"] = len(s & marked) / len(s) if s else float("nan")
            row[f"R_{theta}"] = len(s & c) / len(c) if c else float("nan")
        row["OR"] = (odds(row["P_0.2"]) / odds(row["P_inf"])
                     if row["I_0.2"] and row["I_inf"] and 0 < row["P_inf"] < 1 else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def cycles(dataset: str, heap: str) -> Path:
    """The theta = inf sets of closed chains from CycleMembership (labeling S only)."""
    roles = OUT / "accounts_theta_inf.tsv"
    cmd = ["java", f"-Xmx{heap}", "-cp", str(JAR), "edu.huflit.inchusm.cycles.CycleMembership",
           "--graph", str(CHAIN / f"{dataset}_S.tg"), "--min-support", str(TAU), "--min-length", str(MIN_CYCLE),
           "--max-length", str(MAX_LENGTH), "--out", str(roles)]
    print("\n[cycle membership, theta=inf]", flush=True)
    subprocess.run(cmd, check=True)
    return roles


def mine(dataset: str, theta: str, heap: str) -> Path:
    roles = OUT / f"accounts_theta_{theta}.tsv"
    cmd = ["java", f"-Xmx{heap}", "-cp", str(JAR), "edu.huflit.inchusm.chain.ChainMain",
           "--graph", str(CHAIN / f"{dataset}_S.tg"), "--theta", theta, "--min-support", str(TAU),
           "--max-length", str(MAX_LENGTH), "--min-cycle", str(MIN_CYCLE),
           "--mark", str(CHAIN / f"{dataset}_laundering_accounts.txt"),
           "--out", str(OUT / f"patterns_theta_{theta}.tsv"), "--accounts-out", str(roles)]
    print(f"\n[miner with closed chains, theta={theta}]", flush=True)
    subprocess.run(cmd, check=True)
    return roles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--heap", default="16g")
    ap.add_argument("--inf-engine", choices=["miner", "cycles"], default="miner",
                    help="what computes the closed chains at theta = inf (same sets)")
    args = ap.parse_args()
    marked = set(np.loadtxt(CHAIN / f"{args.dataset}_laundering_accounts.txt", dtype=np.int64, ndmin=1).tolist())
    _, ptx, _ = load(args.source, args.dataset)
    planted = planted_cycle_accounts(pc.planted_chains(ptx, MAX_LENGTH))
    sets = {"0.2": closed_sets(mine(args.dataset, "0.2", args.heap)),
            "inf": closed_sets(cycles(args.dataset, args.heap) if args.inf_engine == "cycles"
                               else mine(args.dataset, "inf", args.heap))}
    table = effect(sets, marked, planted)
    table.insert(0, "dataset", args.dataset)
    table.to_csv(OUT / "cycle_effect.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\n{args.dataset}, closed chains, labeling S, tau {TAU}, {len(marked)} laundering accounts; "
          f"theta = inf by {args.inf_engine}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for _, r in table.iterrows():
        if r["planted_accounts"] >= MIN_PLANTED:
            print(f"  length {int(r['length'])}: H31 OR >= 2: {r['OR'] >= 2}; R(0.2) >= half R(inf): "
                  f"{r['R_0.2'] >= 0.5 * r['R_inf']}")
        else:
            print(f"  length {int(r['length'])}: {int(r['planted_accounts'])} planted cycle accounts - not counted")


if __name__ == "__main__":
    main()
