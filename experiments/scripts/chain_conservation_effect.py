"""What conservation changes in the mined chains: laundering accounts among their intermediate accounts.

For one data set and one labeling, mines the open chains of the chain input twice with the same
threshold and maximum length, once with conservation (theta = 0.2) and once without (theta = inf),
marking the accounts that have at least one laundering transaction. Closed chains are left out of
both runs: without conservation, deciding whether an account closes a cycle of a given length is a
cycle search over the whole graph, and the measures below concern open chains only. For every open-chain length m >= 2:

  I(m)  accounts at an intermediate position of some reported open chain of length m
  P(m)  share of I(m) that is marked (laundering accounts)
  R(m)  share of the planted-chain intermediate accounts of length m (C(m)) that lie in I(m)
  OR(m) odds of P(m) with conservation divided by the odds without

Writes the two pattern tables, the two account-role tables and conservation_effect.csv into the run
record, and prints the values the hypotheses are decided on.

Run: python experiments/scripts/chain_conservation_effect.py --source amlworld --dataset HI-Small --labeling A
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
THETAS = ("0.2", "inf")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def odds(p: float) -> float:
    return p / (1 - p) if p < 1 else math.inf


def effect_table(roles: dict, marked: set, planted_mid: dict, max_length: int) -> pd.DataFrame:
    """One row per open-chain length: |I|, marked accounts in I, P and R for each theta, and OR."""
    rows = []
    for m in range(2, max_length + 1):
        row = {"length": m, "planted_intermediates": len(planted_mid.get(m, ()))}
        for theta in THETAS:
            inter = set(roles[theta].loc[roles[theta]["length"] == m, "account"].tolist())
            hit = len(inter & marked)
            row[f"I_{theta}"] = len(inter)
            row[f"marked_{theta}"] = hit
            row[f"P_{theta}"] = hit / len(inter) if inter else float("nan")
            c = planted_mid.get(m, set())
            row[f"R_{theta}"] = len(inter & c) / len(c) if c else float("nan")
        p1, p0 = row["P_0.2"], row["P_inf"]
        row["OR"] = odds(p1) / odds(p0) if row["I_0.2"] and row["I_inf"] and 0 < p0 < 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def mine(graph: Path, marked: Path, theta: str, tau: int, length: int, heap: str) -> tuple[Path, Path]:
    patterns = OUT / f"patterns_theta_{theta}.tsv"
    roles = OUT / f"accounts_theta_{theta}.tsv"
    cmd = ["java", f"-Xmx{heap}", "-cp", str(JAR), "edu.huflit.inchusm.chain.ChainMain", "--graph", str(graph),
           "--theta", theta, "--min-support", str(tau), "--max-length", str(length), "--mark", str(marked),
           "--no-closed", "--out", str(patterns), "--accounts-out", str(roles)]
    print(f"\n[miner, theta={theta}]", flush=True)
    subprocess.run(cmd, check=True)
    return patterns, roles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labeling", choices=["S", "A"], required=True)
    ap.add_argument("--tau", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=5)
    ap.add_argument("--heap", default="16g")
    args = ap.parse_args()

    graph = CHAIN / f"{args.dataset}_{args.labeling}.tg"
    marked_file = CHAIN / f"{args.dataset}_laundering_accounts.txt"
    info = json.loads((CHAIN / f"{args.dataset}_{args.labeling}.json").read_text(encoding="utf-8"))
    digest = sha256(graph)
    if digest != info["graph_sha256"]:
        sys.exit(f"{graph.name}: sha256 {digest} differs from the export record {info['graph_sha256']}; re-export it")
    print(f"input {graph.name} sha256 {digest} ({info['transactions']} transactions, {info['accounts']} accounts)")

    marked = set(np.loadtxt(marked_file, dtype=np.int64, ndmin=1).tolist())
    edges, ptx, _ = load(args.source, args.dataset)
    # the ground truth read here must name accounts exactly as the exported graph does
    e = edges[edges["src"] != edges["dst"]]
    flagged = edges["laundering"].to_numpy() == 1
    again = set(np.union1d(edges["src"].to_numpy()[flagged], edges["dst"].to_numpy()[flagged]).tolist())
    again &= set(np.union1d(e["src"].to_numpy(), e["dst"].to_numpy()).tolist())
    if again != marked:
        sys.exit(f"account ids differ between the export and this loader: {len(again ^ marked)} laundering accounts "
                 f"in one set only ({len(again)} here, {len(marked)} in {marked_file.name})")
    print(f"ground truth and export agree on all {len(marked)} laundering accounts")
    del edges, e
    planted = pc.planted_chains(ptx, args.max_length)
    planted = planted[planted["kind"] == "open"]
    planted_mid = {m: set(a for acc in g["accounts"] for a in acc[1:-1]) for m, g in planted.groupby("length")}

    roles = {}
    for theta in THETAS:
        _, role_file = mine(graph, marked_file, theta, args.tau, args.max_length, args.heap)
        r = pd.read_csv(role_file, sep="\t")
        roles[theta] = r[(r["kind"] == "open") & (r["role"] == "intermediate")]

    table = effect_table(roles, marked, planted_mid, args.max_length)
    table.insert(0, "labeling", args.labeling)
    table.insert(0, "dataset", args.dataset)
    table.to_csv(OUT / "conservation_effect.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\n{args.dataset} labeling {args.labeling}: tau={args.tau}, max length {args.max_length}, "
          f"{len(marked)} marked accounts")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for m in (2, 3, 4):
        r = table[table["length"] == m].iloc[0]
        h11 = bool(r["OR"] >= 2) if not math.isnan(r["OR"]) else None
        h12 = (bool(r["R_0.2"] >= 0.5 * r["R_inf"]) if not (math.isnan(r["R_0.2"]) or math.isnan(r["R_inf"]))
               else None)
        print(f"  length {m}: H11 (OR >= 2): {h11}; H12 (R with conservation >= half of R without): {h12}"
              + ("" if r["planted_intermediates"] >= 30 else f"  [only {r['planted_intermediates']} planted "
                 "intermediate accounts: not counted]"))


if __name__ == "__main__":
    main()
