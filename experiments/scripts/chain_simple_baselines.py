"""Conserved chains against simple rules on one task: return the accounts that forward laundering money.

For one data set (structure-only labeling S): the set of accounts each method returns, scored
against C (intermediate accounts of planted open chains of length 2..5) and against M (accounts
with a laundering transaction):

  ours(m)     intermediate accounts of the open chains of length m mined with conservation
              (theta = 0.2, tau = 10), read from the conservation-effect record given with --effect-record
  anchored(m) the same with the rule of the transaction-cycle detectors (every amount within 0.2 of
              the first one), computed by the enumeration engine of the chain miner
  one-in-one-out  accounts with exactly one incoming and one outgoing transaction in the whole graph
  totals      accounts whose total out over total in lies in [0.8, 1.2]

Uncertainty: bootstrap over laundering attempts (1,000 resamples, seed 20261015); the accounts of
an attempt travel with it; the false positives of a method (returned accounts outside the target)
are kept fixed.

Run: python experiments/scripts/chain_simple_baselines.py --source amlworld --dataset HI-Small \\
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
THETA, TAU, MAX_LENGTH, LENGTHS = 0.2, 10, 5, (2, 3, 4, 5)
BOOT, SEED = 1000, 20261015


def account_file(record: Path, theta: str) -> Path:
    """The account-role file of a run, as written or as stored gzip-compressed."""
    plain = record / f"accounts_theta_{theta}.tsv"
    return plain if plain.is_file() else plain.with_name(plain.name + ".gz")


def roles_file_sets(path: Path) -> dict[int, set]:
    r = pd.read_csv(path, sep="\t")
    r = r[(r["kind"] == "open") & (r["role"] == "intermediate")]
    return {m: set(g["account"].tolist()) for m, g in r.groupby("length")}


def anchored_sets(dataset: str, heap: str) -> dict[int, set]:
    roles = OUT / "accounts_anchored.tsv"
    cmd = ["java", f"-Xmx{heap}", "-cp", str(JAR), "edu.huflit.inchusm.chain.ChainMain", "--engine", "enumerate",
           "--rule", "anchored", "--graph", str(CHAIN / f"{dataset}_S.tg"), "--theta", str(THETA), "--min-support",
           str(TAU), "--max-length", str(MAX_LENGTH), "--no-closed", "--out", str(OUT / "patterns_anchored.tsv"),
           "--accounts-out", str(roles)]
    print("[enumerator, anchored rule]", flush=True)
    subprocess.run(cmd, check=True)
    return roles_file_sets(roles)


def simple_rules(edges: pd.DataFrame) -> tuple[set, set]:
    e = edges[edges["src"] != edges["dst"]]
    indeg = e.groupby("dst").size()
    outdeg = e.groupby("src").size()
    both = indeg.index.intersection(outdeg.index)
    one_one = set(both[(indeg[both] == 1).to_numpy() & (outdeg[both] == 1).to_numpy()].tolist())
    sin = e.groupby("dst")["amount"].sum()
    sout = e.groupby("src")["amount"].sum()
    ratio = sout[both] / sin[both]
    totals = set(both[((ratio >= 1 - THETA) & (ratio <= 1 + THETA)).to_numpy()].tolist())
    return one_one, totals


def scores(returned: set, target_by_attempt: list[set], rng: np.random.Generator) -> dict:
    target = set().union(*target_by_attempt) if target_by_attempt else set()
    tp = len(returned & target)
    p = tp / len(returned) if returned else float("nan")
    r = tp / len(target) if target else float("nan")
    f1 = 2 * p * r / (p + r) if returned and target and (p + r) > 0 else 0.0
    fp = len(returned - target)
    hits = np.array([len(returned & a) for a in target_by_attempt], dtype=float)
    sizes = np.array([len(a) for a in target_by_attempt], dtype=float)
    boot = []
    for _ in range(BOOT):
        idx = rng.integers(0, len(sizes), len(sizes))
        tpb, nb = hits[idx].sum(), sizes[idx].sum()
        pb = tpb / (tpb + fp) if tpb + fp > 0 else 0.0
        rb = tpb / nb if nb > 0 else 0.0
        boot.append((pb, rb, 2 * pb * rb / (pb + rb) if pb + rb > 0 else 0.0))
    lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
    return {"returned": len(returned), "tp": tp, "precision": p, "recall": r, "f1": f1,
            "precision_ci": f"[{lo[0]:.3f}, {hi[0]:.3f}]", "recall_ci": f"[{lo[1]:.3f}, {hi[1]:.3f}]",
            "f1_ci": f"[{lo[2]:.3f}, {hi[2]:.3f}]"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--effect-record", required=True, help="conservation-effect run of this data set, labeling S")
    ap.add_argument("--heap", default="16g")
    args = ap.parse_args()
    rec = ROOT / args.effect_record
    meta = json.loads((rec / "run_meta.json").read_text())
    if f"--dataset {args.dataset}" not in " ".join(meta["command"]) or "--labeling S" not in " ".join(meta["command"]):
        sys.exit(f"{rec}: not the conservation-effect run of {args.dataset} with labeling S")
    ours = roles_file_sets(account_file(rec, "0.2"))

    edges, ptx, _ = load(args.source, args.dataset)
    planted = pc.planted_chains(ptx, MAX_LENGTH)
    planted = planted[planted["kind"] == "open"]
    c_by_attempt = [set(a for acc in g["accounts"] for a in acc[1:-1]) for _, g in planted.groupby("attempt")]
    flagged = edges["laundering"].to_numpy() == 1
    m_all = set(np.union1d(edges["src"].to_numpy()[flagged], edges["dst"].to_numpy()[flagged]).tolist())
    m_by_attempt = [set(g["src"]) | set(g["dst"]) for _, g in ptx.groupby("attempt")]
    outside = m_all - set().union(*m_by_attempt)
    if outside:
        m_by_attempt.append(outside)          # flagged accounts in no attempt form one fixed cluster
    one_one, totals = simple_rules(edges)
    anchored = anchored_sets(args.dataset, args.heap)

    methods = {f"ours({m})": ours.get(m, set()) for m in LENGTHS}
    methods.update({f"anchored({m})": anchored.get(m, set()) for m in LENGTHS})
    methods["one-in-one-out"] = one_one
    methods["totals"] = totals
    rows = []
    for target, groups in (("C", c_by_attempt), ("M", m_by_attempt)):
        rng = np.random.default_rng(SEED)
        for name, s in methods.items():
            rows.append({"dataset": args.dataset, "target": target, "method": name, **scores(s, groups, rng)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "simple_baselines.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\n{args.dataset}: |C| = {len(set().union(*c_by_attempt))} over {len(c_by_attempt)} attempts, |M| = {len(m_all)}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    c = table[table["target"] == "C"].set_index("method")
    for m in (4, 5):
        best = max(c.loc[[f"anchored({m})", "one-in-one-out", "totals"], "f1"])
        print(f"  H18 at length {m}: F1 ours {c.loc[f'ours({m})', 'f1']:.3f} vs best simple {best:.3f}; "
              f"holds here: {c.loc[f'ours({m})', 'f1'] >= best + 0.10}")


if __name__ == "__main__":
    main()
