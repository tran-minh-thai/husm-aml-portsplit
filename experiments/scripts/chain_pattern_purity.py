"""Pattern-level evaluation with attribute labels: how much of each returned chain pattern is laundering.

For one data set and labeling A (AMLworld: payment format of each transaction; AMLGentex: bank of
each account), three methods return chain patterns at minimum-image support tau = 10, length <= 5,
open chains only:

  conserved      pairwise conservation, theta = 0.2   (from the conservation-effect run)
  unconstrained  no amount constraint, theta = inf    (from the same run)
  anchored       every amount within 0.2 of the first (enumeration engine, run here)

Purity of a pattern P of length m: sum over the intermediate positions i = 1..m-1 of the marked
(laundering) accounts in the image set of position i, divided by the sum of the image set sizes.

  paired     for m = 4, 5: every pattern found with conservation is also found without it (its valid
             occurrences are a subset); per pattern, purity(theta 0.2) - purity(theta inf)
  top ten    patterns ranked by length, then support, both decreasing (no laundering labels used);
             the mean purity of the first ten, as an expectation over uniformly random order within
             ties at the cut (computed exactly)

Run: python experiments/scripts/chain_pattern_purity.py --source amlworld --dataset LI-Small \\
         --effect-record results-invariant/chain-conservation-effect/<run, labeling A>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
JAR = ROOT / "inchusm-miner" / "target" / "inchusm-miner-0.1.0-SNAPSHOT.jar"
CHAIN = ROOT / "datasets" / "chain"
OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"
ALPHA, TAU, MAX_LENGTH, TOP, MIN_PAIRED = 0.2, 10, 5, 10, 3


def purity_table(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path, sep="\t", dtype={"account_labels": str, "transaction_labels": str,
                                            "image_counts": str, "marked_counts": str})
    t = t[t["kind"] == "open"].copy()
    if t["marked_counts"].isna().any():
        sys.exit(f"{path}: some patterns have no marked counts (run without --mark)")

    def middle(s):
        v = [int(x) for x in s.split(",")]
        return sum(v[1:-1])
    t["image_mid"] = t["image_counts"].map(middle)
    t["marked_mid"] = t["marked_counts"].map(middle)
    t = t[t["length"] >= 2]
    t["purity"] = t["marked_mid"] / t["image_mid"]
    return t[["length", "account_labels", "transaction_labels", "support", "image_mid", "marked_mid", "purity"]]


def paired(conserved: pd.DataFrame, unconstrained: pd.DataFrame, m: int) -> tuple[pd.DataFrame, int]:
    key = ["length", "account_labels", "transaction_labels"]
    a = conserved[conserved["length"] == m]
    j = a.merge(unconstrained, on=key, how="left", suffixes=("_conserved", "_unconstrained"), indicator=True)
    missing = int((j["_merge"] != "both").sum())
    j = j[j["_merge"] == "both"].copy()
    j["difference"] = j["purity_conserved"] - j["purity_unconstrained"]
    return j, missing


def top(table: pd.DataFrame, k: int = TOP) -> dict:
    """Expected mean purity, marked and image totals of the first k patterns, ties at the cut in random order."""
    k = min(k, len(table))
    if k == 0:
        return {"patterns": 0, "mean_purity": float("nan"), "weighted_purity": float("nan"), "image_mid": 0.0}
    taken, purity, marked, image = 0, 0.0, 0.0, 0.0
    for _, g in sorted(table.groupby(["length", "support"]), key=lambda kv: (-kv[0][0], -kv[0][1])):
        share = min(1.0, (k - taken) / len(g))
        purity += share * g["purity"].sum()
        marked += share * g["marked_mid"].sum()
        image += share * g["image_mid"].sum()
        taken += min(len(g), k - taken)
        if taken == k:
            break
    return {"patterns": k, "mean_purity": purity / k, "weighted_purity": marked / image, "image_mid": image}


def run_anchored(dataset: str, heap: str) -> Path:
    out = OUT / "patterns_anchored.tsv"
    cmd = ["java", f"-Xmx{heap}", "-cp", str(JAR), "edu.huflit.inchusm.chain.ChainMain", "--engine", "enumerate",
           "--rule", "anchored", "--graph", str(CHAIN / f"{dataset}_A.tg"), "--theta", str(ALPHA), "--min-support",
           str(TAU), "--max-length", str(MAX_LENGTH), "--no-closed", "--mark",
           str(CHAIN / f"{dataset}_laundering_accounts.txt"), "--out", str(out)]
    print("[enumerator, anchored rule, labeling A]", flush=True)
    subprocess.run(cmd, check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--effect-record", required=True, help="conservation-effect run of this data set, labeling A")
    ap.add_argument("--heap", default="16g")
    args = ap.parse_args()
    rec = ROOT / args.effect_record
    cmd = " ".join(json.loads((rec / "run_meta.json").read_text())["command"])
    if f"--dataset {args.dataset} " not in cmd + " " or "--labeling A" not in cmd:
        sys.exit(f"{rec}: not the conservation-effect run of {args.dataset} with labeling A")

    tables = {"conserved": purity_table(rec / "patterns_theta_0.2.tsv"),
              "unconstrained": purity_table(rec / "patterns_theta_inf.tsv"),
              "anchored": purity_table(run_anchored(args.dataset, args.heap))}

    rows = []
    for m in (4, 5):
        j, missing = paired(tables["conserved"], tables["unconstrained"], m)
        if missing:
            sys.exit(f"length {m}: {missing} conserved patterns are missing without conservation - impossible")
        rows.append({"dataset": args.dataset, "length": m, "paired_patterns": len(j),
                     "counted": len(j) >= MIN_PAIRED,
                     "median_difference": j["difference"].median() if len(j) else float("nan"),
                     "min_difference": j["difference"].min() if len(j) else float("nan"),
                     "median_purity_conserved": j["purity_conserved"].median() if len(j) else float("nan"),
                     "median_purity_unconstrained": j["purity_unconstrained"].median() if len(j) else float("nan")})
    pair_table = pd.DataFrame(rows)
    pair_table.to_csv(OUT / "paired_purity.csv", index=False)

    trows = []
    for name, t in tables.items():
        counts = t.groupby("length").size().to_dict()
        trows.append({"dataset": args.dataset, "method": name, **top(t),
                      **{f"patterns_length_{m}": counts.get(m, 0) for m in range(2, MAX_LENGTH + 1)}})
    top_table = pd.DataFrame(trows)
    top_table.to_csv(OUT / "top_patterns.csv", index=False)
    for name, t in tables.items():
        t.assign(method=name).to_csv(OUT / f"purity_{name}.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"\n{args.dataset}, labeling A: paired purity, conserved minus unconstrained, same pattern")
    print(pair_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nfirst {TOP} patterns by length then support:")
    print(top_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    counted = pair_table[pair_table["counted"]]
    print(f"  H27 here: {len(counted)} counted cells; all with median >= 0.3: "
          f"{bool(len(counted)) and bool((counted['median_difference'] >= 0.3).all())}")
    mp = top_table.set_index("method")["mean_purity"]
    print(f"  H28 here: conserved {mp['conserved']:.3f} vs anchored {mp['anchored']:.3f}; "
          f"holds: {mp['conserved'] >= mp['anchored'] + 0.10}")


if __name__ == "__main__":
    main()
