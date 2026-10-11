"""Two application questions: drifting chains that the anchored rule misses, and ranking for investigation.

Drift (structures a simpler rule misses): a planted open chain of length m that is valid under the
pairwise rule (theta = 0.2) but not under the rule anchored on the first transaction (alpha = 0.2),
i.e. its amount drifts step by step. D(m) = intermediate accounts of such chains. For conserved
chains and for the anchored rule, R_D(m) = share of D(m) in the returned set I(m).

Ranking (investigation priority): every account with incoming and outgoing transactions gets a
score from each method; precision@k against the laundering accounts M (and against the planted
intermediates C) is the expected precision when ties at the k-th position are ordered uniformly at
random, computed exactly.

  conserved   largest m with the account in I(m) (theta = 0.2), 0 if none
  anchored    largest m with the account in the anchored rule's I(m)
  totals      -|ln(total out / total in)|
  one-in-one-out  1 for accounts with exactly one incoming and one outgoing transaction, else 0
  low degree  -(incoming + outgoing transactions)

Run: python experiments/scripts/chain_prioritisation.py --source amlworld --dataset HI-Small \\
         --effect-record results-invariant/chain-conservation-effect/<run, labeling S> \\
         --simple-record results-invariant/chain-simple-rules/<run>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "probes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inchusm_aml.eval import planted_chains as pc  # noqa: E402
from round2_measurements import load  # noqa: E402

OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"
THETA = ALPHA = 0.2
LENGTHS, KS, MIN_DRIFT = (2, 3, 4, 5), (100, 1000, 10000), 30


def roles(path: Path) -> dict[int, set]:
    p = path if path.is_file() else path.with_name(path.name + ".gz")
    r = pd.read_csv(p, sep="\t")
    r = r[(r["kind"] == "open") & (r["role"] == "intermediate")]
    return {m: set(g["account"].tolist()) for m, g in r.groupby("length")}


def drift_intermediates(planted: pd.DataFrame) -> dict[int, set]:
    """Intermediates of planted open chains valid pairwise but not anchored, by length."""
    p = planted[planted["kind"] == "open"]
    drift = p[p["amounts"].map(lambda a: pc.pairwise(a, THETA) and not pc.anchored(a, ALPHA))]
    return {m: set(x for acc in g["accounts"] for x in acc[1:-1]) for m, g in drift.groupby("length")}


def max_length_score(by_length: dict[int, set], universe) -> dict:
    score = {v: 0 for v in universe}
    for m in sorted(by_length):
        for v in by_length[m]:
            if v in score:
                score[v] = max(score[v], m)
    return score


def precision_at_k(score: dict, positives: set, k: int) -> float:
    """Expected precision of the top k when ties are ordered uniformly at random."""
    k = min(k, len(score))
    if k == 0:
        return float("nan")
    groups: dict = {}
    for v, s in score.items():
        g = groups.setdefault(s, [0, 0])
        g[0] += 1
        g[1] += v in positives
    taken, hits = 0, 0.0
    for s in sorted(groups, reverse=True):
        size, pos = groups[s]
        if taken + size <= k:
            taken += size
            hits += pos
        else:
            hits += (k - taken) * pos / size
            taken = k
        if taken == k:
            break
    return hits / k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--effect-record", required=True)
    ap.add_argument("--simple-record", required=True)
    args = ap.parse_args()
    eff, simp = ROOT / args.effect_record, ROOT / args.simple_record
    for rec, needle in ((eff, "--labeling S"), (simp, "chain_simple_baselines.py")):
        cmd = " ".join(json.loads((rec / "run_meta.json").read_text())["command"])
        if f"--dataset {args.dataset}" not in cmd or needle not in cmd:
            sys.exit(f"{rec}: not the expected run for {args.dataset}")
    ours, anchored = roles(eff / "accounts_theta_0.2.tsv"), roles(simp / "accounts_anchored.tsv")

    edges, ptx, _ = load(args.source, args.dataset)
    flagged = edges["laundering"].to_numpy() == 1
    marked = set(np.union1d(edges["src"].to_numpy()[flagged], edges["dst"].to_numpy()[flagged]).tolist())
    planted = pc.planted_chains(ptx, max(LENGTHS))
    c_all = set(x for acc in planted[planted["kind"] == "open"]["accounts"] for x in acc[1:-1])
    drift = drift_intermediates(planted)

    rows = []
    for m in LENGTHS:
        d = drift.get(m, set())
        row = {"dataset": args.dataset, "length": m, "drift_intermediates": len(d)}
        for name, sets in (("conserved", ours), ("anchored", anchored)):
            s = sets.get(m, set())
            row[f"I_{name}"] = len(s)
            row[f"P_{name}"] = len(s & marked) / len(s) if s else float("nan")
            row[f"R_drift_{name}"] = len(s & d) / len(d) if d else float("nan")
        row["R_drift_gap"] = row["R_drift_conserved"] - row["R_drift_anchored"] if d else float("nan")
        rows.append(row)
    drift_table = pd.DataFrame(rows)
    drift_table.to_csv(OUT / "drift_recovery.csv", index=False)

    e = edges[edges["src"] != edges["dst"]]
    indeg, outdeg = e.groupby("dst").size(), e.groupby("src").size()
    universe = indeg.index.intersection(outdeg.index)
    sin, sout = e.groupby("dst")["amount"].sum(), e.groupby("src")["amount"].sum()
    deviation = np.abs(np.log((sout[universe] / sin[universe]).to_numpy()))
    scores = {
        "conserved": max_length_score(ours, universe),
        "anchored": max_length_score(anchored, universe),
        "totals": dict(zip(universe, -deviation)),
        "one-in-one-out": dict(zip(universe, ((indeg[universe] == 1) & (outdeg[universe] == 1)).astype(int).to_numpy())),
        "low degree": dict(zip(universe, -(indeg[universe] + outdeg[universe]).to_numpy())),
    }
    prow = []
    for target, positives in (("M", marked), ("C", c_all)):
        for name, sc in scores.items():
            prow.append({"dataset": args.dataset, "target": target, "method": name,
                         **{f"precision_at_{k}": precision_at_k(sc, positives, k) for k in KS}})
    prec = pd.DataFrame(prow)
    prec.to_csv(OUT / "precision_at_k.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"{args.dataset}: {len(universe)} accounts with incoming and outgoing transactions, {len(marked)} laundering")
    print("\ndrifting planted chains (pairwise-valid, anchored-invalid):")
    print(drift_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nprecision at k:")
    print(prec.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for m in (4, 5):
        r = drift_table[drift_table["length"] == m].iloc[0]
        if r["drift_intermediates"] >= MIN_DRIFT:
            print(f"  H22 at length {m}: gap {r['R_drift_gap']:.3f}; holds here: {r['R_drift_gap'] >= 0.5}")
        else:
            print(f"  H22 at length {m}: only {int(r['drift_intermediates'])} drift intermediates - not counted")
    pm = prec[prec["target"] == "M"].set_index("method")["precision_at_1000"]
    best = pm.drop("conserved").max()
    print(f"  H23: precision@1000 conserved {pm['conserved']:.4f} vs best other {best:.4f}; "
          f"holds here: {pm['conserved'] >= best + 0.10}")


if __name__ == "__main__":
    main()
