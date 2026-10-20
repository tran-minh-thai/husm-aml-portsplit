"""Uncertainty of the conservation effect: OR by a bootstrap over laundering attempts, R by Clopper-Pearson.

OR = odds(P with conservation) / odds(P without) = r_L / r_B. Its 95% interval is the bootstrap of chain_controls.py:
1,000 resamples (seed 20261015) of the laundering attempts with replacement; the returned sets stay fixed and the
marked accounts in them are re-weighted, an account of c attempts counting 1/c in each and a laundering account outside
every attempt counting as fixed. The recovery R = |I_0.2 & C| / |C| on the planted accounts C gets the exact binomial
(Clopper-Pearson) 95% interval, which treats the planted accounts as independent.

For a record of chain_conservation_effect.py (--kind open) or chain_cycle_effect.py (--kind closed), reads the two
account-role files of the record and the laundering attempts of the data set, and writes effect_intervals.csv:
one row per length with |I|, P, OR and its interval, R and its interval. The function or_interval works on any
hashable accounts (windowed_effect.py passes (window, account) pairs).

Run: python experiments/scripts/effect_intervals.py --kind closed --dataset HI-Large \\
         --record results-invariant/chain-cycle-effect-final/<run> [--source amlworld]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "probes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chain_controls import BOOT, SEED_BOOT, attempt_shares, odds  # noqa: E402
from inchusm_aml.eval import planted_chains as pc  # noqa: E402
from round2_measurements import load  # noqa: E402

OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"


def or_interval(s1: set, s0: set, marked: set, groups: list[set]) -> tuple[float, float]:
    """95% bootstrap interval of the odds ratio of s1 (with conservation) against s0 (without), as chain_controls.py."""
    outside = marked - set().union(*groups) if groups else set(marked)
    hit1, hit0 = attempt_shares(s1, groups), attempt_shares(s0, groups)
    fixed1, fixed0 = len(s1 & outside), len(s0 & outside)
    clean1, clean0 = len(s1 - marked), len(s0 - marked)
    rng = np.random.default_rng(SEED_BOOT)
    vals = []
    for _ in range(BOOT):
        idx = rng.integers(0, len(groups), len(groups))
        t1 = hit1[idx].sum() + fixed1
        t0 = hit0[idx].sum() + fixed0
        q1 = t1 / (t1 + clean1)
        q0 = t0 / (t0 + clean0)
        vals.append(odds(q1) / odds(q0) if 0 < q0 < 1 else np.nan)
    v = np.array(vals, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return float("nan"), float("nan")
    # resamples without any background account in s1 give an infinite ratio; interpolating between two infinite
    # order statistics is undefined, so then the bounds are order statistics (finite bounds unchanged otherwise)
    method = "inverted_cdf" if np.isinf(v).any() else "linear"
    return float(np.percentile(v, 2.5, method=method)), float(np.percentile(v, 97.5, method=method))


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def role_sets(record: Path, theta: str, kind: str) -> dict[int, set]:
    plain = record / f"accounts_theta_{theta}.tsv"
    path = plain if plain.is_file() else plain.with_name(plain.name + ".gz")
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")) as f:
        r = pd.read_csv(f, sep="\t")
    r = r[r["kind"] == kind]
    if kind == "open":
        r = r[r["role"] == "intermediate"]
    return {int(m): set(g["account"].tolist()) for m, g in r.groupby("length")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["open", "closed"], required=True)
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], default="amlworld")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--record", required=True)
    args = ap.parse_args()
    rec = ROOT / args.record
    cmd = json.loads((rec / "run_meta.json").read_text(encoding="utf-8"))["command"]
    if "--dataset" not in cmd or cmd[cmd.index("--dataset") + 1] != args.dataset:
        sys.exit(f"{args.record} is not a run on {args.dataset}")
    s02, sinf = role_sets(rec, "0.2", args.kind), role_sets(rec, "inf", args.kind)
    edges, ptx, _ = load(args.source, args.dataset)
    flagged = edges["laundering"].to_numpy() == 1
    nonself = edges[edges["src"] != edges["dst"]]
    ids = np.union1d(nonself["src"].to_numpy(), nonself["dst"].to_numpy())
    marked = set(np.intersect1d(np.union1d(edges["src"].to_numpy()[flagged], edges["dst"].to_numpy()[flagged]), ids).tolist())
    attempts = sorted(ptx["attempt"].unique())
    acc_of = {a: set(g["src"]) | set(g["dst"]) for a, g in ptx.groupby("attempt")}
    groups = [acc_of[a] for a in attempts]
    planted = pc.planted_chains(ptx, 5)
    planted = planted[planted["kind"] == args.kind]
    take = (lambda acc: acc[1:-1]) if args.kind == "open" else (lambda acc: acc)
    plant = {int(m): set(x for acc in g["accounts"] for x in take(acc)) for m, g in planted.groupby("length")}
    rows = []
    for m in sorted(set(s02) | set(sinf)):
        s1, s0, c = s02.get(m, set()), sinf.get(m, set()), plant.get(m, set())
        row = {"dataset": args.dataset, "kind": args.kind, "length": m, "I_0.2": len(s1), "I_inf": len(s0),
               "marked_0.2": len(s1 & marked), "marked_inf": len(s0 & marked), "planted": len(c)}
        p1 = len(s1 & marked) / len(s1) if s1 else float("nan")
        p0 = len(s0 & marked) / len(s0) if s0 else float("nan")
        row["OR"] = odds(p1) / odds(p0) if s1 and s0 and 0 < p0 < 1 else float("nan")
        row["OR_lo"], row["OR_hi"] = or_interval(s1, s0, marked, groups) if s1 and s0 else (float("nan"), float("nan"))
        k = len(s1 & c)
        row["R_0.2"] = k / len(c) if c else float("nan")
        row["R_lo"], row["R_hi"] = clopper_pearson(k, len(c))
        rows.append(row)
        print(f"{args.dataset} {args.kind} m={m}: OR {row['OR']:.4f} [{row['OR_lo']:.4f}, {row['OR_hi']:.4f}]; "
              f"R {row['R_0.2']:.4f} [{row['R_lo']:.4f}, {row['R_hi']:.4f}] of {len(c)} planted", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "effect_intervals.csv", index=False)
    if any(not math.isnan(r["OR"]) and not (r["OR_lo"] <= r["OR"] <= r["OR_hi"]) for r in rows):
        print("note: a point estimate lies outside its bootstrap interval", flush=True)


if __name__ == "__main__":
    main()
