"""Check the amount assumption and the lower bound of the background retention sandwich, from data and existing records.

For a data set, labeling S and theta:

* p_m(theta): the probability that m amounts drawn independently from the empirical amount distribution conserve at
  every consecutive pair, (1 - theta) x_i <= x_{i+1} <= (1 + theta) x_i. Computed exactly for the empirical
  distribution by dynamic programming over the sorted amounts: h_1 = 1, h_{k+1}(x) = mean of h_k(y) over the amounts y
  with (1 - theta) x <= y <= (1 + theta) x, p_m = mean of h_m. Every transaction of the chain input is used, laundering
  ones included (a small share), so this is the distribution of all amounts, not of background amounts only.
* w_m: the fraction of sampled background chains that conserve, read from a chain-planted-rules record.
* r_B, r_L and OR: background and laundering retention and their odds ratio, read from a chain-conservation-effect record:
  r_B = (I_theta - marked_theta) / (I_inf - marked_inf), r_L = marked_theta / marked_inf; OR must equal r_L / r_B.

If background amounts are independent of the graph and identically distributed, every chain conserves with probability
exactly p_m, so w_m should be close to p_m, and every background intermediate account is retained with probability at
least p_m, so r_B >= p_m.

Run: python experiments/probes/saturation_check.py --dataset HI-Medium --effect <run id> --planted <run id>
     python experiments/probes/saturation_check.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else None


def p_conserving(amounts: np.ndarray, theta: float, max_m: int) -> list[float]:
    """[p_1, ..., p_max_m] for the empirical distribution of amounts."""
    x = np.sort(np.asarray(amounts, dtype=np.float64))
    n = len(x)
    lo = np.searchsorted(x, (1 - theta) * x, side="left")     # first y >= (1 - theta) x
    hi = np.searchsorted(x, (1 + theta) * x, side="right")    # past the last y <= (1 + theta) x
    h = np.ones(n)
    out = [1.0]
    for _ in range(1, max_m):
        prefix = np.concatenate(([0.0], np.cumsum(h)))
        h = (prefix[hi] - prefix[lo]) / n
        out.append(float(h.mean()))
    return out


def chain_amounts(dataset: str, labeling: str = "S") -> np.ndarray:
    path = ROOT / "datasets" / "chain" / f"{dataset}_{labeling}.tg"
    frame = pd.read_csv(path, sep=" ", header=None, comment="#", usecols=[0, 4], names=["kind", "amount"],
                        dtype={"kind": "category"}, engine="c")
    return frame.loc[frame["kind"] == "e", "amount"].to_numpy(dtype=np.float64)


def self_test() -> None:
    rng = np.random.default_rng(3)
    amounts = rng.choice([60.0, 75.0, 80.0, 96.0, 100.0, 120.0, 125.0], size=9)
    theta = 0.25

    def ok(a, b):
        return (1 - theta) * a <= b <= (1 + theta) * a

    for m in (2, 3, 4):
        brute = np.mean([all(ok(s[i], s[i + 1]) for i in range(m - 1)) for s in itertools.product(amounts, repeat=m)])
        got = p_conserving(amounts, theta, m)[m - 1]
        assert abs(got - brute) < 1e-12, (m, got, brute)
    # a distribution with every amount equal conserves always; two far-apart values conserve only within themselves
    assert p_conserving(np.full(5, 7.0), 0.0, 4) == [1.0, 1.0, 1.0, 1.0]
    assert abs(p_conserving(np.array([1.0, 100.0]), 0.2, 3)[2] - 0.25) < 1e-12
    print("self-test: p_m equals brute force over all sequences for m = 2, 3, 4 (9 amounts, boundaries included); "
          "two closed-form cases pass")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--effect", help="run id under results-invariant/chain-conservation-effect")
    ap.add_argument("--planted", help="run id under results-invariant/chain-planted-rules")
    ap.add_argument("--theta", type=float, default=0.2)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    effect = list(csv.DictReader(open(ROOT / "results-invariant" / "chain-conservation-effect" / args.effect
                                      / "conservation_effect.csv")))
    planted = {int(r["length"]): r for r in csv.DictReader(open(ROOT / "results-invariant" / "chain-planted-rules"
                                                                 / args.planted / "background_rules.csv"))}
    if {r["dataset"] for r in effect} != {args.dataset} or {r["labeling"] for r in effect} != {"S"}:
        sys.exit(f"the effect record is not {args.dataset}, labeling S")
    amounts = chain_amounts(args.dataset)
    p = p_conserving(amounts, args.theta, 6)
    rows = []
    for r in effect:
        m = int(r["length"])
        i_t, mk_t, i_i, mk_i = (float(r[k]) for k in ("I_0.2", "marked_0.2", "I_inf", "marked_inf"))
        r_b = (i_t - mk_t) / (i_i - mk_i)
        r_l = mk_t / mk_i if mk_i else float("nan")
        w = planted.get(m)
        rows.append({"dataset": args.dataset, "labeling": "S", "theta": args.theta, "length": m,
                     "transactions": len(amounts), "p_m": p[m - 1],
                     "w_m": float(w["pairwise"]) if w else "", "background_chains": int(w["n"]) if w else "",
                     "w_over_p": float(w["pairwise"]) / p[m - 1] if w else "",
                     "r_B": r_b, "r_L": r_l, "OR_record": float(r["OR"]), "OR_identity": r_l / r_b,
                     "lower_bound_holds": r_b >= p[m - 1]})
    for row in rows:
        print(", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    if OUT is not None:
        with open(OUT / "saturation_check.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
