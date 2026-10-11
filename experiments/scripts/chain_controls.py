"""Controls for the conservation effect: degree, chance, and uncertainty.

Reads one conservation-effect record (the intermediate accounts of the open chains mined with
theta = 0.2 and without conservation) and, for every length m:

  degree control   P of the accounts returned with conservation against P of random draws, with
                   replacement, from the accounts returned without conservation that match them
                   bin by bin in degree (bin = floor(log2(in-degree + out-degree)), counted in
                   transactions); 200 draws, seed 20261014; OR_degree = odds(P) / odds(mean matched P)
  chance control   the laundering flag shuffled among all accounts of the same degree bin, 200 times,
                   seed 20261014; OR recomputed under each shuffle (expected near 1)
  intervals        bootstrap over laundering attempts (1,000 resamples, seed 20261015) for P with
                   conservation, R with conservation, OR and OR_degree; an account in c attempts
                   counts 1/c in each, so the bootstrap is centred on the point estimate

Run: python experiments/scripts/chain_controls.py --source amlworld --dataset HI-Small --labeling S \\
         --effect-record results-invariant/chain-conservation-effect/<run>
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

from inchusm_aml.eval import planted_chains as pc  # noqa: E402
from round2_measurements import load  # noqa: E402

OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"
DRAWS, SEED_DRAWS, BOOT, SEED_BOOT = 200, 20261014, 1000, 20261015
MIN_UNITS = 30


def account_file(record: Path, theta: str) -> Path:
    """The account-role file of a run, as written or as stored gzip-compressed."""
    plain = record / f"accounts_theta_{theta}.tsv"
    return plain if plain.is_file() else plain.with_name(plain.name + ".gz")


def odds(p: float) -> float:
    return p / (1 - p) if 0 <= p < 1 else float("inf")


def attempt_shares(returned: set, groups: list[set]) -> np.ndarray:
    """Each group's share of |returned & union(groups)|: an account in c groups counts 1/c in each.

    Summing the shares over the groups gives |returned & union(groups)| exactly, so resampling the
    groups with replacement gives a bootstrap whose expectation is the point estimate. Counting an
    account once in every group that contains it instead overstates every resample when groups overlap.
    """
    count: dict = {}
    for g in groups:
        for v in g:
            count[v] = count.get(v, 0) + 1
    return np.array([sum(1.0 / count[v] for v in g & returned) for g in groups], dtype=float)


def degree_bins(edges: pd.DataFrame) -> pd.Series:
    e = edges[edges["src"] != edges["dst"]]
    deg = e["src"].value_counts().add(e["dst"].value_counts(), fill_value=0)
    return np.floor(np.log2(deg)).astype(int)


def matched_precision(returned: np.ndarray, pool: np.ndarray, bins: pd.Series, marked: set,
                      rng: np.random.Generator, draws: int) -> np.ndarray:
    """P of `draws` random samples from pool that match `returned` bin by bin (with replacement)."""
    pool_bins = bins.reindex(pool).to_numpy()
    need = pd.Series(bins.reindex(returned).to_numpy()).value_counts()
    by_bin = {b: pool[pool_bins == b] for b in need.index}
    is_marked = {b: np.array([a in marked for a in by_bin[b]]) for b in need.index}
    out = np.empty(draws)
    for d in range(draws):
        hit = 0
        for b, k in need.items():
            hit += is_marked[b][rng.integers(0, len(by_bin[b]), k)].sum()
        out[d] = hit / need.sum()
    return out


def shuffled_marks(bins: pd.Series, marked: set, rng: np.random.Generator) -> set:
    """The laundering flag permuted among accounts of the same degree bin."""
    accounts = bins.index.to_numpy()
    flags = np.array([a in marked for a in accounts])
    b = bins.to_numpy()
    new = flags.copy()
    for value in np.unique(b):
        idx = np.flatnonzero(b == value)
        new[idx] = flags[rng.permutation(idx)]
    return set(accounts[new].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labeling", choices=["S", "A"], required=True)
    ap.add_argument("--effect-record", required=True)
    args = ap.parse_args()
    rec = ROOT / args.effect_record
    cmd = " ".join(json.loads((rec / "run_meta.json").read_text())["command"])
    if f"--dataset {args.dataset}" not in cmd or f"--labeling {args.labeling}" not in cmd:
        sys.exit(f"{rec}: not the conservation-effect run of {args.dataset}, labeling {args.labeling}")

    def roles(theta):
        r = pd.read_csv(account_file(rec, theta), sep="\t")
        r = r[(r["kind"] == "open") & (r["role"] == "intermediate")]
        return {m: g["account"].to_numpy() for m, g in r.groupby("length")}
    with_c, without_c = roles("0.2"), roles("inf")

    edges, ptx, _ = load(args.source, args.dataset)
    bins = degree_bins(edges)
    flagged = edges["laundering"].to_numpy() == 1
    marked = set(np.union1d(edges["src"].to_numpy()[flagged], edges["dst"].to_numpy()[flagged]).tolist())
    planted = pc.planted_chains(ptx, 5)
    planted = planted[planted["kind"] == "open"]
    attempts = sorted(ptx["attempt"].unique())
    acc_of = {a: set(g["src"]) | set(g["dst"]) for a, g in ptx.groupby("attempt")}
    mid_of = {(a, m): set(x for acc in g["accounts"] for x in acc[1:-1])
              for (a, m), g in planted.groupby(["attempt", "length"])}
    outside = marked - set().union(*acc_of.values())

    rng_draw = np.random.default_rng(SEED_DRAWS)
    shuffles = [shuffled_marks(bins, marked, rng_draw) for _ in range(DRAWS)]
    rows = []
    for m in sorted(set(with_c) | set(without_c)):
        i1, i0 = with_c.get(m, np.array([])), without_c.get(m, np.array([]))
        s1, s0 = set(i1.tolist()), set(i0.tolist())
        row = {"dataset": args.dataset, "labeling": args.labeling, "length": m, "I_0.2": len(s1), "I_inf": len(s0)}
        if len(s1) < MIN_UNITS or not len(s0):
            rows.append(row)
            continue
        p1, p0 = len(s1 & marked) / len(s1), len(s0 & marked) / len(s0)
        matched = matched_precision(i1, i0, bins, marked, rng_draw, DRAWS)
        pm = float(matched.mean())
        row.update({"P_0.2": p1, "P_inf": p0, "OR": odds(p1) / odds(p0), "P_degree_matched": pm,
                    "P_degree_matched_ci": f"[{np.percentile(matched, 2.5):.4f}, {np.percentile(matched, 97.5):.4f}]",
                    "OR_degree": odds(p1) / odds(pm)})
        ors = [odds(len(s1 & sh) / len(s1)) / odds(len(s0 & sh) / len(s0)) for sh in shuffles]
        row["OR_shuffled_median"] = float(np.median(ors))
        row["OR_shuffled_ci"] = f"[{np.percentile(ors, 2.5):.3f}, {np.percentile(ors, 97.5):.3f}]"
        # bootstrap over attempts: laundering accounts and planted intermediates travel with their attempt
        groups = [acc_of[a] for a in attempts]
        hit1, hit0 = attempt_shares(s1, groups), attempt_shares(s0, groups)
        mids = [mid_of.get((a, m), set()) for a in attempts]
        rec_hit = attempt_shares(s1, mids)
        rec_n = attempt_shares(set().union(*mids), mids)
        fixed1, fixed0 = len(s1 & outside), len(s0 & outside)
        rng_b = np.random.default_rng(SEED_BOOT)
        bp, br, bo, bd = [], [], [], []
        for _ in range(BOOT):
            idx = rng_b.integers(0, len(attempts), len(attempts))
            # true positives are re-weighted; the returned sets themselves are fixed
            t1 = hit1[idx].sum() + fixed1
            t0 = hit0[idx].sum() + fixed0
            q1 = t1 / (t1 + len(s1 - marked))
            q0 = t0 / (t0 + len(s0 - marked))
            bp.append(q1)
            bo.append(odds(q1) / odds(q0) if 0 < q0 < 1 else np.nan)
            bd.append(odds(q1) / odds(pm) if 0 < pm < 1 else np.nan)
            br.append(rec_hit[idx].sum() / rec_n[idx].sum() if rec_n[idx].sum() else np.nan)
        row["R_0.2_point"] = rec_hit.sum() / rec_n.sum() if rec_n.sum() else float("nan")
        for name, vals in (("P_0.2", bp), ("R_0.2", br), ("OR", bo), ("OR_degree", bd)):
            v = np.array(vals, float)
            v = v[~np.isnan(v)]
            row[f"{name}_ci"] = f"[{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]" if len(v) else ""
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "controls.csv", index=False)
    outside, checked = 0, 0
    for _, r in table.iterrows():
        for name in ("P_0.2", "R_0.2", "OR", "OR_degree"):
            point = r.get(name if name != "R_0.2" else "R_0.2_point")
            ci = r.get(f"{name}_ci")
            if isinstance(ci, str) and ci and point is not None and not pd.isna(point):
                lo, hi = (float(x) for x in ci.strip("[]").split(", "))
                checked += 1
                outside += not (lo <= point <= hi)
    print(f"intervals checked against their point estimate: {checked}; not containing it: {outside}")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"{args.dataset} labeling {args.labeling}: {len(marked)} laundering accounts, {len(attempts)} attempts")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.source == "amlworld":
        for m in (4, 5):
            r = table[table["length"] == m]
            v = r["OR_degree"].iat[0] if len(r) and "OR_degree" in r and not pd.isna(r["OR_degree"].iat[0]) else None
            print(f"  H17 at length {m}: OR_degree = {v}; holds here: {None if v is None else v >= 2}")


if __name__ == "__main__":
    main()
