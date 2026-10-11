"""Two money-laundering signals outside the chain miner, on AMLworld: structuring and velocity.

Structuring: the ratio of transactions with a US Dollar amount in [9,000, 10,000) to those in
[10,000, 11,000), for laundering and for background transactions, over every payment format and over
cash only. Structuring would show as a laundering ratio well above the background ratio.

Velocity, over the accounts U with incoming and outgoing transactions (self-transactions left out):
  dwell time   for every incoming transaction, the minutes until the account's next outgoing transaction
               (at the same minute or later); the median over incoming transactions that have one.
               Accounts with none rank last.
  night share  share of an account's transactions (in or out) with a clock time in [00:00, 06:00)
Each ranks U (short dwell first; high night share first). Precision at k against the laundering
accounts M is the expectation over uniformly random order within ties (as for the other rankings).
The combined score ranks by the longest conserved chain length containing the account (theta = 0.2,
labeling S), then by dwell time.

Run: python experiments/scripts/chain_signals.py --dataset HI-Small \\
         --effect-record results-invariant/chain-conservation-effect/<run, labeling S>
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chain_prioritisation import max_length_score, precision_at_k, roles  # noqa: E402

OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"
BELOW, ABOVE = (9_000.0, 10_000.0), (10_000.0, 11_000.0)
KS = (100, 1000, 10000)
NIGHT_END = 6 * 60          # minutes after midnight
NO_DWELL = 10 ** 8          # rank accounts without a following outgoing transaction last


def band_ratio(amount: np.ndarray) -> dict:
    below = int(((amount >= BELOW[0]) & (amount < BELOW[1])).sum())
    above = int(((amount >= ABOVE[0]) & (amount < ABOVE[1])).sum())
    return {"below": below, "above": above, "ratio": below / above if above else float("nan")}


def structuring(edges: pd.DataFrame, fmt: np.ndarray) -> pd.DataFrame:
    rows = []
    for subset, mask in (("every format", np.ones(len(edges), bool)), ("cash", fmt == "Cash")):
        for group, flag in (("laundering", 1), ("background", 0)):
            sel = mask & (edges["laundering"].to_numpy() == flag)
            rows.append({"subset": subset, "group": group, "transactions": int(sel.sum()),
                         **band_ratio(edges["amount"].to_numpy()[sel])})
    return pd.DataFrame(rows)


def dwell_minutes(src, dst, time) -> pd.Series:
    """Median minutes from each incoming transaction of an account to its next outgoing one."""
    shift = np.int64(1) << 26
    if time.max() >= shift:
        raise ValueError("time does not fit the key")
    t0 = time - time.min()
    out_key = np.sort(src.astype(np.int64) * shift + t0)
    in_key = dst.astype(np.int64) * shift + t0
    idx = np.searchsorted(out_key, in_key, side="left")
    ok = idx < len(out_key)
    ok[ok] = (out_key[idx[ok]] // shift) == dst[ok]
    gap = (out_key[idx[ok]] % shift) - t0[ok]
    return pd.Series(gap).groupby(dst[ok]).median()


def night_share(src, dst, time) -> pd.Series:
    night = (time % 1440) < NIGHT_END
    acct = np.concatenate([src, dst])
    flag = np.concatenate([night, night]).astype(float)
    return pd.Series(flag).groupby(acct).mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--effect-record", required=True, help="conservation-effect run of this data set, labeling S")
    args = ap.parse_args()
    rec = ROOT / args.effect_record
    cmd = " ".join(json.loads((rec / "run_meta.json").read_text())["command"])
    if f"--dataset {args.dataset} " not in cmd + " " or "--labeling S" not in cmd:
        sys.exit(f"{rec}: not the conservation-effect run of {args.dataset} with labeling S")

    from inchusm_aml.data.amlworld import load_transactions
    path = ROOT / "datasets" / "amlworld" / f"{args.dataset}_Trans.csv"
    edges, _ = load_transactions(path)
    fmt = pd.read_csv(path, usecols=[9], names=["format"], header=0, dtype={"format": str})["format"].to_numpy()
    if len(fmt) != len(edges):
        sys.exit(f"{path.name}: {len(fmt)} payment formats for {len(edges)} transactions")

    st = structuring(edges, fmt)
    st.insert(0, "dataset", args.dataset)
    st.to_csv(OUT / "structuring.csv", index=False)

    keep = edges["src"].to_numpy() != edges["dst"].to_numpy()
    src, dst = edges["src"].to_numpy()[keep], edges["dst"].to_numpy()[keep]
    time = edges["time"].to_numpy().astype(np.int64)[keep]
    flagged = edges["laundering"].to_numpy()[keep] == 1
    marked = set(np.union1d(src[flagged], dst[flagged]).tolist())
    universe = np.intersect1d(np.unique(src), np.unique(dst))
    base = len(marked.intersection(universe.tolist())) / len(universe)

    dwell = dwell_minutes(src, dst, time).reindex(universe).fillna(NO_DWELL)
    night = night_share(src, dst, time).reindex(universe)
    conserved = max_length_score(roles(rec / "accounts_theta_0.2.tsv"), universe)
    scores = {
        "dwell time": dict(zip(universe.tolist(), (-dwell).tolist())),
        "night share": dict(zip(universe.tolist(), night.tolist())),
        "conserved": conserved,
        "conserved then dwell": {v: conserved[v] * 1e9 - dwell[v] for v in universe.tolist()},
    }
    rows = [{"dataset": args.dataset, "method": name, "base_rate": base,
             **{f"precision_at_{k}": precision_at_k(sc, marked, k) for k in KS}} for name, sc in scores.items()]
    rank = pd.DataFrame(rows)
    rank.to_csv(OUT / "velocity_ranking.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"{args.dataset}: {len(universe)} accounts with incoming and outgoing transactions; base rate {base:.4f}; "
          f"{int(np.isin(universe, list(dwell[dwell < NO_DWELL].index)).sum())} with a dwell time")
    print("\nstructuring (US Dollar amounts, [9000, 10000) over [10000, 11000)):")
    print(st.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nranking:")
    print(rank.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    lr = st.set_index(["subset", "group"])["ratio"]
    print(f"  H30 here: laundering {lr[('every format', 'laundering')]:.3f} vs background "
          f"{lr[('every format', 'background')]:.3f}; holds: {lr[('every format', 'laundering')] >= 2 * lr[('every format', 'background')]}")
    p = rank.set_index("method")["precision_at_1000"]
    print(f"  H32 here: dwell {p['dwell time']:.4f} vs 2 x base {2 * base:.4f}; holds: {p['dwell time'] >= 2 * base}")
    print(f"  H33 here: night {p['night share']:.4f} vs 2 x base {2 * base:.4f}; holds: {p['night share'] >= 2 * base}")
    print(f"  H34 here: combined {p['conserved then dwell']:.4f} vs conserved {p['conserved']:.4f}; "
          f"holds: {p['conserved then dwell'] >= p['conserved'] + 0.10}")


if __name__ == "__main__":
    main()
