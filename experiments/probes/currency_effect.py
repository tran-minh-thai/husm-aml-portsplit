"""What converting AMLworld amounts to US Dollar changes in the typology-recovery input.

The recovery experiment labels every account by the quantile bucket of its total amount
(paid plus received) and gives each bucket the mean amount as its utility. Before the
conversion, amounts were summed across 15 currencies whose typical sizes differ by about six
orders of magnitude. This probe rebuilds the same input both ways, with the experiment's own
functions, and reports:

  * the utility table and its spread R, on the raw and on the log scale;
  * how many accounts change bucket (a changed bucket is a changed vertex label, so the graph
    the miner sees changes), overall and inside each typology's attempts;
  * how strongly the bucket encodes the account's currency before and after: the share of
    accounts in each bucket whose most-used currency is each currency, summarized by the
    normalized mutual information between bucket and currency.

It mines nothing; every number is a property of the data and the bucketing.

Run: python experiments/probes/currency_effect.py --dataset HI-Small [--n-buckets 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

from e9_portsplit_recovery import (  # noqa: E402
    DATA, account_signal, fit_buckets, load_patterns, load_transactions, truth_neighborhood,
)
from inchusm_aml.utility.monetary import assign_bucket  # noqa: E402


def dominant_currency(edges: pd.DataFrame) -> pd.Series:
    """Most frequent payment currency over every transaction an account takes part in."""
    both = pd.concat([edges[["src", "currency"]].rename(columns={"src": "acct"}),
                      edges[["dst", "currency"]].rename(columns={"dst": "acct"})])
    counts = both.groupby(["acct", "currency"]).size().reset_index(name="n")
    counts = counts.sort_values(["acct", "n", "currency"], ascending=[True, False, True])
    return counts.drop_duplicates("acct").set_index("acct")["currency"]


def normalized_mi(a: pd.Series, b: pd.Series) -> float:
    """Mutual information of two label series divided by the entropy of b (share of b's
    uncertainty explained by a): 0 = independent, 1 = a determines b."""
    joint = pd.crosstab(a, b).to_numpy(dtype=float)
    p = joint / joint.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    return mi / hb if hb > 0 else float("nan")


def bucketing(edges: pd.DataFrame, scale: str, n_buckets: int):
    signal = account_signal(edges, scale=scale)
    cuts, util = fit_buckets(signal, n_buckets)
    return assign_bucket(signal, cuts), util


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HI-Small")
    ap.add_argument("--n-buckets", type=int, default=4)
    args = ap.parse_args()

    raw_edges, code = load_transactions(DATA / f"{args.dataset}_Trans.csv", currency="raw")
    usd_edges, code_u = load_transactions(DATA / f"{args.dataset}_Trans.csv", currency="usd")
    assert code == code_u, "the two loads encoded accounts differently"
    attempts = load_patterns(DATA / f"{args.dataset}_Patterns.txt", code)
    raw_nb = truth_neighborhood(raw_edges, attempts)
    usd_nb = truth_neighborhood(usd_edges, attempts)
    assert raw_nb[["src", "dst", "time"]].equals(usd_nb[["src", "dst", "time"]]), \
        "conversion changed the graph structure"
    names = usd_edges.attrs["currencies"]
    print(f"{args.dataset}: truth neighborhood {len(usd_nb):,} transactions, "
          f"{len(pd.unique(usd_nb[['src', 'dst']].to_numpy().ravel())):,} accounts, "
          f"{args.n_buckets} buckets")

    cur = dominant_currency(usd_nb).map(lambda i: names[i])
    for scale in ("raw", "log"):
        lab_r, util_r = bucketing(raw_nb, scale, args.n_buckets)
        lab_u, util_u = bucketing(usd_nb, scale, args.n_buckets)
        r_r = max(util_r.values()) / min(util_r.values())
        r_u = max(util_u.values()) / min(util_u.values())
        print(f"\n[{scale} scale] utility table (bucket: mean signal)")
        print(f"   currencies mixed : " + ", ".join(f"{b}:{v:.4g}" for b, v in util_r.items())
              + f"   R = {r_r:.4g}")
        print(f"   US Dollar        : " + ", ".join(f"{b}:{v:.4g}" for b, v in util_u.items())
              + f"   R = {r_u:.4g}")
        if scale == "log":
            continue                          # log is monotone: same buckets as raw
        common = lab_r.index.intersection(lab_u.index)
        moved = (lab_r[common] != lab_u[common])
        print(f"   accounts changing bucket: {int(moved.sum()):,} / {len(common):,} "
              f"({moved.mean():.1%})")
        for t in sorted({a.type for a in attempts}):
            accts = [x for a in attempts if a.type == t for x in a.accounts if x in moved.index]
            m = moved[accts]
            print(f"     {t:15s} attempt accounts changing bucket: {int(m.sum()):>4} / {len(m):>4} "
                  f"({m.mean():.1%})")
        idx = cur.index.intersection(common)
        print(f"   share of currency uncertainty explained by bucket (normalized MI): "
              f"mixed {normalized_mi(lab_r[idx], cur[idx]):.3f}, US Dollar "
              f"{normalized_mi(lab_u[idx], cur[idx]):.3f}")
        for tag, lab in (("currencies mixed", lab_r), ("US Dollar", lab_u)):
            tab = pd.crosstab(lab[idx], cur[idx], normalize="index")
            top = tab.apply(lambda row: ", ".join(f"{c} {v:.0%}" for c, v in
                                                  row.sort_values(ascending=False).head(3).items()),
                            axis=1)
            print(f"   most-used currency by bucket ({tag}):")
            for b, s in top.items():
                print(f"     bucket {b}: {s}")


if __name__ == "__main__":
    main()
