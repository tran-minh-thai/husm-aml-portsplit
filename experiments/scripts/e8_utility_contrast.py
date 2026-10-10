"""E8: is the utility dimension active, or is high-utility mining measuring frequency?

HUSM scores a pattern with u(P) = CMNI(P) * mean(external utility of P's vertex labels). The
second factor is a MEAN, so it is bounded by the extremes of the utility table: for every
pattern, e_min <= ubar(P) <= e_max. Writing R = e_max / e_min,

    u(P)/u(Q) = (CMNI(P)/CMNI(Q)) * (ubar(P)/ubar(Q))  and  ubar(P)/ubar(Q) in [1/R, R],

so whenever CMNI(P)/CMNI(Q) > R the order of P and Q is FORCED by support alone -- no
assignment of utilities can reverse it. R is a hard ceiling on how much any utility table can
matter, and it is a property of the table alone, not of the algorithm or of the data.

That ceiling is low wherever the utilities are drawn from a Gaussian, which is what HUSM
(Inf. Sci. 2024) does on four of its six benchmark graphs -- it draws N(3,1) for Citeseer,
LastFM, Gnutella and Mico, and takes user ratings on the other two. Gaussian extremes grow only
like sigma*sqrt(2 ln k) in the number of labels k, so spanning HUSM's whole range of k, 6 to 29,
moves the median R only from 2.4 to 5.0. Money is different in kind -- it is heavy-tailed -- and
the frozen bucket table of this paper spans R = 4456.

SCOPE. The proposition uses exactly one property of the scoring rule: the utility factor is a
MEAN, hence trapped in [e_min, e_max]. It therefore covers HUSM's utility semantics and any
measure of the form "support times a bounded combination of weights"; a sum instead of a mean
only multiplies the effective R by the pattern-size ratio (N(3,1) reaches 6.6 rather than 2.2,
still far below a real transaction graph's support spread). It says NOTHING about measures of a
different shape -- MaxMin/AveMin weighting uses extremes rather than a mean, and UFTM's utility
sums edge weights maximised over embeddings across the graphs containing a free tree, in a
multi-graph database. This script makes no claim about those.

This does NOT say the utility dimension is useless. The proposition says it is inert exactly
when R is small relative to the support spread, and HUSM's own case study is the other side of
that coin -- but note how it is built. On a 136-vertex Amazon extract it assigns utilities by
RANK OF FREQUENCY (categories sorted by descending count receive 0.3, 0.6, ... 4.8: R = 16, and
rank correlation between utility and count = -0.99), stating the aim as "to enhance the
contrast", and then reports 137 high-utility subgraphs against 113 frequent ones. That confirms
the proposition rather than contradicting it: both conditions the utility dimension needs, a
wide range and a utility that does not track support, are satisfied BY CONSTRUCTION.

That construction appears nowhere else in the paper, and the comparison against frequent mining
is never run on any of the six benchmark datasets -- their experiments compare the algorithm's
own three upper bounds against each other on candidates, runtime and memory. This script runs
the missing measurement, on those datasets under that protocol.

Two arms measure the consequence:

  CITESEER, the primary dataset of HUSM, under the utility protocol of HUSM. The same graph is
  mined twice at the same threshold: once with Gaussian utilities, once with a CONSTANT table.
  A constant table makes u = c * CMNI, so the second run IS frequent subgraph mining under
  CMNI support, sharing the engine and the threshold. The overlap of the two answers measures
  what the utility dimension contributes to the published evaluation. HUSM does not publish its
  utility seed, so the arm sweeps ten seeds and reports the median; the constant of each seed
  is that seed's OWN table mean, so both runs sit on the same scale and share a threshold that
  means the same thing. Seed 42 alone would understate the utility factor -- it gives the
  lowest variance share of the ten.

  ELLIPTIC++ actors, the same contrast with real monetary utility, via a re-scoring of the
  mined patterns under both tables. Re-scoring is exact rather than approximate: u is
  determined by (CMNI, vertex labels), so swapping the weights needs no re-mining, and it
  isolates the weights from the search by holding the pattern set fixed. --skip-mine keeps only
  this part, which is what the utility-contrast table reports.

Note on what the numbers do and do not say. The forced-pair share is computed for each graph
from ITS OWN CMNI distribution -- a protocol's R applied to another graph's support spread
would say nothing. It is also a LOWER bound: R over the labels that actually occur is at most R
over the table, and a smaller R forces more pairs.

Requires e5a (for the Elliptic++ graph) and datasets/citeseer/citeseer.lg:
    python scripts/e5a_ellipticpp_utility.py --min-util 100
    python scripts/e8_utility_contrast.py
Output: results/e8_utility_contrast.csv, results/e8_citeseer_seeds.csv
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inchusm_aml import miner_bridge  # noqa: E402
from inchusm_aml.eval.score_decomposition import order_statistics, variance_decomposition  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parents[1] / "results"
# Inputs produced by earlier steps (e5a) are read from RESULTS. Everything this script writes goes
# to OUT: the record directory when run through experiments/provenance/run_and_record.py, so a
# new run never overwrites a committed artifact; RESULTS otherwise.
OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else RESULTS
CITESEER = ROOT / "datasets" / "citeseer" / "citeseer.lg"

GAUSS_MU, GAUSS_SIGMA = 3.0, 1.0        # the protocol HUSM uses on four of its six datasets
SEEDS = [1, 2, 3, 7, 13, 42, 99, 123, 2024, 31337]
TUPLE = re.compile(r"\((-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+)\)")


class JavaRandom:
    """java.util.Random, bit-exact: same LCG and the same polar-method nextGaussian.

    The miner draws its Gaussian utilities with java.util.Random, so re-scoring its patterns in
    Python needs the same stream, not merely the same distribution. Reproducing the generator
    keeps the tables derived rather than hardcoded; check_table asserts the result against the
    miner's own utility column, so a divergence fails loudly.
    """

    def __init__(self, seed: int) -> None:
        self.s = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
        self.spare: float | None = None

    def _next(self, bits: int) -> int:
        self.s = (self.s * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return self.s >> (48 - bits)

    def next_double(self) -> float:
        return ((self._next(26) << 27) + self._next(27)) / float(1 << 53)

    def next_gaussian(self) -> float:
        if self.spare is not None:
            v, self.spare = self.spare, None
            return v
        while True:
            v1, v2 = 2 * self.next_double() - 1, 2 * self.next_double() - 1
            s = v1 * v1 + v2 * v2
            if 0 < s < 1:
                break
        m = math.sqrt(-2 * math.log(s) / s)
        self.spare = v2 * m
        return v1 * m


def gaussian_table(seed: int, n_labels: int,
                   mu: float = GAUSS_MU, sigma: float = GAUSS_SIGMA) -> dict[int, float]:
    """GraphLoader.gaussianUtilities: max(0.01, mu + sigma*Z) per label, ascending label order."""
    r = JavaRandom(seed)
    return {l: max(0.01, mu + sigma * r.next_gaussian()) for l in range(n_labels)}


def vertex_labels(code: str) -> list[int]:
    """Vertex labels of a pattern, read off its DFS code: tuple (i, j, li, lij, lj)."""
    lab: dict[int, int] = {}
    for i, j, li, _lij, lj in (tuple(map(int, m.groups())) for m in TUPLE.finditer(code)):
        lab[i], lab[j] = li, lj
    return [lab[v] for v in sorted(lab)]


def rescore(frame: pd.DataFrame, table: dict[int, float]) -> pd.Series:
    """u(P) = CMNI(P) * mean(external utility of P's vertex labels), HUSM Def. 8."""
    labels = frame.code.map(vertex_labels)
    return frame.cmni * labels.map(lambda L: float(np.mean([table[x] for x in L])))


def check_table(frame: pd.DataFrame, table: dict[int, float], what: str) -> None:
    """Assert a Python-side utility table reproduces the miner's own utility column.

    The tolerance is RELATIVE, and it is set by the miner's own output precision rather than by
    taste. An absolute tolerance would be wrong because u = CMNI * ubar scales any table error
    by CMNI, which reaches 167830 on the actor graph. The floor is the miner writing utility as
    %.6f: at the smallest reported utility (50 on Citeseer) that rounding alone is ~1e-8
    relative, so anything tighter fails on arithmetic noise. 1e-6 leaves two orders of margin
    and still catches a genuinely wrong table, which would be off by percent.
    """
    err = float(((rescore(frame, table) - frame.utility).abs() / frame.utility).max())
    if err >= 1e-6:
        raise AssertionError(
            f"{what}: the Python table disagrees with the miner by {err:.3g} (relative). Either "
            f"JavaRandom no longer matches java.util.Random or the utility semantics changed; "
            f"do not trust any number this script prints until it is resolved.")


def forced_fraction(cmni: np.ndarray, R: float) -> float:
    """Share of pattern pairs whose order no utility table can change (CMNI ratio > R)."""
    i, j = np.triu_indices(len(cmni), k=1)
    return float((np.maximum(cmni[i], cmni[j]) / np.minimum(cmni[i], cmni[j]) > R).mean())


def stats_of(u: pd.Series, cmni: pd.Series, code: pd.Series, R: float) -> dict:
    ubar = u / cmni
    return {
        "R": R, "patterns": len(u),
        "cmni_span": float(cmni.max() / cmni.min()),
        "ubar_span": float(ubar.max() / ubar.min()),
        "forced_pairs": forced_fraction(cmni.to_numpy(float), R),
        # log u = log CMNI + log ubar: which factor carries the score's variance? Reported with
        # the covariance (see inchusm_aml.eval.score_decomposition); the earlier column
        # var_share_ubar = Var(log ubar)/Var(log u) ignored it and is no longer written.
        **{f"{k}_ubar" if k in ("share_weight", "ratio_weight_no_cov") else k: v
           for k, v in variance_decomposition(cmni, ubar).items()},
        # pairs the utility term actually reverses relative to support, against the pairs the
        # ceiling locks; "order_violations" must be 0 (the support-dominance ceiling checked on this set)
        **{f"order_{k}": v for k, v in order_statistics(cmni, ubar, R).items()},
        "spearman_u_cmni": float(spearmanr(u, cmni).statistic),
        "kendall_u_cmni": float(kendalltau(u, cmni).statistic),
        "top10_vs_cmni": len(set(code[u.nlargest(10).index]) & set(code[cmni.nlargest(10).index])),
    }


def top_n(frame: pd.DataFrame, n: int) -> set[str]:
    return set(frame.nlargest(n, "utility").code)


def citeseer_arm(min_util: float, bound: str, max_edges: int) -> pd.DataFrame:
    """HUSM's own dataset under HUSM's own protocol, against plain frequent mining."""
    if not CITESEER.exists():
        print(f"  skipped: {CITESEER} not present (see docs/reproduce/paper3.md)")
        return pd.DataFrame()
    print("=== Citeseer: the dataset and the utility protocol of HUSM (Inf. Sci. 2024) ===")
    print(f"  {len(SEEDS)} seeds; the constant arm of each seed uses that seed's own table mean,")
    print("  so both runs share a scale and the shared threshold means the same thing.")
    kw = dict(min_util=min_util, bound=bound, max_edges=max_edges)
    n_labels = 1 + max(int(ln.split()[2]) for ln in CITESEER.read_text().splitlines()
                       if ln.startswith("v "))
    rows = []
    for seed in SEEDS:
        table = gaussian_table(seed, n_labels)
        g, _ = miner_bridge.run_husm(CITESEER, CITESEER, util_gauss=f"{GAUSS_MU},{GAUSS_SIGMA},{seed}",
                                     out_jsonl=OUT / f"e8_citeseer_gauss_{seed}.jsonl", **kw)
        check_table(g, table, f"Citeseer Gaussian seed={seed}")
        const_tsv = OUT / f"e8_citeseer_const_{seed}.util.tsv"
        mean = float(np.mean(list(table.values())))
        # same "label value" layout export_utilities writes
        const_tsv.write_text("".join(f"{l} {mean:.12f}\n" for l in range(n_labels)),
                             encoding="ascii")
        k, _ = miner_bridge.run_husm(CITESEER, const_tsv,
                                     out_jsonl=OUT / f"e8_citeseer_const_{seed}.jsonl", **kw)
        G, K = set(g.code), set(k.code)
        row = stats_of(g.utility, g.cmni, g.code, max(table.values()) / min(table.values()))
        row.update({"seed": seed, "fsm_patterns": len(K), "shared_vs_fsm": len(G & K),
                    "jaccard_vs_fsm": len(G & K) / len(G | K),
                    "top20_vs_fsm": len(top_n(g, 20) & top_n(k, 20))})
        rows.append(row)
        print(f"  seed {seed:>5}: HUSM {len(G):>3} vs frequent {len(K):>3} patterns, "
              f"shared {len(G & K):>3}, Jaccard {row['jaccard_vs_fsm']:.3f}, "
              f"utility share {row['share_weight_ubar']:.1%} (cov {row['cov_log']:+.3f})")
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "e8_citeseer_seeds.csv", index=False)
    med = per_seed.median(numeric_only=True)
    # .1f, not .0f: the top-20 median is 17.5 and rounding it to 18 would overstate the
    # overlap, which is the direction that flatters this script's own conclusion.
    print(f"  median over {len(SEEDS)} seeds: Jaccard vs frequent mining "
          f"{med.jaccard_vs_fsm:.3f} (range {per_seed.jaccard_vs_fsm.min():.3f}"
          f"-{per_seed.jaccard_vs_fsm.max():.3f}), top-20 {med.top20_vs_fsm:.1f}/20, "
          f"utility share {med.share_weight_ubar:.1%} "
          f"({per_seed.share_weight_ubar.min():.1%}-{per_seed.share_weight_ubar.max():.1%})")
    print("  -> the published protocol returns nearly the answer that frequency alone gives")
    out = med.to_dict()
    out["arm"] = "citeseer/gaussian"
    return pd.DataFrame([out])


def ellipticpp_arm(gauss_minutil: float | None, bound: str, max_edges: int,
                   skip_mine: bool) -> pd.DataFrame:
    hus_csv, graph_lg = RESULTS / "e5a_ellipticpp_hus.csv", RESULTS / "e5a_ellipticpp.lg"
    util_tsv = RESULTS / "e5a_ellipticpp.util.tsv"
    # the graph itself is read only by the mining arms; re-scoring needs the pattern set and table
    for p in (hus_csv, util_tsv) + (() if skip_mine else (graph_lg,)):
        if not p.exists():
            raise SystemExit(f"missing {p} -- run scripts/e5a_ellipticpp_utility.py first")
    # export_utilities writes "label value" separated by a SPACE despite the .tsv name, and a
    # file produced on Windows carries CRLF, so split on any whitespace rather than on a tab.
    mon_table = {int(ln.split()[0]): float(ln.split()[1])
                 for ln in util_tsv.read_text().splitlines() if ln.strip()}
    gau_table = gaussian_table(42, len(mon_table))
    R_mon = max(mon_table.values()) / min(mon_table.values())
    R_gau = max(gau_table.values()) / min(gau_table.values())

    print("\n=== Elliptic++ actors: the same question with real money ===")
    d = pd.read_csv(hus_csv)
    check_table(d, mon_table, "Elliptic++ monetary")
    u_mon, u_gau = rescore(d, mon_table), rescore(d, gau_table)
    print(f"  the same {len(d)} patterns, weighted two ways (re-scoring is exact):")
    rows = []
    for arm, u, R in [("ellipticpp/monetary", u_mon, R_mon),
                      ("ellipticpp/gaussian (re-scored)", u_gau, R_gau)]:
        r = stats_of(u, d.cmni, d.code, R)
        r["arm"] = arm
        rows.append(r)
        print(f"  {arm:32} R={R:8.1f}  forced {r['forced_pairs']:5.1%}  "
              f"utility share {r['share_weight_ubar']:5.1%} (cov {r['cov_log']:+.3f})  "
              f"reversed {r['order_reversed_frac']:5.1%}  violations {r['order_violations']}  "
              f"rho={r['spearman_u_cmni']:.3f}")

    if not skip_mine:
        # The re-scored set was SELECTED by monetary utility. Mining the other tables for real
        # removes that bias: each arm then chooses its own patterns.
        mean_gau = float(np.mean(list(gau_table.values())))
        theta = gauss_minutil or 3000.0
        print(f"\n  mining the Gaussian and constant arms for real at minutil {theta:g} "
              f"(slow: this graph takes ~15 min per run)", flush=True)
        g, _ = miner_bridge.run_husm(graph_lg, util_tsv, min_util=theta,
                                     util_gauss=f"{GAUSS_MU},{GAUSS_SIGMA},42",
                                     out_jsonl=OUT / "e8_ellipticpp_gauss.jsonl",
                                     bound=bound, max_edges=max_edges)
        check_table(g, gau_table, "Elliptic++ Gaussian")
        const_tsv = OUT / "e8_ellipticpp_const.util.tsv"
        const_tsv.write_text("".join(f"{l} {mean_gau:.12f}\n" for l in sorted(gau_table)),
                             encoding="ascii")
        k, _ = miner_bridge.run_husm(graph_lg, const_tsv, min_util=theta,
                                     out_jsonl=OUT / "e8_ellipticpp_const.jsonl",
                                     bound=bound, max_edges=max_edges)
        mon = pd.DataFrame({"code": d.code, "utility": u_mon, "cmni": d.cmni})
        n = min(len(g), len(k), len(mon))
        print(f"  Gaussian {len(g)} patterns, frequent {len(k)}, monetary {len(mon)}; "
              f"comparing the top {n} of each")
        for r, arm in [(rows[1], g), (rows[0], mon)]:
            inter = len(top_n(arm, n) & top_n(k, n))
            r["topn_vs_fsm"], r["topn"] = inter, n
            print(f"  {r['arm']:32} shares {inter}/{n} of its top patterns with frequent mining")
        gc = g.cmni.to_numpy(float)
        print(f"  on the Gaussian arm's OWN set (no monetary selection): forced "
              f"{forced_fraction(gc, R_gau):.1%}, rho={spearmanr(g.utility, g.cmni).statistic:.3f}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--citeseer-minutil", type=float, default=50.0)
    ap.add_argument("--gauss-minutil", type=float, default=None)
    ap.add_argument("--skip-mine", action="store_true",
                    help="skip the slow Elliptic++ mining arms; keeps the exact re-scoring")
    ap.add_argument("--max-edges", type=int, default=5)
    ap.add_argument("--bound", default="ceu", choices=["ivu", "eu", "ceu"])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cite = citeseer_arm(args.citeseer_minutil, args.bound, args.max_edges)
    ell = ellipticpp_arm(args.gauss_minutil, args.bound, args.max_edges, args.skip_mine)
    cols = ["arm", "R", "patterns", "cmni_span", "forced_pairs", "var_log_support",
            "var_log_weight", "cov_log", "var_log_score", "share_weight_ubar", "share_support",
            "ratio_weight_no_cov_ubar",
            "spearman_u_cmni", "jaccard_vs_fsm", "top20_vs_fsm"]
    stats = pd.concat([cite, ell], ignore_index=True)
    stats = stats[[c for c in cols if c in stats.columns]
                  + [c for c in stats.columns if c not in cols]]

    print("\n" + "=" * 78)
    print("Does the utility dimension carry the answer, or does support?")
    print("=" * 78)
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  share_weight_ubar: Shapley share of Var(log u) carried by the utility factor,")
    print("                     (Var(log ubar) + Cov(log CMNI, log ubar)) / Var(log u).")
    print("  forced_pairs:   share of pairs no utility table could reorder (this graph's CMNI).")
    print("  jaccard_vs_fsm: overlap with the answer that frequent subgraph mining returns.")

    out = OUT / "e8_utility_contrast.csv"
    stats.to_csv(out, index=False)
    print(f"\nDone -> {out}")


if __name__ == "__main__":
    main()
