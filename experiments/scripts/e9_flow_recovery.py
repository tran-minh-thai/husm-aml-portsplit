"""E9-FLOW: recover AMLworld typologies with the u_flow measure .

e9_portsplit_recovery.py mines HUSM's u = CMNI x mean-bucket-utility, whose mean term is
confined to the bucket table's spread. u_flow(P) = CMNI(P) x B(P), where B(P) is the
best-funded occurrence's bottleneck edge amount (US Dollar), is ANTI-MONOTONE and needs NO
upper bound; the Java miner implements it as `--measure flow`. B(P) lies between the
smallest and largest edge amounts, so the same support-dominance ceiling applies with R the
spread of the bottlenecks: the measure does not escape the ceiling, it has a much wider R.
The question this script answers: does mining by bottleneck flow recover the
direction-defined typologies, and how often does the money term reorder patterns relative
to support?

Reuses e9's graph construction and occurrence-level recovery verbatim; only the mine
differs (weighted port-split graph + measure=flow). Deterministic: fixed quantile buckets,
deterministic miner, no seed.

Run:  python scripts/e9_flow_recovery.py --dataset HI-Small [--min-flow AUTO|<value>]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e9_portsplit_recovery import (  # noqa: E402
    DATA, OUT, JAR, account_signal, assign_bucket, fit_buckets,
    load_patterns, load_transactions, occurrence_recovery, truth_neighborhood,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inchusm_aml import miner_bridge                       # noqa: E402
from inchusm_aml.data.portsplit import port_split          # noqa: E402
from inchusm_aml.eval.score_decomposition import order_statistics, variance_decomposition  # noqa: E402


def flow_diagnostics(hus: pd.DataFrame) -> dict:
    """How much of u_flow's variance is the money term (bottleneck) vs support (CMNI).

    This is the FLOW analogue of e9's utility_diagnostics: u_flow = CMNI x bottleneck, so
    log u_flow = log CMNI + log bottleneck, decomposed with the covariance. The bottleneck lies
    in [min edge amount, max edge amount], so the same support-dominance ceiling applies with
    R = the spread of the bottlenecks; the measure does not escape it, it only has a wider R.
    """
    d = hus[(hus["cmni"] > 0) & (hus["bottleneck"] > 0)].copy()
    if len(d) < 3:
        return {"n": len(d), "money_share": float("nan"), "rho_flow_cmni": float("nan"),
                "reversed_pairs": float("nan"), "order_violations": 0}
    dec = variance_decomposition(d["cmni"], d["bottleneck"])
    R = float(d["bottleneck"].max() / d["bottleneck"].min())
    order = order_statistics(d["cmni"], d["bottleneck"], R)
    if order["violations"]:
        raise RuntimeError(f"{order['violations']} pair(s) locked by the ceiling yet reversed: "
                           f"the score is not CMNI x bottleneck. Do not use this mine.")
    return {
        "n": len(d),
        "money_share": dec["share_weight"],
        "cov_log": dec["cov_log"],
        "ratio_money_no_cov": dec["ratio_weight_no_cov"],
        "R_bottleneck": R,
        "locked_pairs": order["locked_frac"],
        "reversed_pairs": order["reversed_frac"],
        "order_violations": order["violations"],
        "rho_flow_cmni": float(d["flow"].corr(d["cmni"], method="spearman")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HI-Small")
    ap.add_argument("--mode", default="truth-neighborhood",
                    choices=["truth-neighborhood", "full"])
    ap.add_argument("--hub-caps", type=int, nargs="+", default=[0, 200, 50])
    ap.add_argument("--mer", type=int, default=2)
    ap.add_argument("--max-edges", type=int, default=3)
    ap.add_argument("--min-flow", default="AUTO",
                    help="flow threshold, or AUTO to pick the P70 of single-edge flows")
    ap.add_argument("--heap", default="8g")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading AMLworld {args.dataset}...", flush=True)
    edges, code = load_transactions(DATA / f"{args.dataset}_Trans.csv")
    attempts = load_patterns(DATA / f"{args.dataset}_Patterns.txt", code)
    if args.mode == "truth-neighborhood":
        edges = truth_neighborhood(edges, attempts)
        print(f"  truth-neighborhood: {len(edges):,} transactions", flush=True)
    else:
        # full graph: the bottleneck branch-and-bound over ~1M vertices is the unknown cost
        # this run exists to measure; a timeout is recorded, not a crash.
        print(f"  FULL graph: {len(edges):,} transactions (bottleneck cost unmeasured at this scale)",
              flush=True)

    # labels: same quantile buckets as e9 (structure identical); weights: raw amount
    signal = account_signal(edges, scale="log")           # buckets only; weights carry the money
    bucket_edges, _ = fit_buckets(signal, 4)
    labels = assign_bucket(signal, bucket_edges)

    rows = []
    for cap in args.hub_caps:
        if cap > 0:
            deg = pd.concat([edges["src"], edges["dst"]]).value_counts()
            hot = set(deg[deg > cap].index)
            ecap = edges[~(edges["src"].isin(hot) | edges["dst"].isin(hot))].reset_index(drop=True)
        else:
            ecap = edges
        lab_cap = labels.loc[list(set(ecap["src"]) | set(ecap["dst"]))]
        present = [a for a in attempts if a.accounts <= (set(ecap["src"]) | set(ecap["dst"]))]

        node_labels, split_edges = port_split(ecap, lab_cap, weighted=True)
        tag = f"{args.dataset}_{args.mode}_flow_cap{cap}_me{args.max_edges}"
        lg = OUT / f"e9_{tag}.lg"
        tsv = OUT / f"e9_{tag}.util.tsv"
        miner_bridge.export_lg(node_labels, split_edges, lg)
        tsv.write_text("0 1.0\n", encoding="ascii")        # unused by FLOW, keeps the CLI happy

        if str(args.min_flow).upper() == "AUTO":
            w = split_edges["weight"].to_numpy()
            min_flow = float(np.quantile(w[w > 0], 0.70))  # ~ a high-money 1-edge pattern
        else:
            min_flow = float(args.min_flow)

        print(f"  [{tag}] {len(node_labels):,} vertices, {len(split_edges):,} edges; "
              f"mining measure=flow, min-flow={min_flow:,.0f}...", flush=True)
        ts = time.time()
        try:
            hus, _ = miner_bridge.run_husm(lg, tsv, min_util=min_flow, mer=args.mer,
                                           max_edges=args.max_edges, measure="flow",
                                           out_jsonl=OUT / f"e9_{tag}.hus.jsonl",
                                           heap=args.heap, timeout_s=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"  [{tag}] TIMED OUT after {args.timeout}s -- bottleneck search too costly "
                  f"at this scale; recorded as timed_out.", flush=True)
            types = sorted({a.type for a in attempts})
            tbl = pd.DataFrame({"type": types, "recovery_rate": float("nan"), "timed_out": 1,
                                "hub_cap": cap, "min_flow": min_flow, "n_patterns": 0,
                                "money_share": float("nan"), "attempts_present": len(present)})
            rows.append(tbl)
            continue
        diag = flow_diagnostics(hus)
        print(f"  [{tag}] {len(hus)} patterns ({time.time()-ts:.0f}s); "
              f"money share {diag['money_share']:.1%} (Shapley), "
              f"{diag['reversed_pairs']:.1%} pairs reversed, "
              f"rho(flow,CMNI)={diag['rho_flow_cmni']:.3f}", flush=True)

        codes = list(hus.code) if len(hus) else []
        if not codes:
            print(f"  [{tag}] no patterns at this threshold -- lower --min-flow")
        tbl = occurrence_recovery(attempts, codes, lab_cap)
        tbl["timed_out"] = 0
        tbl["hub_cap"] = cap
        tbl["min_flow"] = min_flow
        tbl["n_patterns"] = len(hus)
        tbl["money_share"] = diag["money_share"]
        tbl["attempts_present"] = len(present)
        rows.append(tbl)
        print(tbl[["type", "recovery_rate"]].to_string(index=False))

    out = pd.concat(rows, ignore_index=True)
    dst = OUT / f"e9_flow_recovery_{args.dataset}_{args.mode}.csv"
    out.to_csv(dst, index=False)
    print(f"\nDone ({time.time()-t0:.0f}s) -> {dst}")


if __name__ == "__main__":
    main()
