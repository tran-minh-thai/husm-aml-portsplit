"""E3+ : pattern-recovery over all EIGHT AMLworld typologies.

The first paper recovered five typologies (fan-out, fan-in, gather-scatter, scatter-gather,
cycle). This driver adds the remaining three (bipartite, stack, random) via motif_ext, so
recovery is measured over the whole AMLworld pattern set. Recovery is set-based and
encoding-independent: an attempt is recovered when a mined structure of the same family
shares at least min(3, |attempt|) accounts with it.

Run:  python scripts/e3plus_recovery8.py --dataset HI-Small [--dt 7]
Output: results/e3plus_recovery8_<dataset>.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inchusm_aml.data.amlworld import load_patterns, load_transactions  # noqa: E402
from inchusm_aml.baselines.motif_join import (  # noqa: E402
    scatter_gather_pairs,
    temporal_cycles,
    windowed_fan,
    windowed_gather_scatter,
)
from inchusm_aml.baselines.motif_ext import (  # noqa: E402
    bipartite_cores,
    random_paths,
    stack_chains,
)
from inchusm_aml.eval.pattern_match import recovery_table  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
DATA = Path(__file__).resolve().parents[1].parent / "datasets" / "amlworld"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HI-Small", help="e.g. HI-Small, LI-Small, HI-Medium")
    ap.add_argument("--dt", type=int, default=7, help="window in days")
    ap.add_argument("--data-dir", default=str(DATA))
    # search budget for the path miners (random / stack). The defaults suit Small; on Medium the
    # graph is ~6x larger and the budget runs out early, which is what depresses RANDOM/STACK
    # recovery, so raise it there (e.g. --global-cap 5000000).
    ap.add_argument("--hub-cap", type=int, default=50, help="degree cap for the path miners")
    ap.add_argument("--per-start-cap", type=int, default=20_000, help="DFS budget per start node")
    ap.add_argument("--global-cap", type=int, default=500_000, help="total structures to collect")
    ap.add_argument("--min-side", type=int, default=2, help="bipartite: min accounts per side")
    args = ap.parse_args()
    dt = args.dt * 1440   # minutes

    t0 = time.time()
    root = Path(args.data_dir)
    print(f"Loading AMLworld {args.dataset} (dT={args.dt}d)...", flush=True)
    edges, code = load_transactions(root / f"{args.dataset}_Trans.csv")
    attempts = load_patterns(root / f"{args.dataset}_Patterns.txt", code)
    print(f"  {len(edges):,} transactions, {len(attempts)} ground-truth attempts", flush=True)

    # five first-paper typologies
    fan_out = set(windowed_fan(edges, "out", dt)["center"])
    fan_in = set(windowed_fan(edges, "in", dt)["center"])
    gs = set(windowed_gather_scatter(edges, dt)["center"])
    sgp = scatter_gather_pairs(edges, dt)
    sg_pairs = set(zip(sgp["u"], sgp["w"]))
    cycles = temporal_cycles(edges, dt)

    # three journal typologies
    print(f"  mining bipartite / stack / random (hub_cap={args.hub_cap}, "
          f"global_cap={args.global_cap:,})...", flush=True)
    budget = dict(hub_cap=args.hub_cap, per_start_cap=args.per_start_cap,
                  global_cap=args.global_cap)
    rnd = random_paths(edges, dt, **budget)
    stk = stack_chains(edges, dt, **budget)
    bip = bipartite_cores(edges, dt, min_side=args.min_side)
    print(f"  mined structures: random={len(rnd)} stack={len(stk)} bipartite={len(bip)}",
          flush=True)

    table = recovery_table(
        attempts, fan_out, fan_in, gs, sg_pairs, cycles,
        random_structs=rnd, bipartite_structs=bip, stack_structs=stk,
    )
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"e3plus_recovery8_{args.dataset}.csv"
    table.to_csv(out, index=False)
    print("\nRecovery over eight typologies:")
    print(table.to_string(index=False))
    print(f"\nDone ({time.time() - t0:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
