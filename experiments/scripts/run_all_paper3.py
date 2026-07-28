"""Run all experiments in order, with wall-clock timing and a PASS/FAIL summary.

Cross-platform (any shell with Python). Order: E5a -> E5b -> E5c -> E6 -> E7 -> E8 -> E3+ -> E4+.
A failing step does NOT stop the following steps; each step's output also goes to the console.
The Java engine jar is built first if missing (needs Maven + JDK 17); the E3+/E4+ steps are
pure Python and do not need the jar.

Usage:
  python scripts/run_all_paper3.py                 # Small everywhere (default)
  python scripts/run_all_paper3.py --include-medium # add AMLworld Medium for E3+
  python scripts/run_all_paper3.py --skip e4        # skip a step (e5a/e5b/e5c/e6/e7/e8/e3/e4)
Log: results/run_all_paper3_log.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
MINER = SCRIPTS.parents[1] / "inchusm-miner"
JAR = MINER / "target" / "inchusm-miner-0.1.0-SNAPSHOT.jar"
RESULTS = SCRIPTS.parent / "results"


def ensure_jar() -> bool:
    """Build the Java engine if the jar is missing; return True if it is available."""
    if JAR.exists():
        return True
    print(f"[build] jar missing, running: mvn -f {MINER / 'pom.xml'} package", flush=True)
    try:
        r = subprocess.run(["mvn", "-f", str(MINER / "pom.xml"), "package", "-q"])
        return r.returncode == 0 and JAR.exists()
    except FileNotFoundError:
        print("[build] Maven not found. Build the jar manually (needs JDK 17):", flush=True)
        print(f"        mvn -f {MINER / 'pom.xml'} package", flush=True)
        return False


def check_datasets() -> None:
    """Warn (do not fail) when the datasets the steps read are not placed under datasets/.

    These commands do NOT download data; place it under datasets/ first (docs/reproduce/paper3.md).
    """
    root = SCRIPTS.parents[1] / "datasets"
    need = {
        "ellipticpp/AddrAddr_edgelist.csv": "E5a/b/c",
        "ellipticpp/wallets_features.csv": "E5a/b/c",
        "amlworld/HI-Small_Trans.csv": "E3+",
        "elliptic1/elliptic_txs_features.csv": "E4+ and E6",
        "elliptic1/elliptic_txs_edgelist.csv": "E6",
        "citeseer/citeseer.lg": "E8 (the Citeseer arm is skipped without it)",
    }
    missing = [f"{p} (needed by {who})" for p, who in need.items() if not (root / p).exists()]
    if missing:
        print("[data] MISSING datasets under datasets/ (see docs/reproduce/paper3.md):", flush=True)
        for m in missing:
            print(f"       - {m}", flush=True)
        print("[data] steps needing them will fail; the others still run.\n", flush=True)


def steps(args) -> list[tuple[str, list[str]]]:
    # A larger path-search budget for the bigger graphs: the default runs out early and
    # depresses RANDOM/STACK recovery. HI-Small (smaller graph) recovers fully at the default,
    # but LI-Small (~6.9M tx) and both Medium graphs need this to reach full recovery.
    big = ["--global-cap", "5000000", "--per-start-cap", "50000"]
    plan: list[tuple[str, list[str]]] = [
        # thresholds calibrated from the first full run: AUTO gave 765 HUS (median utility 20);
        # 100 keeps the 151 strongest patterns and keeps the incremental state small enough
        # that the per-snapshot cost stays flat.
        ("E5a monetary-utility mining",
         ["e5a_ellipticpp_utility.py", "--n-buckets", "4", "--hub-cap", "50", "--min-util", "100"]),
        ("E5b BHUS-incremental",
         ["e5b_bhus_incremental.py", "--hub-cap", "50", "--min-value", "100"]),
        ("E5c IncHUSM-mu", ["e5c_inchusm_mu.py", "--mu-upper", "0.02", "--mu-lower", "0.01"]),
        # E6 re-runs static mining after every snapshot in every mode to prove exactness,
        # which is the slow part (~25 min); --skip e6 drops it if only the state curve
        # is wanted, and the script's own --no-compare does the same from the command line.
        ("E6 sliding window (Elliptic, proxy utility)",
         ["e6_sliding_window.py", "--stream", "elliptic", "--max-steps", "49",
          "--windows", "5", "10"]),
        # The same window on the MONETARY actor stream: the paper claims the window and the
        # monetary utility together, so they have to be shown together. add-only is skipped
        # here because its static re-run grows with the accumulated union (the window's does
        # not), and e5b already reports its state at the same threshold.
        ("E6 sliding window (actor, monetary utility)",
         ["e6_sliding_window.py", "--stream", "actor", "--max-steps", "49",
          "--windows", "5", "--no-add-only"]),
        # E7 mines a frozen vocabulary on the train subgraph and matches it snapshot by
        # snapshot (~25 min). It reports a negative result, which the paper states.
        ("E7 actor-level HUS detection", ["e7_actor_hus_detection.py", "--min-util", "100"]),
        # E8 needs e5a's graph and HUS set. Its Citeseer arm is seconds; the Elliptic++ arms
        # mine the same graph twice more and dominate the cost (~1h), so --skip-mine drops
        # them and keeps the re-scoring, which is exact and carries the utility contrast.
        ("E8 utility contrast (is the utility dimension active?)",
         ["e8_utility_contrast.py"]),
        ("E3+ recovery HI-Small", ["e3plus_recovery8.py", "--dataset", "HI-Small"]),
        ("E3+ recovery LI-Small", ["e3plus_recovery8.py", "--dataset", "LI-Small", *big]),
    ]
    if args.include_medium:
        plan += [
            ("E3+ recovery HI-Medium", ["e3plus_recovery8.py", "--dataset", "HI-Medium", *big]),
            ("E3+ recovery LI-Medium", ["e3plus_recovery8.py", "--dataset", "LI-Medium", *big]),
        ]
    plan += [("E4+ detection + drift", ["e4plus_gnn_drift.py", "--seeds", "10", "--trees", "100"])]
    skip = {s.lower() for s in args.skip}
    return [(name, cmd) for name, cmd in plan
            if cmd[0].split("_")[0].replace("plus", "").replace("+", "") not in skip
            and not any(cmd[0].startswith(p) for p in skip)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-medium", action="store_true", help="also run AMLworld Medium for E3+")
    ap.add_argument("--skip", nargs="*", default=[], help="scripts to skip, e.g. e4plus_gnn_drift.py")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    log = RESULTS / "run_all_paper3_log.txt"
    jar_ok = ensure_jar()
    if not jar_ok:
        print("[warn] engine jar unavailable: E5a/E5b/E5c will fail; E3+/E4+ still run.", flush=True)
    check_datasets()

    results = []
    t_all = time.time()
    for name, cmd in steps(args):
        print(f"\n{'=' * 70}\n{name}: {' '.join(cmd)}\n{'=' * 70}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(SCRIPTS / cmd[0]), *cmd[1:]])
        dt = time.time() - t0
        status = "PASS" if r.returncode == 0 else f"FAIL (exit {r.returncode})"
        results.append((name, status, dt))
        print(f"-> {name}: {status} ({dt:.0f}s)", flush=True)

    total = time.time() - t_all
    lines = [f"{status:20s} {dt:7.0f}s  {name}" for name, status, dt in results]
    summary = "\n".join(lines) + f"\n{'-' * 40}\nTOTAL {total:.0f}s"
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}\n{summary}", flush=True)
    log.write_text(summary + "\n", encoding="utf-8")
    print(f"\nlog -> {log}")

    # A step can exit 0 and still produce the wrong numbers, so grade the deterministic
    # values rather than trusting the exit codes.
    print(f"\n{'=' * 70}\nGRADING deterministic values\n{'=' * 70}", flush=True)
    subprocess.run([sys.executable, str(SCRIPTS / "check_results.py")])


if __name__ == "__main__":
    main()
