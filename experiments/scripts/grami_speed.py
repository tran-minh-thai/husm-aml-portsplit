"""Time the chain miner against GraMi on the same output: open chains with no amount constraint.

GraMi (Elseidy et al., PVLDB 2014) mines frequent subgraphs of one graph under the same minimum image based
support. Restricted to directed paths it computes exactly the frequent open chains at theta = inf, which the
chain miner computes with --theta inf --no-closed. GraMi takes a simple graph, so both programs read the same
graph: a prefix of the transactions in time order, self-transactions dropped and parallel transactions merged
(grami_crosscheck.prefix_graph). The miner and the GraMi variants of --engines run on every prefix:

  miner                           ChainMain --theta inf --no-closed
  grami-path                      GraMi restricted to directed paths of at most --max-length transactions,
                                  as published otherwise
  grami-path-nocache              the same with GraMi's constant Settings.CACHING set to false
  grami-path-sharedtimer          grami-path and grami-path-nocache with one shared timer thread instead of a
  grami-path-nocache-sharedtimer  new java.util.Timer thread per search

Built by third_party/grami/build_grami.py. The frequent chains of every program that finishes are compared
with the miner's: a nocache variant must report the same set (a difference stops the tool with exit status 1);
for the others the chains they miss and the chains they add are counted (an added chain also stops the tool),
and a shared-timer variant must report the same set as its variant with a thread per search when both ran.
A GraMi variant that reaches the time limit is not run on the larger prefixes of the same data set.

Machine-independent output (the record directory): grami_speed.csv, one row per prefix and program, with the
graph size, status, number of frequent chains, and missing and added chains. Machine-dependent output
(RUN_TIMING_DIR, measurement records only): the wall time of each process, the time each program reports
(the miner: mining only; GraMi: loading and mining), the peak resident memory, and the JVM.

Run: python experiments/scripts/grami_speed.py --dataset HI-Small --labeling A --grami-build <dir> \\
         [--prefixes 10000,30000,...] [--tau 10] [--max-length 4] [--limit 1800] [--heap 48g] \\
         [--engines grami-path,grami-path-nocache]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chain_performance as cp  # noqa: E402
import grami_crosscheck as gc  # noqa: E402

OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"
TIMING = Path(os.environ["RUN_TIMING_DIR"]) if os.environ.get("RUN_TIMING_DIR") else None
PROBE = os.environ.get("RUN_TIMING_KIND", "").startswith("probe")
PREFIXES = "10000,30000,100000,300000,1000000,3000000,10000000,30000000"
GRAMI = ("grami-path", "grami-path-nocache", "grami-path-sharedtimer", "grami-path-nocache-sharedtimer")
ENGINES = "grami-path,grami-path-nocache"
ELAPSED = re.compile(r"mining elapsed ms: (\d+)")
KEYS = ["dataset", "labeling", "prefix", "transactions_read", "accounts", "pairs", "tau", "max_length", "engine",
        "status", "chains", "missing", "added"]


def run_in(cmd: list[str], cwd: Path, log: Path, limit: float) -> dict:
    """cp.run with a working directory (GraMi reads and writes in its working directory)."""
    here = os.getcwd()
    os.chdir(cwd)
    try:
        return cp.run(cmd, log, limit)
    finally:
        os.chdir(here)


def transactions(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.startswith("e "))


def ours(java: str, heap: str, work: Path, tau: int, length: int, limit: float):
    table = work / "ours.tsv"
    cmd = [java, f"-Xmx{heap}", "-cp", str(cp.JAR), "edu.huflit.inchusm.chain.ChainMain", "--graph", str(work / "g.tg"),
           "--theta", "inf", "--min-support", str(tau), "--max-length", str(length), "--no-closed", "--out", str(table)]
    log = work / "ours.log"
    r = cp.run(cmd, log, limit)
    text = log.read_text(encoding="utf-8", errors="replace")
    chains = None
    if r["status"] == "ok":
        chains = set()
        for row in table.read_text(encoding="utf-8").splitlines()[1:]:
            kind, _, vls, els, s = row.split("\t")[:5]
            v, e = vls.split(","), els.split(",")
            if int(s) >= tau:
                chains.add("O " + v[0] + "".join(f" >{e[i]} {v[i + 1]}" for i in range(len(e))))
    e = ELAPSED.search(text)
    r["reported_s"] = int(e.group(1)) / 1000 if e and r["status"] == "ok" else ""
    return r, chains, text


def grami(java: str, heap: str, binary: Path, work: Path, tau: int, length: int, limit: float):
    run_dir = work / binary.parent.name
    run_dir.mkdir()
    (run_dir / "g.lg").write_bytes((work / "g.lg").read_bytes())
    cmd = [java, f"-Xmx{heap}", f"-Dgrami.maxEdges={length}", "-cp", str(binary), "Dijkstra.main", f"freq={tau}",
           "filename=g.lg", "datasetFolder=./", "distance=1", "type=1", "mlabels=false", "maxLabelAppearance=-1",
           "approximate=1", "approxConst=0"]
    log = work / f"{binary.parent.name}.log"
    r = run_in(cmd, run_dir, log, limit)
    chains, other = None, None
    output = run_dir / "Output.txt"
    if r["status"] == "ok" and not output.is_file():
        r["status"] = "failed (exit 0 but no Output.txt)"
    if r["status"] == "ok":
        text = output.read_text(encoding="utf-8")
        r["reported_s"] = float(text.splitlines()[0])
        gc.MAX_LENGTH = length
        chains, other = gc.parse_output(text, count_other=True)
        if other:
            r["status"] = f"failed ({other} reported patterns are not directed paths of at most {length} transactions)"
            chains = None
    else:
        r["reported_s"] = ""
    return r, chains


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labeling", choices=["S", "A"], required=True)
    ap.add_argument("--grami-build", required=True, help="output directory of third_party/grami/build_grami.py")
    ap.add_argument("--prefixes", default=PREFIXES, help="numbers of transactions, ascending")
    ap.add_argument("--tau", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=4)
    ap.add_argument("--limit", type=float, default=1800.0, help="seconds per run")
    ap.add_argument("--heap", default="48g")
    ap.add_argument("--engines", default=ENGINES, help="GraMi variants to run, in this order, comma separated")
    args = ap.parse_args()

    identity = cp.jvm_identity()
    java = cp.jvm_executable(identity)
    refusal = cp.check_lock(identity) if not PROBE else None
    if refusal:
        sys.exit(f"refusing to run: {refusal}")
    graph = cp.CHAIN / f"{args.dataset}_{args.labeling}.tg"
    if not graph.is_file():
        sys.exit(f"refusing to run: {graph.relative_to(ROOT).as_posix()} does not exist (export the chain input first)")
    engines = args.engines.split(",")
    unknown = [g for g in engines if g not in GRAMI]
    if unknown or len(set(engines)) != len(engines):
        sys.exit(f"--engines must name distinct variants of {', '.join(GRAMI)}; got {args.engines}")
    binaries = {g: Path(args.grami_build).resolve() / g / "bin" for g in engines}
    for g, b in binaries.items():
        if not (b / "Dijkstra" / "main.class").is_file():
            sys.exit(f"refusing to run: {g} is not built under {args.grami_build} (run third_party/grami/build_grami.py)")
    prefixes = [int(x) for x in args.prefixes.split(",")]
    if prefixes != sorted(set(prefixes)) or prefixes[0] <= 0:
        sys.exit("--prefixes must be positive and ascending")
    total = transactions(graph)
    timed_out = set()
    rows, timing = [], []
    for k in prefixes:
        last = k >= total
        vlab, edges = gc.prefix_graph(graph, k)
        work = Path(tempfile.mkdtemp(prefix="grami-speed-"))
        (work / "g.tg").write_text("".join(f"v {i} {l}\n" for i, l in enumerate(vlab))
                                   + "".join(f"e {u} {v} {l} 1\n" for u, v, l in edges), encoding="utf-8")
        (work / "g.lg").write_text("t # 0\n" + "".join(f"v {i} {l}\n" for i, l in enumerate(vlab))
                                   + "".join(f"e {u} {v} {l}\n" for u, v, l in edges), encoding="utf-8")
        base = {"dataset": args.dataset, "labeling": args.labeling, "prefix": k, "transactions_read": min(k, total),
                "accounts": len(vlab), "pairs": len(edges), "tau": args.tau, "max_length": args.max_length}
        print(f"[grami] prefix {k}: {len(vlab)} accounts, {len(edges)} pairs", flush=True)
        r, mine, log = ours(java, args.heap, work, args.tau, args.max_length, args.limit)
        if mine is None:
            sys.exit(f"the miner did not finish on prefix {k} ({r['status']}):\n{log[-2000:]}")
        results = [("miner", r, mine)]
        for g in engines:
            if g in timed_out:
                results.append((g, {"status": "skipped (timeout on a smaller prefix)", "wall_s": "", "peak_bytes": None,
                                    "reported_s": ""}, None))
                continue
            rg, theirs = grami(java, args.heap, binaries[g], work, args.tau, args.max_length, args.limit)
            if rg["status"] == "timeout":
                timed_out.add(g)
            results.append((g, rg, theirs))
        for engine, res, chains in results:
            row = dict(base, engine=engine, status=res["status"], chains=len(chains) if chains is not None else "",
                       missing=len(mine - chains) if chains is not None else "",
                       added=len(chains - mine) if chains is not None else "")
            rows.append(row)
            timing.append(dict(base, engine=engine, status=res["status"],
                               wall_s=round(res["wall_s"], 3) if res["wall_s"] != "" else "",
                               reported_s=res["reported_s"], peak_bytes=res["peak_bytes"] or "", limit_s=args.limit,
                               heap=args.heap))
            print(f"[grami] prefix {k} {engine}: {res['status']}; chains {row['chains']}, missing {row['missing']}, "
                  f"added {row['added']}", flush=True)
        shutil.rmtree(work)
        with open(OUT / "grami_speed.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=KEYS)
            w.writeheader()
            w.writerows(rows)
        if TIMING is not None:
            with open(TIMING / "timing.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(timing[0]))
                w.writeheader()
                w.writerows(timing)
        here = rows[-len(results):]
        bad = [x for x in here if "nocache" in x["engine"] and x["chains"] != "" and (x["missing"] or x["added"])]
        added = [x for x in here if "nocache" not in x["engine"] and x["added"]]
        found = {engine: chains for engine, _, chains in results}
        timer = [f"{g} and {g.replace('-sharedtimer', '')} report different chains" for g in engines
                 if g.endswith("-sharedtimer") and found.get(g) is not None
                 and found.get(g.replace("-sharedtimer", "")) is not None and found[g] != found[g.replace("-sharedtimer", "")]]
        if bad or added or timer:
            sys.exit(f"OUTPUT DIFFERS on prefix {k}: {bad + added + timer}")
        if last:
            break
    if TIMING is not None:
        (TIMING / "jvm.txt").write_text(cp.jvm_text(identity), encoding="utf-8")
        print("[grami] timing written to the paired measurement record")
    else:
        print("[grami] no measurement record paired: no timing written")


if __name__ == "__main__":
    main()
