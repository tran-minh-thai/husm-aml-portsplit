"""Time the chain miner against the enumeration baseline on the same output, and the miner without
slot consistency.

For one data set, labeling and parameter set, runs ChainMain three ways:

  miner             the level-wise miner with slot consistency
  miner-no-filter   the same miner with --no-consistency (same output; counts what the filter saves)
  miner-from-L      the same miner with slot consistency only from candidate length L (L = 3, 4, 5; only
                    when named with --engines)
  enumerate         the baseline that enumerates every valid occurrence and aggregates image sets

Every run that finishes must write exactly the same pattern table as the miner; a difference stops
the tool with exit status 1 and no timing is kept. Runs share a time limit and a fixed heap; a run that
reaches the limit is killed and reported as "timeout".

Machine-independent output (the record directory): the pattern table's sha256 for each engine, the
equality check and the miner's work counters. Machine-dependent output (RUN_TIMING_DIR, only when the
recorder pairs a measurement record): the mining time measured inside the JVM, the wall time of the
process and its peak resident memory. Without a timing directory no timing is written anywhere.

Before running, the JVM's runtime version and vendor are compared with docs/reproduce/jdk.lock; a
missing lock file is written from this JVM (to be committed), a different JVM is refused. A probe (the
recorder sets RUN_TIMING_KIND to a probe kind) skips the lock and records its JVM in jvm.txt.

Run: python experiments/scripts/chain_performance.py --dataset HI-Medium --labeling A \\
         [--theta 0.2] [--tau 10] [--max-length 5] [--min-cycle 3] [--limit 7200] [--heap 48g]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JAR = ROOT / "inchusm-miner" / "target" / "inchusm-miner-0.1.0-SNAPSHOT.jar"
CHAIN = ROOT / "datasets" / "chain"
LOCK = ROOT / "docs" / "reproduce" / "jdk.lock"
OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"
TIMING = Path(os.environ["RUN_TIMING_DIR"]) if os.environ.get("RUN_TIMING_DIR") else None
PROBE = os.environ.get("RUN_TIMING_KIND", "").startswith("probe")
ENGINES = {"miner": [], "miner-no-filter": ["--no-consistency"], "enumerate": ["--engine", "enumerate"],
           "miner-from-3": ["--consistency-from", "3"], "miner-from-4": ["--consistency-from", "4"],
           "miner-from-5": ["--consistency-from", "5"]}
DEFAULT_ENGINES = ["miner", "miner-no-filter", "enumerate"]
ELAPSED = re.compile(r"mining elapsed ms: (\d+)")
COUNTERS = re.compile(r"candidates=(\d+) pruned_by_domain=(\d+) pruned_by_consistency=(\d+) searches=(\d+) "
                      r"found=(\d+) failed=(\d+)")


def jvm_identity(java: str = "java") -> dict:
    out = subprocess.run([java, "-XshowSettings:properties", "-version"], capture_output=True, text=True).stderr
    props = {}
    for line in out.splitlines():
        m = re.match(r"\s*(java\.runtime\.version|java\.vendor|java\.vm\.name|java\.home) = (.*)", line)
        if m:
            props[m.group(1)] = m.group(2).strip()
    return props


def jvm_executable(identity: dict) -> str:
    """The JVM binary itself, from java.home: on Windows the java on PATH can be a launcher that starts the real JVM
    as a child, which would make the peak memory and the time limit apply to the launcher instead."""
    home = identity.get("java.home")
    if not home:
        sys.exit(f"refusing to run: the JVM did not report java.home: {identity}")
    exe = Path(home) / "bin" / ("java.exe" if os.name == "nt" else "java")
    if not exe.is_file():
        sys.exit(f"refusing to run: {exe} does not exist")
    return str(exe)


def check_lock(identity: dict, lock: Path | None = None) -> str | None:
    """None when this JVM matches the lock (writing the lock if there is none), else why not."""
    lock = LOCK if lock is None else lock
    keys = ("java.runtime.version", "java.vendor")
    if any(k not in identity for k in keys):
        return f"could not read {keys} from the JVM: {identity}"
    if not lock.is_file():
        lock.write_text("".join(f"{k} = {identity[k]}\n" for k in keys), encoding="utf-8")
        print(f"[performance] wrote {lock.name} from this JVM; commit it with the results")
        return None
    pinned = dict(line.split(" = ", 1) for line in lock.read_text(encoding="utf-8").splitlines() if " = " in line)
    diff = {k: (pinned.get(k), identity[k]) for k in keys if pinned.get(k) != identity[k]}
    return None if not diff else f"JVM differs from {lock.name} (locked, found): {diff}"


def peak_memory_bytes(proc) -> int | None:
    """Peak resident memory of a finished child process; None when the platform gives none."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        ok = ctypes.WinDLL("psapi").GetProcessMemoryInfo(wintypes.HANDLE(int(proc._handle)), ctypes.byref(c), c.cb)
        return int(c.PeakWorkingSetSize) if ok else None
    return getattr(proc, "peak_rss", None)


def run(cmd: list[str], log: Path, limit: float) -> dict:
    """Run a command with its output in a file; status ok, failed or timeout; wall seconds; peak memory."""
    t0 = time.monotonic()
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        status = None
        if os.name == "nt":
            try:
                code = proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                status = "timeout"
        else:
            while True:
                pid, code, usage = os.wait4(proc.pid, os.WNOHANG)
                if pid:
                    # ru_maxrss: bytes on macOS, kilobytes on Linux
                    proc.peak_rss = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
                    proc.returncode = os.waitstatus_to_exitcode(code)
                    code = proc.returncode
                    break
                if time.monotonic() - t0 > limit:
                    proc.kill()
                    _, _, usage = os.wait4(proc.pid, 0)
                    status = "timeout"
                    break
                time.sleep(0.2)
    wall = time.monotonic() - t0
    if status is None:
        status = "ok" if code == 0 else f"failed (exit {code})"
    return {"status": status, "wall_s": wall, "peak_bytes": peak_memory_bytes(proc) if status == "ok" else None}


def table_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keep_log(raw: Path, engine: str) -> str:
    """Move a run's log out of the way: timing lines redacted in the record, the original in the timing record."""
    text = raw.read_text(encoding="utf-8", errors="replace")
    kept = "\n".join("[timing redacted: machine-dependent]" if "elapsed" in line else line for line in text.splitlines())
    (OUT / f"log_{engine}.txt").write_text(kept + "\n", encoding="utf-8")
    if TIMING is not None:
        (TIMING / f"log_{engine}.txt").write_text(text, encoding="utf-8")
    raw.unlink()
    return text


KEYS = ["dataset", "labeling", "theta", "tau", "max_length", "min_cycle", "heap", "engine", "status", "table_sha256",
        "candidates", "pruned_by_domain", "pruned_by_consistency", "searches", "found", "failed"]


def write_rows(rows: list[dict]) -> None:
    with open(OUT / "performance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labeling", choices=["S", "A"], required=True)
    ap.add_argument("--theta", default="0.2")
    ap.add_argument("--tau", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=5)
    ap.add_argument("--min-cycle", type=int, default=3)
    ap.add_argument("--limit", type=float, default=7200.0, help="seconds per run")
    ap.add_argument("--heap", default="48g")
    ap.add_argument("--engines", nargs="+", default=DEFAULT_ENGINES, choices=list(ENGINES))
    args = ap.parse_args()

    identity = jvm_identity()
    java = jvm_executable(identity)
    refusal = check_lock(identity) if not PROBE else None
    if refusal:
        sys.exit(f"refusing to run: {refusal}")
    if PROBE:
        print(f"[performance] probe timing: JVM not checked against {LOCK.name} ({identity.get('java.runtime.version')})")
    graph = CHAIN / f"{args.dataset}_{args.labeling}.tg"
    if not graph.is_file():
        shown = graph.relative_to(ROOT).as_posix() if graph.is_relative_to(ROOT) else graph.name
        sys.exit(f"refusing to run: {shown} does not exist (export the chain input first)")

    rows, timing = [], []
    reference = None
    for engine in args.engines:
        table = OUT / f"patterns_{engine}.tsv"
        cmd = [java, f"-Xmx{args.heap}", "-cp", str(JAR), "edu.huflit.inchusm.chain.ChainMain", "--graph", str(graph),
               "--theta", args.theta, "--min-support", str(args.tau), "--max-length", str(args.max_length),
               "--min-cycle", str(args.min_cycle), "--out", str(table)] + ENGINES[engine]
        print(f"[performance] {engine}: started", flush=True)
        # the child writes its log on local disk: a record directory inside a synchronised folder has been read
        # back empty right after the child closed the file, which turned a finished run into a row without numbers
        raw = Path(tempfile.mkdtemp(prefix="chain-performance-")) / f"raw_log_{engine}.txt"
        r = run(cmd, raw, args.limit)
        log = keep_log(raw, engine)
        raw.parent.rmdir()
        complete = ELAPSED.search(log) and (engine == "enumerate" or COUNTERS.search(log))
        if r["status"] == "ok" and not complete:
            r["status"] = "failed (exit 0 but the log has no counters or elapsed line)"
        row = {"dataset": args.dataset, "labeling": args.labeling, "theta": args.theta, "tau": args.tau,
               "max_length": args.max_length, "min_cycle": args.min_cycle, "heap": args.heap, "engine": engine,
               "status": r["status"], "table_sha256": table_digest(table) if r["status"] == "ok" else ""}
        c = COUNTERS.search(log)
        if c:
            row.update(dict(zip(["candidates", "pruned_by_domain", "pruned_by_consistency", "searches", "found",
                                 "failed"], map(int, c.groups()))))
        if r["status"] == "ok":
            if reference is None:
                reference = (engine, row["table_sha256"])
            elif row["table_sha256"] != reference[1]:
                write_rows(rows + [row])
                sys.exit(f"OUTPUT DIFFERS: {engine} wrote a different pattern table from {reference[0]}; no timing kept")
        rows.append(row)
        e = ELAPSED.search(log)
        timing.append({"dataset": args.dataset, "labeling": args.labeling, "engine": engine, "status": r["status"],
                       "mining_ms": int(e.group(1)) if e and r["status"] == "ok" else "",
                       "wall_s": round(r["wall_s"], 3), "peak_bytes": r["peak_bytes"] or "", "limit_s": args.limit,
                       "heap": args.heap})
        print(f"[performance] {engine}: {r['status']}", flush=True)

    write_rows(rows)
    finished = [r for r in rows if r["status"] == "ok"]
    print(f"[performance] {len(finished)} of {len(rows)} engines finished; all finished tables identical: "
          f"{len({r['table_sha256'] for r in finished}) <= 1}")
    if TIMING is not None:
        with open(TIMING / "timing.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(timing[0]))
            w.writeheader()
            w.writerows(timing)
        (TIMING / "jvm.txt").write_text("".join(f"{k} = {v}\n" for k, v in sorted(identity.items())), encoding="utf-8")
        print(f"[performance] timing written to the paired measurement record")
    else:
        print("[performance] no measurement record paired: no timing written")


if __name__ == "__main__":
    main()
