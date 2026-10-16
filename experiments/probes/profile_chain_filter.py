"""Where the chain miner spends its time, with and without slot consistency (a feasibility probe).

Runs ChainMain under Java Flight Recorder once per mode and reports, for a fixed set of methods, the
share of execution samples in which the method is on the stack (inclusive) and on top (self). Shares
are fractions of samples, not times: the probe decides which part of the filter is worth changing,
it measures no performance. Both modes must write the same pattern table.

Run: python experiments/probes/profile_chain_filter.py --dataset HI-Medium --labeling A [--heap 16g]
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JAR = ROOT / "inchusm-miner" / "target" / "inchusm-miner-0.1.0-SNAPSHOT.jar"
OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "results-probe" / "scratch"
MODES = {"filter": [], "no-filter": ["--no-consistency"]}
PREFIX = "edu.huflit.inchusm.chain."


def java_home() -> Path:
    out = subprocess.run(["java", "-XshowSettings:properties", "-version"], capture_output=True, text=True).stderr
    for line in out.splitlines():
        if line.strip().startswith("java.home = "):
            return Path(line.split(" = ", 1)[1].strip())
    sys.exit("could not read java.home")


def frame_name(frame: dict) -> str:
    m = frame["method"]
    cls = m["type"]["name"].replace("/", ".")
    return f"{cls}.{m['name']}"


def shares(samples: list[list[str]]) -> tuple[dict, dict]:
    """Per method: fraction of samples with it anywhere on the stack, and with it on top."""
    inclusive, self_ = collections.Counter(), collections.Counter()
    for stack in samples:
        if stack:
            self_[stack[0]] += 1
        for name in set(stack):
            inclusive[name] += 1
    n = len(samples)
    return ({k: v / n for k, v in inclusive.items()}, {k: v / n for k, v in self_.items()}) if n else ({}, {})


def read_samples(jfr: Path, recording: Path) -> list[list[str]]:
    text = subprocess.run([str(jfr), "print", "--json", "--stack-depth", "64", "--events", "jdk.ExecutionSample",
                           str(recording)], capture_output=True, text=True, check=True).stdout
    events = json.loads(text)["recording"]["events"]
    return [[frame_name(f) for f in e["values"]["stackTrace"]["frames"]] for e in events
            if e["values"].get("stackTrace")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labeling", choices=["S", "A"], required=True)
    ap.add_argument("--theta", default="0.2")
    ap.add_argument("--tau", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=5)
    ap.add_argument("--min-cycle", type=int, default=3)
    ap.add_argument("--heap", default="16g")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    graph = ROOT / "datasets" / "chain" / f"{args.dataset}_{args.labeling}.tg"
    if not graph.is_file():
        sys.exit(f"missing: {graph.relative_to(ROOT).as_posix()}")
    home = java_home()
    java, jfr = home / "bin" / "java", home / "bin" / "jfr"
    OUT.mkdir(parents=True, exist_ok=True)
    rows, digests = [], {}
    for mode, extra in MODES.items():
        rec = OUT / f"profile_{mode}.jfr"
        table = OUT / f"patterns_{mode}.tsv"
        cmd = [str(java), f"-Xmx{args.heap}", f"-XX:StartFlightRecording=filename={rec},settings=profile",
               "-cp", str(JAR), "edu.huflit.inchusm.chain.ChainMain", "--graph", str(graph), "--theta", args.theta,
               "--min-support", str(args.tau), "--max-length", str(args.max_length),
               "--min-cycle", str(args.min_cycle), "--out", str(table)] + extra
        print(f"[profile] {mode}: started", flush=True)
        log = subprocess.run(cmd, capture_output=True, text=True)
        (OUT / f"log_{mode}.txt").write_text(
            "\n".join("[timing redacted: machine-dependent]" if "elapsed" in l else l
                      for l in (log.stdout + log.stderr).splitlines()) + "\n", encoding="utf-8")
        if log.returncode != 0:
            sys.exit(f"{mode}: ChainMain exited {log.returncode}")
        digests[mode] = hashlib.sha256(table.read_bytes()).hexdigest()
        samples = read_samples(jfr, rec)
        rec.unlink()
        inc, slf = shares(samples)
        names = sorted({k for k in inc if k.startswith(PREFIX)} | set(sorted(slf, key=slf.get, reverse=True)[:args.top]),
                       key=lambda k: -inc.get(k, 0))
        for k in names:
            rows.append({"mode": mode, "samples": len(samples), "method": k,
                         "inclusive_share": round(inc.get(k, 0), 4), "self_share": round(slf.get(k, 0), 4)})
        print(f"[profile] {mode}: {len(samples)} samples", flush=True)
    if len(set(digests.values())) != 1:
        sys.exit(f"OUTPUT DIFFERS between modes: {digests}")
    with open(OUT / "profile_shares.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[profile] same pattern table in both modes: {next(iter(digests.values()))}")


if __name__ == "__main__":
    main()
