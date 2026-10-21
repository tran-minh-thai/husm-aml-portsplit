"""Which mechanism of GraMi's caching makes GraMi-path miss frequent directed paths.

With Settings.CACHING on, GraMi keeps, for every evaluated pattern, a table of the data nodes found not to extend
each pattern node ("non-candidates") and reuses it in three places. Each variant below is GraMi-path as built by
third_party/grami/build_grami.py with caching on and one of these places switched off; grami-path (as published)
and grami-path-nocache (Settings.CACHING = false) are the two references:

  no-automorphism   Settings.isAutomorphismOn = false: a rejected node is not marked for the symmetric pattern nodes
  no-inheritance    a grown pattern starts with no non-candidates instead of a copy of its parent's
  no-import         a pattern does not import the non-candidates cached for its sub-patterns with one edge removed

Every variant and the chain miner (theta = inf, open chains) run on the same prefix graph as grami_speed.py; the
frequent chains of each variant are compared with the miner's, and diagnosis.csv lists the chains each variant misses.

Run: python experiments/scripts/grami_cache_diagnosis.py --dataset HI-Small --labeling A --prefix 100000 \\
         --grami-build <build_grami.py output> [--tau 10] [--max-length 4] [--limit 3600] [--heap 24g]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "third_party" / "grami"))

import build_grami as bg  # noqa: E402
import chain_performance as cp  # noqa: E402
import grami_crosscheck as gc  # noqa: E402
import grami_speed as gs  # noqa: E402

OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"
# variant -> [(file below src, text, replacement, expected occurrences)]
EDITS = {
    "no-automorphism": [("utilities/Settings.java", "public static boolean isAutomorphismOn= true;",
                         "public static boolean isAutomorphismOn= false;", 1)],
    "no-inheritance": [("search/RightMostExtension.java", "utilities.Util.clone(code.getNonCandidates())", "null", 1),
                       ("dataStructures/DFSCode.java", "singleGraph,Util.clone(nonCandidates)).set(", "singleGraph,null).set(", 1)],
    "no-import": [("dataStructures/DFSCode.java", "if(Settings.CACHING)\n\t\t\t\t{\n\t\t\t\t\t//TODO zabat 2l non-candidates !!",
                   "if(false)\n\t\t\t\t{\n\t\t\t\t\t//TODO zabat 2l non-candidates !!", 2)],
}


def build_variant(base: Path, name: str, out: Path) -> Path:
    dest = out / name
    shutil.copytree(base / "src", dest / "src")
    for rel, old, new, count in EDITS[name]:
        path = dest / "src" / rel
        text = path.read_bytes().decode("ISO-8859-1").replace("\r\n", "\n")
        if text.count(old) != count:
            sys.exit(f"{name}: {rel} contains the text to change {text.count(old)} times, expected {count}")
        path.write_bytes(text.replace(old, new).encode("ISO-8859-1"))
    sources = sorted(str(p) for p in (dest / "src").rglob("*.java")) + sorted(str(p) for p in (bg.HERE / "stubs").rglob("*.java"))
    args = dest / "sources.txt"
    args.write_text("\n".join('"' + s.replace("\\", "/") + '"' for s in sources) + "\n", encoding="utf-8")
    (dest / "bin").mkdir()
    subprocess.run([bg.javac(), "-nowarn", "-encoding", "ISO-8859-1", "-d", str(dest / "bin"), f"@{args}"], check=True)
    return dest / "bin"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labeling", choices=["S", "A"], required=True)
    ap.add_argument("--prefix", type=int, required=True, help="number of transactions, in time order")
    ap.add_argument("--grami-build", required=True)
    ap.add_argument("--tau", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=4)
    ap.add_argument("--limit", type=float, default=3600.0)
    ap.add_argument("--heap", default="24g")
    args = ap.parse_args()
    build = Path(args.grami_build).resolve()
    java = cp.jvm_executable(cp.jvm_identity())
    with tempfile.TemporaryDirectory(prefix="grami-diagnosis-") as tmp:
        tmp = Path(tmp)
        binaries = {"grami-path": build / "grami-path" / "bin", "grami-path-nocache": build / "grami-path-nocache" / "bin"}
        for name in EDITS:
            binaries[name] = build_variant(build / "grami-path", name, tmp / "variants")
        vlab, edges = gc.prefix_graph(cp.CHAIN / f"{args.dataset}_{args.labeling}.tg", args.prefix)
        work = tmp / "graph"
        work.mkdir()
        (work / "g.tg").write_text("".join(f"v {i} {l}\n" for i, l in enumerate(vlab))
                                   + "".join(f"e {u} {v} {l} 1\n" for u, v, l in edges), encoding="utf-8")
        (work / "g.lg").write_text("t # 0\n" + "".join(f"v {i} {l}\n" for i, l in enumerate(vlab))
                                   + "".join(f"e {u} {v} {l}\n" for u, v, l in edges), encoding="utf-8")
        print(f"{args.dataset} {args.labeling}, prefix {args.prefix}: {len(vlab)} accounts, {len(edges)} pairs", flush=True)
        r, mine, log = gs.ours(java, args.heap, work, args.tau, args.max_length, args.limit)
        if mine is None:
            sys.exit(f"the miner did not finish: {r['status']}")
        print(f"miner: {len(mine)} frequent chains", flush=True)
        rows = []
        for name, binary in binaries.items():
            res, chains = gs.grami(java, args.heap, binary, work, args.tau, args.max_length, args.limit)
            missing = sorted(mine - chains) if chains is not None else []
            added = sorted(chains - mine) if chains is not None else []
            rows.append({"dataset": args.dataset, "labeling": args.labeling, "prefix": args.prefix, "variant": name,
                         "status": res["status"], "chains": len(chains) if chains is not None else "",
                         "miner_chains": len(mine), "missing": len(missing) if chains is not None else "",
                         "added": len(added) if chains is not None else "", "missing_chains": " | ".join(missing),
                         "added_chains": " | ".join(added)})
            print(f"{name}: {res['status']}; {rows[-1]['chains']} chains, missing {rows[-1]['missing']}, added {rows[-1]['added']}"
                  + (f": {' | '.join(missing)}" if missing else ""), flush=True)
    with open(OUT / "diagnosis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
