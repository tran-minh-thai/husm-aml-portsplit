"""Find the accounts GraMi counts for a chain without an occurrence behind them.

GraMi decides that a pattern is frequent when every variable keeps at least tau values in its domain after the
search (DFSSearch.getFrequencyOfPattern). The variant grami-path-nocache-sharedtimer-dump writes those domains
for every frequent pattern with --max-length edges (grami-dump.patch). This tool runs it and the chain miner on one
prefix and checks every value GraMi kept against the definition: is the account the image of that position in
some occurrence of the chain (grami_added_check.has_occurrence)?

Machine-independent output (the record directory):
  dump_check.csv  one row per (chain, position) for every chain GraMi reports and the miner does not, and for a
                  fixed random sample of the chains both report: the number and the full list of the values
                  GraMi kept, of those without an occurrence at that position, and of those with one (the
                  images), so that every count can be traced to accounts
The dump is written by GraMi with limitedtime=false, so no search is stopped and the output does not depend on
the speed of the machine.

Exit status 1 when a value's account label differs from the label of its variable (the dump is then not read
as intended), 2 when a check passes --budget search steps.

Run: python experiments/scripts/grami_dump_check.py --dataset sweden_100K_easy --labeling A --prefix 1000000 \\
         --grami-build <dir> [--tau 10] [--max-length 4] [--heap 24g] [--sample 50]
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chain_performance as cp  # noqa: E402
import grami_added_check as ac  # noqa: E402
import grami_crosscheck as gc  # noqa: E402
import grami_speed as gs  # noqa: E402

OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"
ENGINE = "grami-path-nocache-sharedtimer-dump"
SAMPLE_SEED = 20261022


def parse_dump_line(line: str):
    """(chain key, [values of position 0], [values of position 1], ...) or None when not a directed path.

    Line: P <node labels> | a>b:label ... | id:label:values | ...
    """
    parts = [p.strip() for p in line[1:].split("|")]
    labels = [int(x) for x in parts[0].split(",")]
    outs, ins = {}, {}
    for e in parts[1].split():
        ab, lab = e.split(":")
        if ">" in ab:
            a, b = (int(x) for x in ab.split(">"))
        else:
            b, a = (int(x) for x in ab.split("<"))
        if a in outs or b in ins:
            return None
        outs[a], ins[b] = (b, int(float(lab))), a
    starts = [v for v in range(len(labels)) if v not in ins]
    if len(starts) != 1:
        return None
    order, v = [starts[0]], starts[0]
    key = f"O {labels[v]}"
    while v in outs:
        w, lab = outs[v]
        key += f" >{lab} {labels[w]}"
        order.append(w)
        v = w
    if len(order) != len(labels):
        return None
    values, var_labels = {}, {}
    for p in parts[2:]:
        vid, vlab, vals = p.split(":", 2)
        values[int(vid)] = [int(x) for x in vals.split(",") if x]
        var_labels[int(vid)] = int(vlab)
    if sorted(values) != list(range(len(labels))) or any(var_labels[i] != labels[i] for i in values):
        raise ValueError(f"variables do not match the pattern nodes: {line[:200]}")
    return key, [values[node] for node in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labeling", choices=["S", "A"], required=True)
    ap.add_argument("--prefix", type=int, required=True)
    ap.add_argument("--grami-build", required=True)
    ap.add_argument("--tau", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=4)
    ap.add_argument("--heap", default="24g")
    ap.add_argument("--limit", type=float, default=14400.0)
    ap.add_argument("--sample", type=int, default=50, help="chains both programs report, checked the same way")
    ap.add_argument("--budget", type=int, default=10**9, help="search steps over the whole check")
    args = ap.parse_args()
    ac.self_test()
    identity = cp.jvm_identity()
    java = cp.jvm_executable(identity)
    binary = Path(args.grami_build).resolve() / ENGINE / "bin"
    if not (binary / "Dijkstra" / "main.class").is_file():
        sys.exit(f"refusing to run: {ENGINE} is not built under {args.grami_build}")
    vlab, edges = gc.prefix_graph(cp.CHAIN / f"{args.dataset}_{args.labeling}.tg", args.prefix)
    print(f"[dump] {args.dataset} {args.labeling} prefix {args.prefix}: {len(vlab)} accounts, {len(edges)} pairs", flush=True)
    work = Path(tempfile.mkdtemp(prefix="grami-dump-"))
    (work / "g.tg").write_text("".join(f"v {i} {l}\n" for i, l in enumerate(vlab))
                               + "".join(f"e {u} {v} {l} 1\n" for u, v, l in edges), encoding="utf-8")
    (work / "g.lg").write_text("t # 0\n" + "".join(f"v {i} {l}\n" for i, l in enumerate(vlab))
                               + "".join(f"e {u} {v} {l}\n" for u, v, l in edges), encoding="utf-8")
    r, mine, log = gs.ours(java, args.heap, work, args.tau, args.max_length, args.limit)
    if mine is None:
        sys.exit(f"the miner did not finish ({r['status']}):\n{log[-2000:]}")
    run_dir = work / ENGINE
    run_dir.mkdir()
    (run_dir / "g.lg").write_bytes((work / "g.lg").read_bytes())
    cmd = [java, f"-Xmx{args.heap}", f"-Dgrami.maxEdges={args.max_length}", f"-Dgrami.dump={args.max_length}", "-cp",
           str(binary), "Dijkstra.main", f"freq={args.tau}", "filename=g.lg", "datasetFolder=./", "distance=1", "type=1",
           "mlabels=false", "maxLabelAppearance=-1", "approximate=1", "approxConst=0", "limitedtime=false"]
    rg = gs.run_in(cmd, run_dir, work / "grami.log", args.limit)
    if rg["status"] != "ok":
        sys.exit(f"GraMi did not finish: {rg['status']}")
    gc.MAX_LENGTH = args.max_length
    theirs = gc.parse_output((run_dir / "Output.txt").read_text(encoding="utf-8"))
    domains, not_paths = {}, 0
    for line in (run_dir / "grami-dump.txt").read_text(encoding="utf-8").splitlines():
        parsed = parse_dump_line(line)
        if parsed is None:
            not_paths += 1
            continue
        domains[parsed[0]] = parsed[1]
    shutil.rmtree(work)
    longest = {c for c in theirs if len(c.split()) == 2 * args.max_length + 2}
    print(f"[dump] miner {len(mine)} chains, GraMi {len(theirs)} ({len(longest)} of length {args.max_length}); "
          f"dumped patterns {len(domains)} directed paths, {not_paths} other", flush=True)
    added = sorted(c for c in theirs - mine if c in domains)
    unexplained = sorted(c for c in theirs - mine if c not in domains)
    agreed = sorted(c for c in (theirs & mine) if c in domains)
    sample = sorted(random.Random(SAMPLE_SEED).sample(agreed, min(args.sample, len(agreed))))
    g = ac.Graph(vlab, edges)
    budget = ac.Budget(args.budget)
    rows, label_mismatch = [], 0
    try:
        for kind, keys in (("added by GraMi", added), ("both report, sample", sample)):
            for key in keys:
                vl, el = ac.parse_key(key)
                for i, values in enumerate(domains[key]):
                    label_mismatch += sum(1 for x in values if vlab[x] != vl[i])
                    without = [x for x in values if not ac.has_occurrence(g, vl, el, i, x, budget)]
                    images = sorted(set(values) - set(without))
                    rows.append(dict(chain=key, kind=kind, position=i, grami_values=len(values),
                                     values_without_occurrence=len(without), images=len(images),
                                     grami_value_ids=" ".join(str(x) for x in sorted(values)),
                                     without_occurrence_ids=" ".join(str(x) for x in sorted(without)),
                                     image_ids=" ".join(str(x) for x in images)))
    except TimeoutError:
        print(f"[dump] the check passed {args.budget} search steps", flush=True)
        sys.exit(2)
    with open(OUT / "dump_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["chain", "kind", "position", "grami_values", "values_without_occurrence",
                                          "images", "grami_value_ids", "without_occurrence_ids", "image_ids"])
        w.writeheader()
        w.writerows(rows)
    for kind in ("added by GraMi", "both report, sample"):
        mine_rows = [r for r in rows if r["kind"] == kind]
        chains = {r["chain"] for r in mine_rows}
        with_bad = {r["chain"] for r in mine_rows if r["values_without_occurrence"]}
        print(f"[dump] {kind}: {len(chains)} chains, {len(with_bad)} with values that have no occurrence; "
              f"{sum(r['values_without_occurrence'] for r in mine_rows)} such values over {len(mine_rows)} positions")
    print(f"[dump] chains GraMi adds with no dumped domains: {len(unexplained)}; values whose label differs from "
          f"their variable: {label_mismatch}")
    if label_mismatch:
        sys.exit(1)


if __name__ == "__main__":
    main()
