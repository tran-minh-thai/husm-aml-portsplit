"""Decide from the definition which program is right where GraMi and the chain miner report different chains.

grami_speed.py stops when a GraMi variant reports a frequent open chain that the chain miner does not. This tool
reruns one such prefix and keeps everything needed to decide:

  * the chain miner (ChainMain --theta inf --no-closed) and one GraMi variant run twice on the same graph: with
    GraMi's default setting limitedtime=true, under which a search for one value of the first pattern vertex is
    stopped after five seconds (seven in the second pass), and with limitedtime=false, a command-line option of
    GraMi itself;
  * the number of searches GraMi reports as stopped ("Time's up!" lines of its output);
  * for every chain reported by one program and not by another, and for a fixed random sample of the chains all
    programs agree on, whether the chain is frequent by the definition: every position of the chain has at least
    tau distinct accounts that are the image of that position in some occurrence (an injective mapping of the
    chain onto accounts joined by transactions with the chain's labels). The check searches one occurrence through
    each candidate account of each position and stops a position once tau accounts are found.

Machine-independent output (the record directory):
  runs.csv              one row per program: status, chains, missing and added against the miner, stopped searches
  definition_check.csv  one row per checked chain: in which outputs it is, frequent by the definition, and for a
                        chain that is not frequent the exact number of images of the first position found to
                        have fewer than tau (an upper bound of the chain's support)
The outcome of a GraMi run with limitedtime=true can depend on the speed of the machine.

Exit status 1 when the definition contradicts the miner on a checked chain, 2 when a chain stays undecided within
--budget search steps.

Run: python experiments/scripts/grami_added_check.py --dataset sweden_100K_easy --labeling A --prefix 1000000 \\
         --grami-build <dir> [--engine grami-path-nocache-sharedtimer] [--tau 10] [--max-length 4] [--heap 24g]
     python experiments/scripts/grami_added_check.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import random
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chain_performance as cp  # noqa: E402
import grami_crosscheck as gc  # noqa: E402
import grami_speed as gs  # noqa: E402

OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"
SAMPLE_SEED = 20261022


def parse_key(key: str):
    """'O 3 >1 4 >2 3' -> vertex labels (3, 4, 3), edge labels (1, 2)."""
    t = key.split()
    assert t[0] == "O" and len(t) % 2 == 0, key
    return tuple(int(x) for x in t[1::2]), tuple(int(x[1:]) for x in t[2::2])


class Graph:
    def __init__(self, vlab: list[int], edges: list[tuple[int, int, int]]):
        self.vlab = vlab
        self.out = defaultdict(list)    # (account, transaction label) -> receivers
        self.inn = defaultdict(list)    # (account, transaction label) -> senders
        for u, v, l in edges:
            self.out[(u, l)].append(v)
            self.inn[(v, l)].append(u)


class Budget:
    def __init__(self, limit: int):
        self.limit, self.steps = limit, 0


def has_occurrence(g: Graph, vl: tuple, el: tuple, i: int, x: int, budget: Budget) -> bool:
    """Whether account x is the image of position i in some occurrence; raises TimeoutError past the budget."""
    k = len(el)
    if g.vlab[x] != vl[i]:
        return False
    order = list(range(i + 1, k + 1)) + list(range(i - 1, -1, -1))

    def extend(assign: dict, j: int) -> bool:
        if j == len(order):
            return True
        p = order[j]
        # right of the start the account before is assigned, left of it the account after
        cands = g.out.get((assign[p - 1], el[p - 1]), ()) if p > i else g.inn.get((assign[p + 1], el[p]), ())
        used = set(assign.values())
        for w in cands:
            budget.steps += 1
            if budget.steps > budget.limit:
                raise TimeoutError
            if g.vlab[w] != vl[p] or w in used:
                continue
            assign[p] = w
            if extend(assign, j + 1):
                del assign[p]
                return True
            del assign[p]
        return False

    return extend({i: x}, 0)


def images_at_least(g: Graph, key: str, tau: int, budget: int):
    """(frequent, images, steps): images is the exact number of images of the first position with fewer than tau,
    or tau when the chain is frequent; frequent is None when the search passes the budget."""
    vl, el = parse_key(key)
    k = len(el)
    b = Budget(budget)
    try:
        for i in range(k + 1):
            found = 0
            for x in range(len(g.vlab)):
                if g.vlab[x] != vl[i]:
                    continue
                if (i < k and not g.out.get((x, el[i]))) or (i > 0 and not g.inn.get((x, el[i - 1]))):
                    continue
                if has_occurrence(g, vl, el, i, x, b):
                    found += 1
                    if found >= tau:
                        break
            if found < tau:
                return False, found, b.steps
    except TimeoutError:
        return None, "", b.steps
    return True, tau, b.steps


def brute_images(vlab, edges, max_edges: int) -> dict:
    """Images of every position of every chain with an occurrence, by enumerating every simple path (small graphs)."""
    out = defaultdict(list)
    for u, v, l in edges:
        out[u].append((v, l))
    images = defaultdict(lambda: defaultdict(set))

    def grow(path, key):
        if len(path) > 1:
            for i, v in enumerate(path):
                images[key][i].add(v)
        if len(path) == max_edges + 1:
            return
        for w, l in out[path[-1]]:
            if w not in path:
                grow(path + [w], key + f" >{l} {vlab[w]}")
    for v in range(len(vlab)):
        grow([v], f"O {vlab[v]}")
    return images


def brute_supports(vlab, edges, max_edges: int) -> dict:
    """Minimum image support of every chain with an occurrence."""
    return {key: min(len(s) for s in img.values()) for key, img in brute_images(vlab, edges, max_edges).items()}


def self_test() -> None:
    rnd = random.Random(7)
    compared = disagree = frequent = accounts = account_disagree = 0
    for _ in range(150):
        n = rnd.randint(2, 7)
        vlab = [rnd.randint(0, 1) for _ in range(n)]
        pairs = [(u, v) for u in range(n) for v in range(n) if u != v]
        edges = [(u, v, rnd.randint(0, 1)) for u, v in rnd.sample(pairs, rnd.randint(1, len(pairs)))]
        g = Graph(vlab, edges)
        img = brute_images(vlab, edges, 4)
        sup = {key: min(len(s) for s in v.values()) for key, v in img.items()}
        for m in range(1, 5):
            for vl in itertools.product((0, 1), repeat=m + 1):
                for el in itertools.product((0, 1), repeat=m):
                    key = "O " + str(vl[0]) + "".join(f" >{el[i]} {vl[i + 1]}" for i in range(m))
                    s = sup.get(key, 0)
                    pv, pe = parse_key(key)
                    for i in range(m + 1):
                        for x in range(n):
                            accounts += 1
                            if has_occurrence(g, pv, pe, i, x, Budget(10**7)) != (x in img.get(key, {}).get(i, ())):
                                account_disagree += 1
                    for tau in (1, 2, 3):
                        freq, images, _ = images_at_least(g, key, tau, 10**7)
                        compared += 1
                        frequent += s >= tau
                        if freq != (s >= tau) or (not freq and not s <= images < tau):
                            disagree += 1
    print(f"self-test: {compared} (graph, chain, tau) cases compared with full enumeration ({frequent} frequent), "
          f"{disagree} disagree; {accounts} (chain, position, account) cases, {account_disagree} disagree")
    if compared == 0 or frequent == 0 or frequent == compared or disagree or accounts == 0 or account_disagree:
        sys.exit(1)


def run_grami(java, heap, binary, work, tau, length, limit, limited: bool):
    name = f"{binary.parent.name}-limitedtime-{str(limited).lower()}"
    run_dir = work / name
    run_dir.mkdir()
    (run_dir / "g.lg").write_bytes((work / "g.lg").read_bytes())
    cmd = [java, f"-Xmx{heap}", f"-Dgrami.maxEdges={length}", "-cp", str(binary), "Dijkstra.main", f"freq={tau}",
           "filename=g.lg", "datasetFolder=./", "distance=1", "type=1", "mlabels=false", "maxLabelAppearance=-1",
           "approximate=1", "approxConst=0", f"limitedtime={str(limited).lower()}"]
    log = work / f"{name}.log"
    r = gs.run_in(cmd, run_dir, log, limit)
    stopped = sum(1 for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith("Time's up!"))
    chains = None
    if r["status"] == "ok":
        text = (run_dir / "Output.txt").read_text(encoding="utf-8")
        gc.MAX_LENGTH = length
        chains, other = gc.parse_output(text, count_other=True)
        if other:
            r["status"], chains = f"failed ({other} patterns are not directed paths)", None
    return name, r, chains, stopped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dataset")
    ap.add_argument("--labeling", choices=["S", "A"])
    ap.add_argument("--prefix", type=int)
    ap.add_argument("--grami-build")
    ap.add_argument("--engine", default="grami-path-nocache-sharedtimer", choices=gs.GRAMI)
    ap.add_argument("--tau", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=4)
    ap.add_argument("--heap", default="24g")
    ap.add_argument("--limit", type=float, default=14400.0, help="seconds per GraMi run")
    ap.add_argument("--sample", type=int, default=200, help="agreed chains checked against the definition")
    ap.add_argument("--budget", type=int, default=10**8, help="search steps per chain")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if None in (args.dataset, args.labeling, args.prefix, args.grami_build):
        sys.exit("--dataset, --labeling, --prefix and --grami-build are required")
    self_test()
    identity = cp.jvm_identity()
    java = cp.jvm_executable(identity)
    binary = Path(args.grami_build).resolve() / args.engine / "bin"
    if not (binary / "Dijkstra" / "main.class").is_file():
        sys.exit(f"refusing to run: {args.engine} is not built under {args.grami_build}")
    graph = cp.CHAIN / f"{args.dataset}_{args.labeling}.tg"
    vlab, edges = gc.prefix_graph(graph, args.prefix)
    print(f"[added] {args.dataset} {args.labeling} prefix {args.prefix}: {len(vlab)} accounts, {len(edges)} pairs", flush=True)
    work = Path(tempfile.mkdtemp(prefix="grami-added-"))
    (work / "g.tg").write_text("".join(f"v {i} {l}\n" for i, l in enumerate(vlab))
                               + "".join(f"e {u} {v} {l} 1\n" for u, v, l in edges), encoding="utf-8")
    (work / "g.lg").write_text("t # 0\n" + "".join(f"v {i} {l}\n" for i, l in enumerate(vlab))
                               + "".join(f"e {u} {v} {l}\n" for u, v, l in edges), encoding="utf-8")
    r, mine, log = gs.ours(java, args.heap, work, args.tau, args.max_length, args.limit)
    if mine is None:
        sys.exit(f"the miner did not finish ({r['status']}):\n{log[-2000:]}")
    outputs = {"miner": mine}
    rows = [dict(program="miner", status="ok", chains=len(mine), missing=0, added=0, stopped_searches="")]
    for limited in (True, False):
        name, rg, theirs, stopped = run_grami(java, args.heap, binary, work, args.tau, args.max_length, args.limit, limited)
        rows.append(dict(program=name, status=rg["status"], chains=len(theirs) if theirs is not None else "",
                         missing=len(mine - theirs) if theirs is not None else "",
                         added=len(theirs - mine) if theirs is not None else "", stopped_searches=stopped))
        print(f"[added] {name}: {rg['status']}; chains {rows[-1]['chains']}, missing {rows[-1]['missing']}, "
              f"added {rows[-1]['added']}, stopped searches {stopped}", flush=True)
        if theirs is not None:
            outputs[name] = theirs
    shutil.rmtree(work)
    with open(OUT / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    union = set().union(*outputs.values())
    agreed = set.intersection(*outputs.values())
    differing = sorted(union - agreed)
    sample = sorted(random.Random(SAMPLE_SEED).sample(sorted(agreed), min(args.sample, len(agreed))))
    g = Graph(vlab, edges)
    checked, contradict, undecided = [], 0, 0
    for kind, keys in (("differing", differing), ("agreed sample", sample)):
        for key in keys:
            freq, smallest, steps = images_at_least(g, key, args.tau, args.budget)
            row = dict(chain=key, kind=kind, **{f"in_{p}": key in s for p, s in outputs.items()},
                       frequent_by_definition="" if freq is None else freq, images=smallest,
                       search_steps=steps)
            checked.append(row)
            if freq is None:
                undecided += 1
            elif freq != (key in mine):
                contradict += 1
    with open(OUT / "definition_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(checked[0]) if checked else ["chain"])
        w.writeheader()
        w.writerows(checked)
    for p, s in outputs.items():
        if p == "miner":
            continue
        right = sum(1 for row in checked if row["frequent_by_definition"] != "" and
                    row["frequent_by_definition"] == (row["chain"] in s))
        decided = sum(1 for row in checked if row["frequent_by_definition"] != "")
        print(f"[added] {p}: agrees with the definition on {right} of {decided} decided chains")
    print(f"[added] checked {len(differing)} differing and {len(sample)} agreed chains: the definition contradicts "
          f"the miner on {contradict}, {undecided} undecided")
    if contradict:
        sys.exit(1)
    if undecided:
        sys.exit(2)


if __name__ == "__main__":
    main()
