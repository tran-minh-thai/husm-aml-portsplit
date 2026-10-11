"""Cross-check the chain miner against GraMi (Elseidy et al., PVLDB 2014), an independent directed miner.

GraMi mines every frequent connected subgraph of one directed graph under minimum-image support
and reports the frequent patterns without their support values. Its directed paths are exactly the
open chains, so with conservation switched off (theta = inf) the two programs must agree on which
open chains of length <= L are frequent at every threshold tau. Agreement at every tau from 1 to
one past the largest support is the same as agreeing on every support value.

GraMi takes a simple directed graph (at most one edge per ordered pair of accounts); parallel
transactions are merged before both programs see the graph, so both mine the same graph. GraMi is
used as published (built from its sources; two unused imports of types removed from the JDK are
satisfied by empty stand-ins); set GRAMI_BIN to the directory of its compiled directed-subgraph
classes.

Run: GRAMI_BIN=~/work/third_party/GraMi/GRAMI_DIRECTED_SUBGRAPHS/bin \\
     python experiments/scripts/grami_crosscheck.py random --graphs 200
     python experiments/scripts/grami_crosscheck.py prefix --graph datasets/chain/HI-Small_S.tg --transactions 5000 --tau 20
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JAR = ROOT / "inchusm-miner" / "target" / "inchusm-miner-0.1.0-SNAPSHOT.jar"
MAX_LENGTH = 4
WITH_PATH = False


def grami_bin(variable: str = "GRAMI_BIN") -> str:
    b = os.environ.get(variable)
    if not b or not Path(b, "Dijkstra", "main.class").is_file():
        sys.exit(f"set {variable} to the compiled GRAMI_DIRECTED_SUBGRAPHS/bin of "
                 + ("GraMi-path (GraMi with grami-path.patch)" if variable == "GRAMI_PATH_BIN" else "GraMi"))
    return b


def run_grami(vlab: list[int], edges: list[tuple[int, int, int]], tau: int, work: Path, timeout: float | None,
              path_only: bool = False):
    """Frequent directed-path patterns (open chains) of length <= MAX_LENGTH that GraMi reports.

    With path_only, runs GraMi-path (GraMi patched to generate directed paths of at most MAX_LENGTH
    edges only, GRAMI_PATH_BIN) and also returns how many reported patterns were not such paths."""
    work.mkdir(parents=True, exist_ok=True)
    lines = ["t # 0"] + [f"v {i} {l}" for i, l in enumerate(vlab)] + [f"e {u} {v} {l}" for u, v, l in edges]
    (work / "g.lg").write_text("\n".join(lines) + "\n")
    binary = grami_bin("GRAMI_PATH_BIN" if path_only else "GRAMI_BIN")
    cmd = ["java", f"-Dgrami.maxEdges={MAX_LENGTH}", "-cp", binary, "Dijkstra.main", f"freq={tau}", "filename=g.lg",
           "datasetFolder=./",
           "distance=1", "type=1", "mlabels=false", "maxLabelAppearance=-1", "approximate=1", "approxConst=0"]
    subprocess.run(cmd, cwd=work, check=True, capture_output=True, timeout=timeout)
    return parse_output((work / "Output.txt").read_text(), count_other=path_only)


def parse_output(text: str, count_other: bool = False):
    rows = text.splitlines()[2:]
    patterns, cur = [], None
    for r in rows:
        if r.endswith(":") and r[:-1].isdigit():
            cur = {"v": {}, "e": []}
            patterns.append(cur)
        elif r.startswith("v "):
            _, i, l = r.split()
            cur["v"][int(i)] = int(l)
        elif r.startswith("e "):
            _, a, b, l = r.split()
            cur["e"].append((int(a), int(b), int(float(l))))
    out, other = set(), 0
    for p in patterns:
        key = path_key(p["v"], p["e"])
        if key is not None and (len(key.split()) - 2) // 2 <= MAX_LENGTH:
            out.add(key)
        else:
            other += 1
    return (out, other) if count_other else out


def path_key(vl: dict[int, int], es: list[tuple[int, int, int]]) -> str | None:
    """The chain key if the pattern is a directed path (each vertex at most one in and one out edge)."""
    outs, ins = {}, {}
    for a, b, l in es:
        if a in outs or b in ins:
            return None
        outs[a], ins[b] = (b, l), a
    starts = [v for v in vl if v not in ins]
    if len(starts) != 1 or len(es) != len(vl) - 1:
        return None
    v = starts[0]
    key = f"O {vl[v]}"
    seen = {v}
    while v in outs:
        w, l = outs[v]
        if w in seen:
            return None
        key += f" >{l} {vl[w]}"
        seen.add(w)
        v = w
    return key if len(seen) == len(vl) else None


def run_ours(vlab: list[int], edges: list[tuple[int, int, int]], work: Path) -> dict[str, int]:
    """Every open chain of length <= MAX_LENGTH with its support, conservation off."""
    work.mkdir(parents=True, exist_ok=True)
    g = work / "g.tg"
    lines = [f"v {i} {l}" for i, l in enumerate(vlab)] + [f"e {u} {v} {l} 1" for u, v, l in edges]
    g.write_text("\n".join(lines) + "\n")
    out = work / "ours.tsv"
    subprocess.run(["java", "-cp", str(JAR), "edu.huflit.inchusm.chain.ChainMain", "--graph", str(g), "--theta", "inf",
                    "--min-support", "1", "--max-length", str(MAX_LENGTH), "--no-closed", "--out", str(out)],
                   check=True, capture_output=True)
    sup = {}
    for row in out.read_text().splitlines()[1:]:
        kind, _, vls, els, s = row.split("\t")[:5]
        v, e = vls.split(","), els.split(",")
        key = "O " + v[0] + "".join(f" >{e[i]} {v[i + 1]}" for i in range(len(e)))
        sup[key] = int(s)
    return sup


def compare(vlab, edges, work: Path, taus=None, timeout=None):
    """(number of (pattern, tau) cells compared, list of disagreements)."""
    ours = run_ours(vlab, edges, work / "ours")
    top = max(ours.values(), default=0)
    cells, bad = {}, []
    for tau in (taus or range(1, top + 2)):
        theirs = run_grami(vlab, edges, tau, work / f"grami_{tau}", timeout)
        mine = {k for k, s in ours.items() if s >= tau}
        if WITH_PATH:
            restricted, other = run_grami(vlab, edges, tau, work / f"grami_path_{tau}", timeout, path_only=True)
            if other:
                bad.append((tau, f"{other} reported patterns are not directed paths of <= {MAX_LENGTH} edges", "GraMi-path"))
            for k in sorted(restricted ^ theirs):
                bad.append((tau, k, "GraMi-path only" if k in restricted else "GraMi only, missing in GraMi-path"))
        for k in mine | theirs:
            m = (len(k.split()) - 2) // 2
            cells[m] = cells.get(m, 0) + 1
        for k in sorted(mine ^ theirs):
            bad.append((tau, k, "ours only" if k in mine else "GraMi only"))
    return cells, bad


def random_graph(rnd: random.Random):
    """Every other graph contains a directed path through all accounts, so chains of length 4 occur."""
    n = rnd.randint(3, 7)
    vlab = [rnd.randint(0, 1) for _ in range(n)]
    pairs = [(u, v) for u in range(n) for v in range(n) if u != v]
    chosen = set()
    if rnd.random() < 0.5:
        order = list(range(n))
        rnd.shuffle(order)
        chosen.update(zip(order, order[1:]))
    extra = rnd.randint(1, 4)
    chosen.update(rnd.sample([p for p in pairs if p not in chosen], min(extra, len(pairs) - len(chosen))))
    return vlab, [(u, v, rnd.randint(0, 1)) for u, v in sorted(chosen)]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("random")
    r.add_argument("--graphs", type=int, default=200)
    r.add_argument("--seed", type=int, default=20261016)
    r.add_argument("--with-path", action="store_true", help="also run GraMi-path and compare it with GraMi and ours")
    p = sub.add_parser("prefix")
    p.add_argument("--graph", required=True)
    p.add_argument("--transactions", type=int, required=True)
    p.add_argument("--tau", type=int, required=True)
    p.add_argument("--timeout", type=float, default=1800)
    args = ap.parse_args()
    global WITH_PATH
    WITH_PATH = getattr(args, "with_path", False)
    with tempfile.TemporaryDirectory() as tmp:
        if args.mode == "random":
            rnd = random.Random(args.seed)
            cells, graphs, bad = {}, 0, []
            for i in range(args.graphs):
                vlab, edges = random_graph(rnd)
                c, b = compare(vlab, edges, Path(tmp) / str(i))
                for m, x in c.items():
                    cells[m] = cells.get(m, 0) + x
                graphs += 1
                bad += [(i,) + x for x in b]
            print(f"random graphs: {graphs}; (chain, threshold) cells compared by length: "
                  f"{dict(sorted(cells.items()))}, total {sum(cells.values())}; disagreements: {len(bad)}")
        else:
            vlab, edges = prefix_graph(Path(args.graph), args.transactions)
            try:
                c, bad = compare(vlab, edges, Path(tmp), taus=[args.tau], timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print(f"prefix of {args.transactions} transactions: {len(vlab)} accounts, {len(edges)} distinct edges; "
                      f"tau {args.tau}; GraMi did not finish within {args.timeout:.0f} s - not compared")
                sys.exit(2)
            print(f"prefix of {args.transactions} transactions: {len(vlab)} accounts, {len(edges)} distinct edges; "
                  f"tau {args.tau}; chains compared by length: {dict(sorted(c.items()))}; disagreements: {len(bad)}")
        for x in bad[:20]:
            print("  disagreement:", x)
    sys.exit(1 if bad else 0)


def prefix_graph(path: Path, k: int):
    """The first k transactions of a .tg file (time order), self-transactions dropped, parallel ones merged."""
    ids, labels, edges, seen = {}, [], [], set()
    acc_label = {}
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t or t[0] == "#":
                continue
            if t[0] == "v":
                acc_label[t[1]] = int(t[2])
            elif t[0] == "e":
                if k == 0:
                    break
                k -= 1
                u, v, l = t[1], t[2], int(t[3])
                if u == v:
                    continue
                for a in (u, v):
                    if a not in ids:
                        ids[a] = len(labels)
                        labels.append(acc_label[a])
                e = (ids[u], ids[v])
                if e not in seen:
                    seen.add(e)
                    edges.append((e[0], e[1], l))
    return labels, edges


if __name__ == "__main__":
    main()
