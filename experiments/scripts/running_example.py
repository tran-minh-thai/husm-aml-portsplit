"""The running example of the manuscript: a ten-transaction graph, its chains, images and supports, and the slot filter.

Every quantity is computed twice: by a brute force written from the definitions (occurrence, validity, image, support,
and the pairwise slot filter on a path), and by the miner (ChainMain) on the same graph; the two are compared and the
run fails on any difference. Writes running_example.tg (the graph), running_example.json (graph, occurrences, images,
supports, filter result) and the miner's pattern table.

Run: python experiments/scripts/running_example.py
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JAR = ROOT / "inchusm-miner" / "target" / "inchusm-miner-0.1.0-SNAPSHOT.jar"
OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"
THETA, TAU, MAX_LENGTH, MIN_CYCLE = 0.2, 2, 4, 3

ACCOUNTS = "abcdghmw"
# (source, target, amount); one account label and one transaction label (labelling S)
EDGES = [("a", "b", 100), ("b", "c", 95), ("c", "d", 90), ("d", "a", 75),
         ("g", "h", 50), ("h", "c", 48), ("h", "m", 200), ("m", "w", 190),
         ("b", "g", 40), ("w", "g", 30)]
FILTER_FROM = 3       # slot filter from this candidate length in the step-by-step run of the level-wise algorithm


def cons(x: float, y: float) -> bool:
    return (1 - THETA) * x <= y <= (1 + THETA) * x


def occurrences(m: int, closed: bool):
    """Every (accounts, edge indices, valid, reason) of the chain of length m with one label, from the definition."""
    out = []
    for seq in itertools.product(range(len(EDGES)), repeat=m):
        if any(EDGES[seq[i]][1] != EDGES[seq[i + 1]][0] for i in range(m - 1)):
            continue
        acc = [EDGES[seq[0]][0]] + [EDGES[i][1] for i in seq]
        if closed:
            if acc[-1] != acc[0] or len(set(acc[:-1])) != m:
                continue
        elif len(set(acc)) != m + 1:
            continue
        bad = [i for i in range(m - 1) if not cons(EDGES[seq[i]][2], EDGES[seq[i + 1]][2])]   # root pair never checked
        out.append({"accounts": acc, "edges": list(seq), "valid": not bad,
                    "violations": [[EDGES[seq[i]][2], EDGES[seq[i + 1]][2], acc[i + 1]] for i in bad]})
    return out


def images(occ, m: int, closed: bool):
    positions = range(m) if closed else range(m + 1)
    return [sorted({o["accounts"][p] for o in occ if o["valid"]}) for p in positions]


def slot_filter(prefix_images: list):
    """Slot filtering of the candidate of length 3 joined from the chain of length 2 with itself (prefix P, suffix Q), as in
    the manuscript: domains D(0) = Img_P(0), D(i) = Img_P(i) & Img_Q(i-1) for 0 < i < 3, D(3) = Img_Q(2); slots
    S_s = {e: s(e) in D(s-1), t(e) in D(s)}; forward F_1 = S_1, F_{s+1} = {f in S_{s+1}: exists e in F_s, e >< f};
    backward B_3 = F_3, B_s = {e in F_s: exists f in B_{s+1}, e >< f}; e >< f iff t(e) = s(f) and cons(a(e), a(f)).
    Returns the domains, the kept slots and the accounts kept per position."""
    img = [set(x) for x in prefix_images]
    D = [img[0], img[1] & img[0], img[2] & img[1], img[2]]
    step = lambda e, f: EDGES[e][1] == EDGES[f][0] and cons(EDGES[e][2], EDGES[f][2])
    S = [{k for k, (u, v, _) in enumerate(EDGES) if u in D[s] and v in D[s + 1]} for s in range(3)]
    F = [S[0]]
    for t in range(1, 3):
        F.append({f for f in S[t] if any(step(e, f) for e in F[t - 1])})
    B = [None, None, F[2]]
    for t in (1, 0):
        B[t] = {e for e in F[t] if any(step(e, f) for f in B[t + 1])}
    kept = [sorted({EDGES[e][0] for e in B[0]})] + [sorted({EDGES[e][1] for e in B[t]}) for t in range(3)]
    return [sorted(d) for d in D], [sorted(b) for b in B], kept


def level_wise_trace() -> dict:
    """The level-wise algorithm on this graph, step by step, in the order the miner (ChainMiner) uses: positions by
    increasing domain size then index, accounts by increasing id (the order of ACCOUNTS), the forward phase of a pinned
    search before the backward one, transactions by increasing amount then id. One label, so every level holds one open
    chain and every candidate joins it with itself. Returns every candidate with its domains, the accounts the slot
    filter rejects, every pinned search with its result, and the outcome."""
    idx = {v: i for i, v in enumerate(ACCOUNTS)}
    order = lambda ids: sorted(ids, key=lambda e: (EDGES[e][2], e))
    out_of = {v: order([e for e, x in enumerate(EDGES) if x[0] == v]) for v in ACCOUNTS}
    into = {v: order([e for e, x in enumerate(EDGES) if x[1] == v]) for v in ACCOUNTS}
    img = [sorted({s for s, _, _ in EDGES}, key=idx.get), sorted({t for _, t, _ in EDGES}, key=idx.get)]
    levels, trace = [img], []
    for k in range(1, MAX_LENGTH):
        P = levels[-1]
        found_open = None
        for closed in (False, True):
            if closed and k + 1 < MIN_CYCLE:
                continue
            m = k + 1
            if closed:
                dom = [sorted(set(P[0]) & set(P[k]), key=idx.get)] + [sorted(set(P[i]) & set(P[i - 1]), key=idx.get) for i in range(1, k + 1)]
            else:
                dom = [P[0]] + [sorted(set(P[i]) & set(P[i - 1]), key=idx.get) for i in range(1, k + 1)] + [P[k]]
            pos = lambda s: 0 if closed and s == m else s
            entry = {"length": m, "kind": "closed" if closed else "open", "domains": dom, "rejected": [], "searches": []}
            trace.append(entry)
            if min(len(d) for d in dom) < TAU:
                entry["outcome"] = "domain"
                continue
            rejected = [set() for _ in dom]
            slots = None
            if m >= FILTER_FROM:
                slots = [None] + [{e for e, (u, v, _) in enumerate(EDGES) if u in dom[s - 1] and v in dom[pos(s)]} for s in range(1, m + 1)]
                entry["initial_slots"] = [None] + [sorted(slots[s]) for s in range(1, m + 1)]
                step = lambda e, f: EDGES[e][1] == EDGES[f][0] and cons(EDGES[e][2], EDGES[f][2])
                rounds = []
                while True:
                    before = [set(x) for x in slots[1:]]
                    for s in range(2, m + 1):
                        slots[s] = {f for f in slots[s] if any(step(e, f) for e in slots[s - 1])}
                    after_forward = [sorted(x) for x in slots[1:]]
                    for s in range(m - 1, 0, -1):
                        slots[s] = {e for e in slots[s] if any(step(e, f) for f in slots[s + 1])}
                    if closed:
                        z = {EDGES[e][0] for e in slots[1]} & {EDGES[e][1] for e in slots[m]}
                        slots[1] = {e for e in slots[1] if EDGES[e][0] in z}
                        slots[m] = {e for e in slots[m] if EDGES[e][1] in z}
                    rounds.append({"after_forward": after_forward})
                    if [set(x) for x in slots[1:]] == before:
                        break
                entry["filter_rounds"] = rounds
                kept = [{EDGES[e][0] for e in slots[1]}] + [{EDGES[e][1] for e in slots[q]} for q in range(1, len(dom))]
                for q in range(len(dom)):
                    rejected[q] = {v for v in dom[q] if v not in kept[q]}
                entry["rejected"] = [sorted(r, key=idx.get) for r in rejected]
                if any(len(dom[q]) - len(rejected[q]) < TAU for q in range(len(dom))):
                    entry["outcome"] = "filter"
                    continue
            witnessed = [set() for _ in dom]

            def find(pin, u):
                vert = [None] * len(dom)
                tx = [None] * (m + 1)
                vert[pin] = u

                def admissible(q, v):
                    return v in dom[q] and v not in rejected[q] and v not in vert

                def right(s):
                    if s > m:
                        return left(pin)
                    for e in out_of[vert[s - 1]]:
                        if s - 1 > pin and not cons(EDGES[tx[s - 1]][2], EDGES[e][2]):
                            continue
                        if slots is not None and e not in slots[s]:
                            continue
                        t, q = EDGES[e][1], pos(s)
                        assign = vert[q] is None
                        if (not admissible(q, t)) if assign else t != vert[q]:
                            continue
                        if assign:
                            vert[q] = t
                        tx[s] = e
                        if right(s + 1):
                            return True
                        tx[s] = None
                        if assign:
                            vert[q] = None
                    return False

                def left(s):
                    if s < 1:
                        return True
                    for e in into[vert[s]]:
                        if s <= m - 1 and tx[s + 1] is not None and not cons(EDGES[e][2], EDGES[tx[s + 1]][2]):
                            continue
                        if slots is not None and e not in slots[s]:
                            continue
                        f, q = EDGES[e][0], s - 1
                        assign = vert[q] is None
                        if (not admissible(q, f)) if assign else f != vert[q]:
                            continue
                        if assign:
                            vert[q] = f
                        tx[s] = e
                        if left(s - 1):
                            return True
                        tx[s] = None
                        if assign:
                            vert[q] = None
                    return False

                return list(vert) if right(pin + 1) else None

            stopped = False
            for q in sorted(range(len(dom)), key=lambda q: (len(dom[q]), q)):
                for u in dom[q]:
                    if u in witnessed[q] or u in rejected[q]:
                        continue
                    occ = find(q, u)
                    entry["searches"].append({"position": q, "account": u, "occurrence": occ})
                    if occ:
                        for r, v in enumerate(occ):
                            witnessed[r].add(v)
                    else:
                        rejected[q].add(u)
                        if len(dom[q]) - len(rejected[q]) < TAU:
                            stopped = True
                            break
                if stopped:
                    break
            if stopped:
                entry["outcome"] = "search"
                continue
            entry["outcome"] = "frequent"
            entry["images"] = [sorted(w, key=idx.get) for w in witnessed]
            if not closed:
                found_open = entry["images"]
        if found_open is None:
            break
        levels.append(found_open)
    return {"filter_from": FILTER_FROM, "level_one_images": img, "candidates": trace,
            "searches": sum(len(c["searches"]) for c in trace),
            "found": sum(1 for c in trace for x in c["searches"] if x["occurrence"]),
            "failed": sum(1 for c in trace for x in c["searches"] if not x["occurrence"])}


def miner_counters(graph: Path) -> dict:
    """The miner's machine-independent counters on this graph with the slot filter from FILTER_FROM."""
    out = subprocess.run(["java", "-cp", str(JAR), "edu.huflit.inchusm.chain.ChainMain", "--graph", str(graph), "--theta", str(THETA),
                          "--min-support", str(TAU), "--max-length", str(MAX_LENGTH), "--min-cycle", str(MIN_CYCLE),
                          "--consistency-from", str(FILTER_FROM), "--out", str(OUT / "patterns_running_example_level_wise.tsv")],
                         check=True, capture_output=True, text=True).stdout
    found = {}
    for key in ("searches", "found", "failed"):
        m = re.search(rf"\b{key}=(\d+)", out)
        if not m:
            sys.exit(f"miner output has no counter {key}")
        found[key] = int(m.group(1))
    return found


def write_graph(path: Path) -> None:
    lines = ["# running example of the manuscript: accounts " + ", ".join(ACCOUNTS) + " are 1..8, one label (labelling S)"]
    lines += [f"v {ACCOUNTS.index(v) + 1} 0" for v in ACCOUNTS]
    lines += [f"e {ACCOUNTS.index(s) + 1} {ACCOUNTS.index(t) + 1} 0 {a}" for s, t, a in EDGES]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    graph = OUT / "running_example.tg"
    write_graph(graph)
    mined_at = {}
    for tau in (TAU, 1):          # tau = 1 also reports the closed chain, whose support is below TAU
        table = OUT / f"patterns_running_example_tau{tau}.tsv"
        subprocess.run(["java", "-cp", str(JAR), "edu.huflit.inchusm.chain.ChainMain", "--graph", str(graph), "--theta", str(THETA),
                        "--min-support", str(tau), "--max-length", str(MAX_LENGTH), "--min-cycle", str(MIN_CYCLE),
                        "--out", str(table)], check=True, stdout=subprocess.DEVNULL)
        mined_at[tau] = {(r["kind"], int(r["length"])): r for r in csv.DictReader(open(table), delimiter="\t")}
    result = {"theta": THETA, "tau": TAU, "max_length": MAX_LENGTH, "min_cycle": MIN_CYCLE, "account_order": ACCOUNTS,
              "edges": [{"source": s, "target": t, "amount": a} for s, t, a in EDGES], "chains": []}
    for closed in (False, True):
        for m in range(MIN_CYCLE if closed else 1, MAX_LENGTH + 1):
            occ = occurrences(m, closed)
            img = images(occ, m, closed)
            result["chains"].append({"kind": "closed" if closed else "open", "length": m, "occurrences": occ,
                                     "images": img, "support": min(len(x) for x in img)})
    compared = 0
    for tau, mined in mined_at.items():
        frequent = {(c["kind"], c["length"]) for c in result["chains"] if c["support"] >= tau}
        if set(mined) != frequent:
            sys.exit(f"tau {tau}: frequent chains, brute force {sorted(frequent)}, miner {sorted(mined)}")
        for c in result["chains"]:
            if (c["kind"], c["length"]) not in frequent:
                continue
            got, sizes = mined[(c["kind"], c["length"])], [len(x) for x in c["images"]]
            if int(got["support"]) != c["support"] or [int(x) for x in got["image_counts"].split(",")][:len(sizes)] != sizes:
                sys.exit(f"tau {tau} {c['kind']} {c['length']}: brute force {c['support']} {sizes}, miner {got['support']} {got['image_counts']}")
            compared += 1
    if compared == 0 or not any(k[0] == "closed" for k in mined_at[1]):
        sys.exit(f"vacuous comparison: {compared} chains compared, closed chains at tau 1: {[k for k in mined_at[1] if k[0] == 'closed']}")
    frequent = {(c["kind"], c["length"]) for c in result["chains"] if c["support"] >= TAU}
    img2 = next(c["images"] for c in result["chains"] if c["kind"] == "open" and c["length"] == 2)
    domains, slots, kept = slot_filter(img2)
    img3 = next(c["images"] for c in result["chains"] if c["kind"] == "open" and c["length"] == 3)
    for p in range(4):
        if not set(img3[p]) <= set(kept[p]):
            sys.exit(f"the filter removed an image account at position {p}: images {img3[p]}, kept {kept[p]}")
    result["filter_length_3"] = {"domains": domains, "slots": slots, "kept_accounts": kept,
                                 "removed": [[p, v] for p in range(4) for v in domains[p] if v not in kept[p]]}
    lw = level_wise_trace()
    counters = miner_counters(graph)
    if {k: lw[k] for k in counters} != counters:
        sys.exit(f"level-wise run: step-by-step {[lw[k] for k in counters]}, miner {counters}")
    frequent_lw = {(c["kind"], c["length"]) for c in lw["candidates"] if c["outcome"] == "frequent"} | {("open", 1)}
    if frequent_lw != frequent:
        sys.exit(f"level-wise run reports {sorted(frequent_lw)}, the definitions {sorted(frequent)}")
    for c in lw["candidates"]:
        if c["outcome"] == "frequent":
            ref = next(x for x in result["chains"] if x["kind"] == c["kind"] and x["length"] == c["length"])
            if c["images"] != [sorted(x, key=ACCOUNTS.index) for x in ref["images"]]:
                sys.exit(f"level-wise run: images of {c['kind']} {c['length']} differ from the definitions")
    lw["miner_counters"] = counters
    result["level_wise"] = lw
    (OUT / "running_example.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"theta {THETA}, tau {TAU}; frequent chains, brute force = miner: {sorted(frequent)}; {compared} chain tables compared (tau {TAU} and 1)")
    for c in result["chains"]:
        valid = sum(o["valid"] for o in c["occurrences"])
        print(f"  {c['kind']} m={c['length']}: {len(c['occurrences'])} occurrences, {valid} valid, images {c['images']}, support {c['support']}")
    print(f"  slot filter, length 3: domains {domains}; kept {kept}; removed {result['filter_length_3']['removed']}")
    print(f"  level-wise run (filter from length {FILTER_FROM}): {lw['searches']} pinned searches, {lw['found']} found, "
          f"{lw['failed']} failed; the miner: {counters}")
    for c in lw["candidates"]:
        print(f"    {c['kind']} {c['length']}: {c['outcome']}, {len(c['searches'])} searches, rejected by the filter {c['rejected']}")


if __name__ == "__main__":
    main()
