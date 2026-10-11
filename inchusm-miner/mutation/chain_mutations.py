#!/usr/bin/env python3
"""Fault injection for the conserved flow chain miner's tests.

A test suite that has never failed has not shown it can. This copies the miner module to a
temporary directory, injects one fault at a time into the miner's source, and runs the chain
tests on each copy. Every fault must make the tests fail; a fault that only breaks compilation,
or that is not found exactly once in the source, is reported as an injection that did not land
rather than as a catch.

Run (from anywhere): python inchusm-miner/mutation/chain_mutations.py
Exit status 0 only if the unmodified copy passes and every fault is caught.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

MODULE = pathlib.Path(__file__).resolve().parents[1]
MINER = "src/main/java/edu/huflit/inchusm/chain/ChainMiner.java"
GRAPH = "src/main/java/edu/huflit/inchusm/chain/TransactionGraph.java"
CONS = "src/main/java/edu/huflit/inchusm/chain/Consistency.java"
ENUM = "src/main/java/edu/huflit/inchusm/chain/ChainEnumerator.java"
TESTS = "Chain*Test,TransactionGraphTest,ConsistencyTest"  # Chain*Test also matches the enumerator tests

# (file, text to replace, replacement, what the fault means)
FAULTS = [
    (ENUM, "if (marked[v]) hits[0]++;", "if (!marked[v]) hits[0]++;",
     "enumerator counts unmarked accounts as marked"),
    (ENUM, "markedCounts[i] = hits[0];", "markedCounts[counts.length - 1 - i] = hits[0];",
     "enumerator writes marked counts to the wrong positions"),
    (MINER, "b -> b >= lo", "b -> b > lo",
     "lower conservation bound made strict when extending to the right"),
    (MINER, "a -> (1 + theta) * a >= b", "a -> (1 + theta) * a > b",
     "upper conservation bound made strict when extending to the left"),
    (MINER, "if (w == v) return false;", "if (w == v && w < 0) return false;",
     "accounts of an occurrence no longer required to be distinct"),
    (MINER, "if (assign ? !admissible(tp, t) : t != vert[tp]) continue;",
     "if (assign ? !admissible(tp, t) : false) continue;",
     "a closed chain may end at any account instead of its origin"),
    (MINER, "min = Math.min(min, counts[i]);", "min = Math.max(min == Integer.MAX_VALUE ? 0 : min, counts[i]);",
     "support taken as the largest image set instead of the smallest"),
    (MINER, "elab[k] = q.pattern().elab()[k - 1];", "elab[k] = p.pattern().elab()[k - 1];",
     "join takes the new transaction label from the wrong chain"),
    (MINER, "if (--alive[pos] < params.minSupport()) {", "if (--alive[pos] <= params.minSupport()) {",
     "early stop discards chains whose support equals the threshold"),
    (MINER, "if (s <= m - 1 && tx[s + 1] >= 0) {", "if (s < m - 1 && tx[s + 1] >= 0) {",
     "conservation at the last intermediate account skipped in the leftward search"),
    (MINER, "if (s - 1 > pin) {", "if (s - 1 > pin + 1) {",
     "conservation skipped at the first intermediate account reached when extending to the right"),
    (GRAPH, "int c = Double.compare(amount[x], amount[y]);", "int c = Double.compare(amount[y], amount[x]);",
     "transactions sorted by decreasing amount, so the range search reads the wrong slots"),
    (GRAPH, "while (i < mid && j < to) a[k++] = compareTx(buf[i], buf[j]) <= 0 ? buf[i++] : buf[j++];",
     "while (i < mid && j < to) a[k++] = compareTx(buf[i], buf[j]) >= 0 ? buf[i++] : buf[j++];",
     "merge step of the sort reversed, so accounts with many transactions are out of order"),
    (MINER, "for (int i = 1; i < p.length(); i++) for (int v : img[i]) mid.set(v);",
     "for (int i = 0; i < p.length(); i++) for (int v : img[i]) mid.set(v);",
     "the first account of a chain reported as intermediate"),
    (MINER, "for (int v : img[i]) if (marked[v]) markedCounts[i]++;",
     "for (int v : img[i]) if (!marked[v]) markedCounts[i]++;",
     "marked counts count the unmarked accounts"),
    (CONS, "if (k < hi && (1 - theta) * g.amount(p[k]) <= b) out[n++] = e;",
     "if (k < hi && (1 - theta) * g.amount(p[k]) < b) out[n++] = e;",
     "forward consistency drops transactions at the lower conservation bound"),
    (CONS, "if (k < hi && g.amount(q[k]) <= (1 + theta) * a) out[n++] = e;",
     "if (k < hi && g.amount(q[k]) < (1 + theta) * a) out[n++] = e;",
     "backward consistency drops transactions at the upper conservation bound"),
    (MINER, "slots[m] = Consistency.byEnd(g, slots[m], origins, false);",
     "slots[m] = Consistency.byEnd(g, slots[m], origins, true);",
     "closed-chain consistency filters the returning transaction by its source instead of its target"),
    (MINER, "for (int s = pin - 1; s >= 1; s--) a[s] = Consistency.expand(g, a[s + 1], elab[s - 1], slots[s], theta, false);",
     "for (int s = pin - 1; s >= 1; s--) a[s] = Consistency.expand(g, a[s + 1], elab[s], slots[s], theta, false);",
     "local propagation of a closed chain expands leftwards with the wrong transaction label"),
    (ENUM, "for (int j = 1; j <= k; j++) {", "for (int j = 2; j <= k; j++) {",
     "enumeration baseline may revisit the second account of an occurrence"),
    (ENUM, "b -> b >= (1 - theta) * a", "b -> b > (1 - theta) * a",
     "enumeration baseline makes the lower conservation bound strict"),
    (ENUM, "double a = g.amount(anchored ? tx[1] : tx[k]);", "double a = g.amount(anchored ? tx[k] : tx[k]);",
     "the anchored rule compares with the previous transaction instead of the first"),
    (ENUM, "for (int i = 1; i < m; i++) a.images[i].forEach(mid::set);",
     "for (int i = 0; i < m; i++) a.images[i].forEach(mid::set);",
     "the enumeration baseline reports the first account of a chain as intermediate"),
    (CONS, "a -> (1 + theta) * a >= b", "a -> (1 + theta) * a > b",
     "leftward local propagation makes the upper conservation bound strict"),
    (CONS, "b -> b >= lo", "b -> b > lo", "rightward local propagation makes the lower conservation bound strict"),
]


def mvn() -> str:
    for name in ("mvn", "mvn.cmd"):
        path = shutil.which(name)
        if path:
            return path
    sys.exit("Maven not found on PATH")


def run_tests(copy: pathlib.Path) -> tuple[str, str]:
    """('pass' | 'fail' | 'compile', last lines of output)."""
    proc = subprocess.run([mvn(), "-B", "-q", "test", f"-Dtest={TESTS}"], cwd=copy,
                          capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    tail = "\n".join(out.strip().splitlines()[-3:])
    if proc.returncode == 0:
        return "pass", tail
    if "COMPILATION ERROR" in out or "Compilation failure" in out:
        return "compile", tail
    return "fail", tail


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        copy = pathlib.Path(tmp) / "module"
        copy.mkdir()
        shutil.copy2(MODULE / "pom.xml", copy / "pom.xml")
        shutil.copytree(MODULE / "src", copy / "src")

        status, tail = run_tests(copy)
        print(f"unmodified copy: {status}")
        if status != "pass":
            print(tail)
            return 1

        caught = 0
        not_landed = []
        survived = []
        for i, (rel, old, new, meaning) in enumerate(FAULTS, 1):
            path = copy / rel
            original = path.read_text(encoding="utf-8")
            occurrences = original.count(old)
            if occurrences != 1:
                not_landed.append(f"fault {i}: text found {occurrences} times, expected once")
                print(f"fault {i}: NOT LANDED ({occurrences} matches) - {meaning}")
                continue
            mutated = original.replace(old, new)
            if mutated == original or new not in mutated:
                not_landed.append(f"fault {i}: source unchanged after replacement")
                continue
            path.write_text(mutated, encoding="utf-8")
            try:
                status, tail = run_tests(copy)
            finally:
                path.write_text(original, encoding="utf-8")
            if status == "fail":
                caught += 1
                print(f"fault {i}: caught - {meaning}")
            elif status == "compile":
                not_landed.append(f"fault {i}: does not compile")
                print(f"fault {i}: NOT LANDED (does not compile) - {meaning}\n{tail}")
            else:
                survived.append(f"fault {i}: {meaning}")
                print(f"fault {i}: SURVIVED - {meaning}")

        print(f"\n{caught} of {len(FAULTS)} faults caught; {len(survived)} survived; "
              f"{len(not_landed)} did not land")
        return 0 if caught == len(FAULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
