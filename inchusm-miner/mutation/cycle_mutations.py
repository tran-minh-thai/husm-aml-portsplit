#!/usr/bin/env python3
"""Fault injection for the tests of the cycle membership program (edu.huflit.inchusm.cycles).

Same procedure as chain_mutations.py: copy the module, inject one fault at a time, run the cycle tests;
every fault must make them fail, and a fault whose text is not found exactly once, or that does not compile,
is reported as not landed. Faults that cannot change the output (for instance the choice of orientation)
are not listed: they would survive by design.

Run (from anywhere): python inchusm-miner/mutation/cycle_mutations.py
Exit status 0 only if the unmodified copy passes and every fault is caught.
"""
from __future__ import annotations

import sys

import chain_mutations as cm

CYCLES = "src/main/java/edu/huflit/inchusm/cycles/CycleMembership.java"

FAULTS = [
    (CYCLES, "table(v, 2, io, ia);", "table(v, 1, io, ia);",
     "length 4 keeps one second-to-last account per tabulated account instead of two"),
    (CYCLES, "table(v, 3, io, ia);", "table(v, 2, io, ia);",
     "length 5 keeps two second-to-last accounts per tabulated account instead of three"),
    (CYCLES, "if (c > 0 && y0[z] != a && y0[z] != b) return y0[z];", "if (c > 0 && y0[z] != b) return y0[z];",
     "the closing account may repeat an account of the leaving path"),
    (CYCLES, "for (int i = 1; i < c.length; i++) r[i] = c[c.length - i];", "for (int i = 1; i < c.length; i++) r[i] = c[i];",
     "a cycle found on the reverse graph is not read backwards"),
    (CYCLES, "if (outOff[v + 1] == outOff[v] || isSet(in, v)) return;",
     "if (outOff[v + 1] == outOff[v] || inOff[v + 1] - inOff[v] < 2 || isSet(in, v)) return;",
     "accounts with one incoming transaction are never searched"),
    (CYCLES, "for (int a : cycle) set(in, a);", "for (int a : cycle) set(in, (a + 1) % n);",
     "the accounts of a cycle found are recorded at the wrong indices"),
    (CYCLES, "if (z == a || z == b || z == u || !contains(oa, from, to, z)) continue;",
     "if (z == a || z == b || z == u) continue;",
     "the join through the table does not check that the transaction exists"),
    (CYCLES, "                    if (z == v) continue;\n", "",
     "the table may contain v itself"),
    (CYCLES, "if (x2 == v) continue;", "if (x2 == v && x2 < 0) continue;",
     "a leaving path of length 5 may return to v early"),
    (CYCLES, "if (w != x && contains(oa, oFrom, oTo, w)) return new int[] {v, x, w};",
     "if (w != x && contains(oa, oFrom, oTo - 1, w)) return new int[] {v, x, w};",
     "length 3 misses the last transaction of an account"),
    (CYCLES, "if (s.intValue() == d.intValue()) {", "if (s.intValue() == d.intValue() && s < 0) {",
     "self-transactions are kept"),
    (CYCLES, "low[v] = Math.min(low[v], index[w]);", "low[v] = Math.min(low[v], low[v]);",
     "strongly connected components ignore back edges, so edges on cycles are removed"),
]


def main() -> int:
    cm.FAULTS = FAULTS
    cm.TESTS = "CycleMembershipTest"
    return cm.main()


if __name__ == "__main__":
    sys.exit(main())
