"""Which amount rules allow exact two-sided level-wise chain mining: brute-force checks on finite grids.

An amount rule R is a set of amount sequences (the amounts of consecutive transactions of a chain);
sequences of length <= 1 are always valid. R is factor-closed when every contiguous subsequence of a
valid sequence is valid. A minimal forbidden factor is an invalid sequence whose proper factors are all
valid; the width ell(R) is the largest length of a minimal forbidden factor. Three statements are
checked here, each against an independent brute force and each with rules for which it must fail:

  pruning     the level-wise miner that joins a frequent chain with a frequent chain sharing all but
              one transaction, and keeps a candidate only when its minimum-image support reaches the
              threshold, returns every frequent chain on every graph  <=>  R is factor-closed
  join        for factor-closed R: a sequence of length m + 1 is valid exactly when its prefix and
              suffix are  <=>  R has no minimal forbidden factor of length m + 1; so the join needs no
              amount check at every length > m  <=>  ell(R) <= m
  filter      for factor-closed R: the sequential forward-then-backward slot filter on consecutive
              pairs keeps exactly the transactions on some valid walk, on every instance  <=>
              ell(R) <= 2; with ell(R) = k, the same filter over walks of k - 1 transactions is exact

This is a check over finite grids and small instances, not a proof. Widths are lower bounds: a minimal
forbidden factor at the longest length tried is reported as possibly unbounded.

Run: python experiments/scripts/amount_rule_locality.py [--max-length 5] [--instances 300] [--seed 20261020]
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR") else ROOT / "experiments" / "results"

GRID = (60, 70, 80, 90, 100, 110, 120, 130, 140)
SYMBOLS = (0, 1, 2)


@dataclass(frozen=True)
class Rule:
    name: str
    alphabet: tuple
    accepts: Callable[[tuple], bool]

    def __call__(self, word: tuple) -> bool:
        return len(word) <= 1 or self.accepts(tuple(word))


def ratio_ok(x, y, theta):
    return (1 - theta) * x <= y <= (1 + theta) * x


def pairwise(theta):
    return Rule(f"pairwise theta={theta}", GRID, lambda w: all(ratio_ok(x, y, theta) for x, y in zip(w, w[1:])))


def anchored(alpha):
    return Rule(f"anchored alpha={alpha}", GRID, lambda w: all(ratio_ok(w[0], y, alpha) for y in w[1:]))


def window_range(delta, k):
    return Rule(f"range within every {k} consecutive, delta={delta}", GRID,
                lambda w: all(max(w[i:i + k]) <= (1 + delta) * min(w[i:i + k]) for i in range(max(1, len(w) - k + 1))))


def global_range(delta):
    return Rule(f"range over the whole chain, delta={delta}", GRID, lambda w: max(w) <= (1 + delta) * min(w))


def positional(thetas):
    return Rule(f"positional thresholds {thetas}", GRID,
                lambda w: all(ratio_ok(x, y, thetas[min(i, len(thetas) - 1)]) for i, (x, y) in enumerate(zip(w, w[1:]))))


def forbidden_factors(forbidden, tag):
    forbidden = frozenset(forbidden)

    def accepts(w):
        return not any(w[i:j] in forbidden for i in range(len(w)) for j in range(i + 2, len(w) + 1))
    return Rule(f"forbids {sorted(forbidden)} ({tag})", SYMBOLS, accepts)


def random_rule(rng, tag):
    """A rule with no structure: each sequence of length >= 2 is valid with probability 0.7."""
    salt = rng.randrange(1 << 30)
    return Rule(f"random rule {tag}", SYMBOLS, lambda w: random.Random(hash((salt, w))).random() < 0.7)


def rule_family(rng: random.Random, n_random: int = 6) -> list[Rule]:
    rules = [pairwise(0.0), pairwise(0.2), anchored(0.2), window_range(0.2, 3), global_range(0.2),
             positional((0.1, 0.3)), forbidden_factors([(0, 0, 0)], "only a length-3 factor")]
    for i in range(n_random):
        words = [tuple(rng.choice(SYMBOLS) for _ in range(rng.randint(2, 4))) for _ in range(rng.randint(1, 3))]
        rules.append(forbidden_factors(words, f"random {i}"))
    for i in range(n_random):
        rules.append(random_rule(rng, i))
    return rules


# ---------------------------------------------------------------- language properties

def words(alphabet, n):
    return itertools.product(alphabet, repeat=n)


def closure(rule: Rule, max_length: int) -> dict:
    prefix = suffix = True
    for n in range(2, max_length + 1):
        for w in words(rule.alphabet, n):
            if rule(w):
                prefix &= rule(w[:-1])
                suffix &= rule(w[1:])
    return {"prefix_closed": prefix, "suffix_closed": suffix, "factor_closed": prefix and suffix}


def minimal_forbidden_lengths(rule: Rule, max_length: int) -> set[int]:
    """Lengths n <= max_length at which some invalid sequence has every proper factor valid."""
    found = set()
    for n in range(2, max_length + 1):
        for w in words(rule.alphabet, n):
            if not rule(w) and all(rule(w[i:j]) for i in range(n) for j in range(i + 1, n + 1) if j - i < n):
                found.add(n)
                break
    return found


def width(lengths: set[int]) -> int:
    return max(lengths) if lengths else 1


def join_exact(rule: Rule, length: int, use_suffix: bool = True) -> bool:
    """A sequence of this length is valid exactly when its prefix (and suffix) are."""
    return all(rule(w) == (rule(w[:-1]) and (rule(w[1:]) if use_suffix else True))
               for w in words(rule.alphabet, length))


# ---------------------------------------------------------------- mining on small graphs

def random_graph(rng, alphabet, n_vertices=6, n_edges=13, labels="AB"):
    vlab = [rng.choice(labels) for _ in range(n_vertices)]
    edges = []
    for _ in range(n_edges):
        s = rng.randrange(n_vertices)
        t = rng.choice([v for v in range(n_vertices) if v != s])
        edges.append((s, t, rng.choice(alphabet)))
    return vlab, edges


def valid_paths(vlab, edges, rule, max_edges):
    """Every valid occurrence: (labels, accounts) of simple paths with 1..max_edges transactions."""
    out_edges = defaultdict(list)
    for s, t, a in edges:
        out_edges[s].append((t, a))
    found = []

    def grow(path, amounts):
        if amounts:
            found.append((tuple(vlab[v] for v in path), tuple(path)))
        if len(amounts) == max_edges:
            return
        for t, a in out_edges[path[-1]]:
            if t not in path and rule(amounts + (a,)):
                grow(path + [t], amounts + (a,))
    for v in range(len(vlab)):
        grow([v], ())
    return found


def supports(occurrences) -> dict:
    images = defaultdict(lambda: defaultdict(set))
    for labels, path in occurrences:
        for pos, v in enumerate(path):
            images[labels][pos].add(v)
    return {p: min(len(s) for s in img.values()) for p, img in images.items()}


def level_wise(support: dict, tau: int, max_edges: int, two_sided: bool = True) -> set:
    """Two-sided join with pruning: candidates of length m + 1 from frequent P, Q with P[1:] == Q[:-1].

    two_sided=False grows each frequent chain by one more account on the right only (prefix growth).
    """
    level = {p for p, s in support.items() if len(p) == 2 and s >= tau}
    result = set(level)
    for _ in range(max_edges - 1):
        cand = set()
        for p in level:
            for q in level:
                if p[1:] == q[:-1]:
                    cand.add(p + q[-1:])
        if not two_sided:
            cand = {p + (x,) for p in level for x in "AB"}
        level = {c for c in cand if support.get(c, 0) >= tau}
        result |= level
    return result


def pruning_check(rule, rng, instances, max_edges=4, two_sided=True) -> dict:
    lost = graphs = frequent = 0
    for _ in range(instances):
        vlab, edges = random_graph(rng, rule.alphabet)
        sup = supports(valid_paths(vlab, edges, rule, max_edges))
        for tau in (1, 2):
            truth = {p for p, s in sup.items() if s >= tau}
            mined = level_wise(sup, tau, max_edges, two_sided)
            assert mined <= truth
            graphs += 1
            frequent += len(truth)
            lost += len(truth - mined)
    return {"graphs": graphs, "frequent_chains": frequent, "lost_chains": lost}


# ---------------------------------------------------------------- slot filters on walks

def random_slots(rng, alphabet, m):
    """Slot i holds transactions (src, dst, amount) from layer i to layer i + 1; layers have 2 vertices.

    Amounts of one instance come from 3 consecutive alphabet values, so that many instances have a valid walk.
    """
    start = rng.randrange(len(alphabet) - 2)
    near = alphabet[start:start + 3]
    return [[(rng.randrange(2), rng.randrange(2), rng.choice(near)) for _ in range(rng.randint(1, 3))]
            for _ in range(m)]


def adjacent(e, f):
    return e[1] == f[0]


def on_valid_walk(slots, rule):
    keep = [set() for _ in slots]
    for walk in itertools.product(*[range(len(s)) for s in slots]):
        tx = [slots[i][j] for i, j in enumerate(walk)]
        if all(adjacent(x, y) for x, y in zip(tx, tx[1:])) and rule(tuple(t[2] for t in tx)):
            for i, j in enumerate(walk):
                keep[i].add(j)
    return keep


def tuple_filter(slots, rule, k, sequential=True):
    """Filter over walks of k - 1 transactions; k = 2 is the pair filter on single slots."""
    m, w = len(slots), max(1, k - 1)
    if m <= w:
        return on_valid_walk(slots, rule)
    layers = []
    for i in range(m - w + 1):
        layer = set()
        for walk in itertools.product(*[range(len(s)) for s in slots[i:i + w]]):
            tx = [slots[i + d][j] for d, j in enumerate(walk)]
            if all(adjacent(x, y) for x, y in zip(tx, tx[1:])) and rule(tuple(t[2] for t in tx)):
                layer.add(walk)
        layers.append(layer)

    def link(i, a, b):
        ta, tb = slots[i + w - 1][a[-1]], slots[i + w][b[-1]]
        amounts = tuple(slots[i + d][j][2] for d, j in enumerate(a)) + (tb[2],)
        return a[1:] == b[:-1] and adjacent(ta, tb) and rule(amounts)
    original = [set(x) for x in layers]
    for i in range(1, len(layers)):
        prev = layers[i - 1] if sequential else original[i - 1]
        layers[i] = {b for b in layers[i] if any(link(i - 1, a, b) for a in prev)}
    for i in range(len(layers) - 2, -1, -1):
        nxt = layers[i + 1] if sequential else original[i + 1]
        layers[i] = {a for a in layers[i] if any(link(i, a, b) for b in nxt)}
    keep = [set() for _ in slots]
    for i, layer in enumerate(layers):
        for walk in layer:
            for d, j in enumerate(walk):
                keep[i + d].add(j)
    return keep


def filter_check(rule, rng, instances, k, sequential=True) -> dict:
    wrong = total = nonempty = 0
    for _ in range(instances):
        slots = random_slots(rng, rule.alphabet, rng.randint(2, 5))
        truth = on_valid_walk(slots, rule)
        total += 1
        nonempty += any(truth)
        wrong += tuple_filter(slots, rule, k, sequential) != truth
    return {"instances": total, "with_a_valid_walk": nonempty, "inexact": wrong}


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-length", type=int, default=5)
    ap.add_argument("--instances", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    n = args.max_length
    rows, joins = [], []
    for rule in rule_family(rng):
        c = closure(rule, n)
        lengths = minimal_forbidden_lengths(rule, n) if c["factor_closed"] else set()
        ell = width(lengths) if c["factor_closed"] else None
        row = {"rule": rule.name, **c, "minimal_forbidden_lengths": " ".join(map(str, sorted(lengths))),
               "width_lower_bound": ell, "possibly_unbounded": bool(lengths) and max(lengths) == n}
        row.update({f"pruning_{k}": v for k, v in pruning_check(rule, rng, args.instances).items()})
        if c["prefix_closed"]:
            row["prefix_growth_lost_chains"] = pruning_check(rule, rng, args.instances, two_sided=False)["lost_chains"]
        if c["factor_closed"]:
            for m in range(1, n):
                exact = join_exact(rule, m + 1)
                exact_beyond = all(join_exact(rule, j) for j in range(m + 1, n + 1))
                joins.append({"rule": rule.name, "m": m, "join_exact_at_m_plus_1": exact,
                              "minimal_forbidden_at_m_plus_1": (m + 1) in lengths,
                              "join_exact_at_every_length_above_m": exact_beyond,
                              "width_at_most_m": ell <= m})
            row.update({f"pair_filter_{k}": v for k, v in filter_check(rule, rng, args.instances, 2).items()})
            if ell <= 4:
                row.update({f"width_filter_{k}": v for k, v in filter_check(rule, rng, args.instances, max(ell, 2)).items()})
        rows.append(row)
    table, join_table = pd.DataFrame(rows), pd.DataFrame(joins)
    table.to_csv(OUT / "rule_properties.csv", index=False)
    join_table.to_csv(OUT / "join_exactness.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"grid {GRID}, symbols {SYMBOLS}, sequences up to length {n}, {args.instances} instances per check, seed {args.seed}\n")
    print(table.to_string(index=False))

    closed, open_ = table[table["factor_closed"]], table[~table["factor_closed"]]
    print(f"\npruning: {len(closed)} factor-closed rules lose {int(closed['pruning_lost_chains'].sum())} chains over "
          f"{int(closed['pruning_frequent_chains'].sum())} frequent over {int(closed['pruning_graphs'].sum())} graph runs; {len(open_)} rules not factor-closed, "
          f"{int((open_['pruning_lost_chains'] > 0).sum())} of them lose at least one")
    a = join_table["join_exact_at_m_plus_1"] == join_table["minimal_forbidden_at_m_plus_1"]
    b = join_table["join_exact_at_every_length_above_m"] != join_table["width_at_most_m"]
    print(f"join: {len(join_table)} (rule, m) cases; exact-at-m+1 agrees with no-minimal-forbidden-at-m+1 in "
          f"{int((~a).sum())}; exact-at-every-length agrees with width<=m in {int((~b).sum())}; "
          f"cases with exact at m+1 but not beyond: "
          f"{int((join_table['join_exact_at_m_plus_1'] & ~join_table['join_exact_at_every_length_above_m']).sum())}")
    print(f"  exact true/false: {int(join_table['join_exact_at_m_plus_1'].sum())}/{int((~join_table['join_exact_at_m_plus_1']).sum())}")
    w2, w3 = closed[closed["width_lower_bound"] <= 2], closed[closed["width_lower_bound"] >= 3]
    print(f"pair filter: {len(w2)} rules of width <= 2 inexact on {int(w2['pair_filter_inexact'].sum())} of "
          f"{int(w2['pair_filter_instances'].sum())} instances ({int(w2['pair_filter_with_a_valid_walk'].sum())} with a valid walk); {len(w3)} rules of width >= 3, "
          f"{int((w3['pair_filter_inexact'] > 0).sum())} of them inexact somewhere")
    wf = closed[closed["width_lower_bound"] <= 4]
    print(f"width filter: {len(wf)} rules of width <= 4 inexact on {int(wf['width_filter_inexact'].sum())} of "
          f"{int(wf['width_filter_instances'].sum())} instances ({int(wf['width_filter_with_a_valid_walk'].sum())} with a valid walk)")


if __name__ == "__main__":
    main()
