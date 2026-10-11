"""Hand-built cases for the amount-rule locality checks, each paired with a variant that must fail."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import amount_rule_locality as arl  # noqa: E402


def test_closure_and_minimal_forbidden_lengths():
    p = arl.pairwise(0.2)
    assert not p((100, 130)) and p((100, 110, 120))
    assert arl.closure(p, 4)["factor_closed"]
    assert arl.minimal_forbidden_lengths(p, 4) == {2}

    a = arl.anchored(0.2)
    assert a((100, 80, 120)) and not a((80, 120))
    c = arl.closure(a, 4)
    assert c["prefix_closed"] and not c["suffix_closed"]

    w = arl.window_range(0.2, 3)
    assert arl.minimal_forbidden_lengths(w, 5) == {2, 3}

    only3 = arl.forbidden_factors([(0, 0, 0)], "test")
    assert arl.minimal_forbidden_lengths(only3, 5) == {3}


def test_join_is_exact_at_one_length_but_not_beyond():
    only3 = arl.forbidden_factors([(0, 0, 0)], "test")
    assert arl.join_exact(only3, 2)          # no minimal forbidden factor of length 2
    assert not arl.join_exact(only3, 3)      # (0, 0, 0): prefix and suffix valid, itself invalid
    assert arl.join_exact(only3, 4)
    assert arl.width(arl.minimal_forbidden_lengths(only3, 5)) == 3
    # fault: checking the prefix alone is wrong for the pairwise rule
    p = arl.pairwise(0.2)
    assert arl.join_exact(p, 3)
    assert not arl.join_exact(p, 3, use_suffix=False)


def test_two_sided_pruning_loses_an_anchored_chain_that_prefix_growth_keeps():
    vlab = ["A", "B", "B", "A"]
    edges = [(0, 1, 100), (1, 2, 80), (2, 3, 120)]
    a = arl.anchored(0.2)
    sup = arl.supports(arl.valid_paths(vlab, edges, a, 3))
    assert sup[("A", "B", "B", "A")] == 1 and ("B", "B", "A") not in sup
    assert ("A", "B", "B", "A") not in arl.level_wise(sup, 1, 3)
    assert ("A", "B", "B", "A") in arl.level_wise(sup, 1, 3, two_sided=False)

    p = arl.pairwise(0.2)
    flat = [(0, 1, 100), (1, 2, 90), (2, 3, 100)]
    sup = arl.supports(arl.valid_paths(vlab, flat, p, 3))
    assert arl.level_wise(sup, 1, 3) == {k for k, s in sup.items() if s >= 1}


def test_pair_filter_is_inexact_for_width_three_and_the_width_filter_is_exact():
    w = arl.window_range(0.2, 3)
    slots = [[(0, 0, 60)], [(0, 0, 70)], [(0, 0, 80)]]
    truth = arl.on_valid_walk(slots, w)
    assert truth == [set(), set(), set()]
    assert arl.tuple_filter(slots, w, 2) == [{0}, {0}, {0}]     # fault: windows one too short
    assert arl.tuple_filter(slots, w, 3) == truth


def test_parallel_filtering_is_inexact_where_sequential_is_exact():
    p = arl.pairwise(0.2)
    slots = [[(0, 0, 100)], [(0, 0, 100), (0, 1, 130)], [(1, 0, 130)]]
    truth = arl.on_valid_walk(slots, p)
    assert truth == [set(), set(), set()]
    assert arl.tuple_filter(slots, p, 2) == truth
    assert arl.tuple_filter(slots, p, 2, sequential=False) != truth


def test_random_checks_report_their_denominators():
    rng = random.Random(7)
    p = arl.pairwise(0.2)
    r = arl.filter_check(p, rng, 200, 2)
    assert r["instances"] == 200 and r["with_a_valid_walk"] > 100 and r["inexact"] == 0
    assert arl.filter_check(p, rng, 200, 2, sequential=False)["inexact"] > 0
    q = arl.pruning_check(p, rng, 50)
    assert q["frequent_chains"] > 100 and q["lost_chains"] == 0
