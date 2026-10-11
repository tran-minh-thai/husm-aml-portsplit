"""Hand-built cases for the degree and chance controls and for the simple rules."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chain_controls as cc  # noqa: E402
import chain_simple_baselines as sb  # noqa: E402


def test_degree_bins_count_transactions_and_skip_self():
    edges = pd.DataFrame({"src": [1, 1, 1, 2, 3], "dst": [2, 2, 3, 3, 3]})
    b = cc.degree_bins(edges)
    # account 1: out 3 -> bin 1; account 2: in 2 + out 1 = 3 -> bin 1; account 3: in 2 -> bin 1 (self dropped)
    assert b.to_dict() == {1: 1, 2: 1, 3: 1}
    edges = pd.DataFrame({"src": [1] * 8 + [2], "dst": list(range(10, 18)) + [1]})
    assert cc.degree_bins(edges)[1] == 3          # 9 transactions -> floor(log2 9) = 3


def test_matched_precision_draws_only_from_the_same_bin():
    bins = pd.Series({1: 0, 2: 0, 3: 5, 4: 5})
    # returned has one account of bin 0; the pool's only bin-0 account is marked, its bin-5 ones are not
    out = cc.matched_precision(np.array([2]), np.array([1, 3, 4]), bins, {1}, np.random.default_rng(0), 50)
    assert np.all(out == 1.0)
    out = cc.matched_precision(np.array([3]), np.array([1, 3, 4]), bins, {1}, np.random.default_rng(0), 50)
    assert np.all(out == 0.0)


def test_shuffled_marks_keep_the_count_in_every_bin():
    bins = pd.Series({i: i % 3 for i in range(300)})
    marked = {i for i in range(300) if i % 7 == 0}
    s = cc.shuffled_marks(bins, marked, np.random.default_rng(1))
    for b in range(3):
        members = {i for i in range(300) if i % 3 == b}
        assert len(members & s) == len(members & marked)
    assert s != marked


def test_simple_rules_by_hand():
    # a -> v (100), v -> c (90): v is one-in-one-out and conserves within 0.2
    # u -> w (100), w -> x (50), w -> y (60): w conserves totals (110/100) but is not one-in-one-out
    # p -> q (100), q -> r (10): q is one-in-one-out but does not conserve totals
    # s -> t (100), t -> z (125): out/in = 1.25 is outside [0.8, 1.2] (in/out = 0.8 would be inside)
    edges = pd.DataFrame({"src": ["a", "v", "u", "w", "w", "p", "q", "s", "t"],
                          "dst": ["v", "c", "w", "x", "y", "q", "r", "t", "z"],
                          "amount": [100, 90, 100, 50, 60, 100, 10.0, 100, 125]})
    one_one, totals = sb.simple_rules(edges)
    assert one_one == {"v", "q", "t"}
    assert totals == {"v", "w"}


def test_scores_by_hand():
    groups = [{1, 2}, {3}]                          # target C = {1, 2, 3}
    s = sb.scores({1, 3, 9, 10}, groups, np.random.default_rng(0))   # returned and target differ in size
    assert (s["tp"], s["returned"]) == (2, 4)
    assert s["precision"] == pytest.approx(2 / 4) and s["recall"] == pytest.approx(2 / 3)
    assert s["f1"] == pytest.approx(2 * (1 / 2) * (2 / 3) / (1 / 2 + 2 / 3))
