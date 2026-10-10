"""The support/weight variance decomposition, checked against hand-computed cases."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inchusm_aml.eval.score_decomposition import variance_decomposition  # noqa: E402

E = np.e


def test_identity_holds_and_shares_sum_to_one():
    rng = np.random.default_rng(0)
    s = np.exp(rng.normal(size=200))
    w = np.exp(0.3 * np.log(s) + rng.normal(scale=0.5, size=200))
    d = variance_decomposition(s, w)
    assert d["var_log_score"] == pytest.approx(d["var_log_support"] + d["var_log_weight"] + 2 * d["cov_log"])
    assert d["share_support"] + d["share_weight"] == pytest.approx(1.0)
    assert d["degenerate"] == 0 and d["n"] == 200


def test_hand_computed_positive_covariance():
    # X = (0, 2), Y = (0, 1): Var X = 1, Var Y = 0.25, Cov = 0.5, Var Z = 2.25
    d = variance_decomposition([1, E ** 2], [1, E])
    assert d["var_log_support"] == pytest.approx(1.0)
    assert d["var_log_weight"] == pytest.approx(0.25)
    assert d["cov_log"] == pytest.approx(0.5)
    assert d["var_log_score"] == pytest.approx(2.25)
    assert d["share_weight"] == pytest.approx(0.75 / 2.25)
    assert d["share_support"] == pytest.approx(1.5 / 2.25)
    assert d["ratio_weight_no_cov"] == pytest.approx(0.25 / 2.25)


def test_negative_covariance_can_make_a_share_negative():
    # X = (0, 2), Y = (0, -1): Var X = 1, Var Y = 0.25, Cov = -0.5, Var Z = 0.25
    d = variance_decomposition([1, E ** 2], [1, 1 / E])
    assert d["share_weight"] == pytest.approx(-1.0)
    assert d["share_support"] == pytest.approx(2.0)
    assert d["ratio_weight_no_cov"] == pytest.approx(1.0)   # the old ratio reads "all utility"


def test_constant_weight_carries_nothing():
    d = variance_decomposition([1, 2, 5, 9], [3, 3, 3, 3])
    assert d["share_weight"] == pytest.approx(0.0, abs=1e-12)
    assert d["share_support"] == pytest.approx(1.0)


def test_perfect_cancellation_is_degenerate():
    s = np.array([1.0, 2.0, 4.0])
    d = variance_decomposition(s, 1.0 / s)          # score constant
    assert d["degenerate"] == 1 and np.isnan(d["share_weight"])
    assert d["var_log_support"] > 0


def test_single_pattern_is_degenerate():
    d = variance_decomposition([5.0], [2.0])
    assert d["degenerate"] == 1 and d["n"] == 1 and np.isnan(d["share_support"])


def test_rejects_nonpositive_and_mismatched_inputs():
    with pytest.raises(ValueError, match="positive"):
        variance_decomposition([1, 0], [1, 1])
    with pytest.raises(ValueError, match="shape"):
        variance_decomposition([1, 2], [1, 2, 3])


from inchusm_aml.eval.score_decomposition import order_statistics  # noqa: E402


def test_order_statistics_hand_case():
    # supports 1, 2, 10; weights 5, 1, 1 (R = 5). Pairs: (1,2): 5 vs 2 -> reversed, unlocked;
    # (1,10): 5 vs 10 -> not reversed, locked (10 > 5*1); (2,10): 2 vs 10 -> not reversed, unlocked.
    d = order_statistics([1, 2, 10], [5, 1, 1], R=5)
    assert d["pairs"] == 3 and d["comparable_pairs"] == 3
    assert d["locked_frac"] == pytest.approx(1 / 3)
    assert d["reversed_frac"] == pytest.approx(1 / 3)
    assert d["reversed_among_unlocked"] == pytest.approx(1 / 2)
    assert d["violations"] == 0


def test_ties_in_support_are_not_comparable_and_ties_in_score_not_reversed():
    d = order_statistics([2, 2, 4], [1, 9, 0.5], R=10)
    # (2,2) incomparable; (2,4): 2 vs 2 tie -> not reversed; (2*9=18, 4*0.5=2) -> reversed
    assert d["comparable_pairs"] == 2
    assert d["reversed_frac"] == pytest.approx(0.5)


def test_violation_is_detected_when_R_is_understated():
    # true weight spread is 20, but R = 2 is claimed: (1, 10) is "locked" yet reversed
    d = order_statistics([1, 10], [20, 1], R=2)
    assert d["violations"] == 1


def test_never_violated_when_R_is_the_true_spread():
    rng = np.random.default_rng(1)
    s = rng.integers(1, 500, size=300).astype(float)
    w = rng.uniform(1.0, 7.0, size=300)
    d = order_statistics(s, w, R=float(w.max() / w.min()))
    assert d["violations"] == 0 and d["comparable_pairs"] > 40000
    assert 0 < d["reversed_frac"] < 1


def test_order_statistics_rejects_bad_R():
    with pytest.raises(ValueError, match="R must be"):
        order_statistics([1, 2], [1, 1], R=0.5)
