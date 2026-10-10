"""Hand-computed cases for the label-choice measurements (entropy, eta squared, leakage)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import label_roles as lr  # noqa: E402


def test_entropy_of_two_equal_values_is_ln2():
    assert lr.entropy(pd.Series([5, 5]).to_numpy()) == pytest.approx(math.log(2))


def test_eta2_is_one_when_groups_fix_the_amount_and_zero_when_they_do_not():
    amounts = pd.Series([1.0, 1.0, math.e, math.e])
    assert lr.eta2(amounts, pd.Series(["a", "a", "b", "b"])) == pytest.approx(1.0)
    assert lr.eta2(amounts, pd.Series(["a", "b", "a", "b"])) == pytest.approx(0.0)


def test_eta2_by_hand():
    # ln amounts 0, 2, 4, 6 in groups {0, 2} and {4, 6}: total SS 20, between SS 2*(1-3)^2 + 2*(5-3)^2 = 16
    amounts = pd.Series([math.exp(v) for v in (0, 2, 4, 6)])
    assert lr.eta2(amounts, pd.Series(["a", "a", "b", "b"])) == pytest.approx(16 / 20)


def test_leakage_is_full_when_the_label_is_the_flag_and_none_when_independent():
    flag = pd.Series([0, 0, 1, 1] * 50)
    nmi, lo, hi = lr.leakage(pd.Series(["x", "x", "y", "y"] * 50), flag)
    assert nmi == pytest.approx(1.0)
    assert (lo, hi) == pytest.approx((0.0, 2.0))
    nmi, lo, hi = lr.leakage(pd.Series(["x", "y", "x", "y"] * 50), flag)
    assert nmi == pytest.approx(0.0)
    assert (lo, hi) == pytest.approx((1.0, 1.0))
