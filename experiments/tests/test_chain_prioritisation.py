"""Hand-computed cases for precision at k with ties and for drifting chains."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chain_prioritisation as cp  # noqa: E402


def test_precision_at_k_takes_the_expectation_over_ties():
    score = {"a": 2, "b": 2, "c": 1, "d": 1, "e": 0}
    pos = {"a", "c"}
    assert cp.precision_at_k(score, pos, 1) == pytest.approx(0.5)           # one of {a, b}, a is positive
    assert cp.precision_at_k(score, pos, 2) == pytest.approx(0.5)           # a and b
    assert cp.precision_at_k(score, pos, 3) == pytest.approx((1 + 0.5) / 3)  # a, b, and one of {c, d}
    assert cp.precision_at_k(score, pos, 10) == pytest.approx(2 / 5)        # k larger than the universe


def test_drift_is_valid_pairwise_and_invalid_anchored():
    # (100, 90, 75): steps 0.90 and 0.83 are within 0.2 (pairwise valid), but 75 < 0.8 * 100 (anchored invalid) -> drift
    # (100, 100): valid under both rules -> not drift
    # (100, 85, 115): 115 / 85 = 1.35 breaks the pairwise rule -> not drift
    # the closed chain is ignored
    planted = pd.DataFrame({
        "kind": ["open", "open", "open", "closed"],
        "length": [3, 2, 3, 3],
        "accounts": [("a", "b", "c", "d"), ("p", "q", "r"), ("x", "y", "z", "w"), ("k", "l", "n")],
        "amounts": [(100.0, 90.0, 75.0), (100.0, 100.0), (100.0, 85.0, 115.0), (100.0, 90.0, 75.0)],
    })
    assert cp.drift_intermediates(planted) == {3: {"b", "c"}}


def test_max_length_score_takes_the_longest_set():
    s = cp.max_length_score({2: {"a", "b"}, 3: {"b"}, 5: {"c"}}, ["a", "b", "c", "d"])
    assert s == {"a": 2, "b": 3, "c": 5, "d": 0}
