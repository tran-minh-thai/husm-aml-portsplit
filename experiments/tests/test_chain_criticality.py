"""mu's continuation counts against a direct double loop over the definition."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chain_criticality as cr  # noqa: E402


def direct(src, dst, amt, theta):
    out = np.zeros(len(src), dtype=np.int64)
    for i in range(len(src)):
        for j in range(len(src)):
            if src[j] == dst[i] and dst[j] != src[i] and (1 - theta) * amt[i] <= amt[j] <= (1 + theta) * amt[i]:
                out[i] += 1
    return out


def test_counts_equal_the_definition_including_boundaries_and_back_and_forth():
    rng = np.random.default_rng(3)
    checked = boundary = 0
    for _ in range(300):
        n, m = rng.integers(2, 7), rng.integers(1, 30)
        src = rng.integers(0, n, m)
        dst = (src + rng.integers(1, n, m)) % n
        amt = rng.choice([48.0, 60.0, 64.0, 75.0, 80.0, 100.0, 120.0, 125.0, 150.0], m)
        for theta in (0.0, 0.2, 0.25, 0.5):
            want = direct(src, dst, amt, theta)
            assert np.array_equal(cr.continuation_counts(src, dst, amt, theta), want)
            checked += 1
            boundary += int(theta == 0.25)
    assert checked == 1200 and boundary == 300


def test_back_and_forth_is_excluded():
    src, dst, amt = np.array([0, 1, 1]), np.array([1, 0, 2]), np.array([10.0, 10.0, 10.0])
    # from 0 -> 1, the continuation 1 -> 0 goes back and is not counted; 1 -> 2 is
    assert cr.continuation_counts(src, dst, amt, 0.0).tolist() == [1, 0, 0]
