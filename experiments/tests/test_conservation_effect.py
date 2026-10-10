"""Hand-computed case for the conservation-effect table."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import chain_conservation_effect as ce  # noqa: E402


def test_effect_table_by_hand():
    roles = {
        "0.2": pd.DataFrame({"length": [2, 2, 2, 2], "account": [1, 2, 3, 4]}),
        "inf": pd.DataFrame({"length": [2] * 8 + [3], "account": [1, 2, 3, 4, 5, 6, 7, 8, 1]}),
    }
    marked = {1, 2, 3, 9}
    planted = {2: {1, 2, 9}}
    t = ce.effect_table(roles, marked, planted, 3).set_index("length")
    assert t.loc[2, "I_0.2"] == 4 and t.loc[2, "marked_0.2"] == 3 and t.loc[2, "P_0.2"] == pytest.approx(0.75)
    assert t.loc[2, "P_inf"] == pytest.approx(3 / 8)
    assert t.loc[2, "OR"] == pytest.approx((0.75 / 0.25) / ((3 / 8) / (5 / 8)))     # = 5
    assert t.loc[2, "R_0.2"] == pytest.approx(2 / 3) and t.loc[2, "R_inf"] == pytest.approx(2 / 3)
    assert t.loc[3, "I_0.2"] == 0 and math.isnan(t.loc[3, "P_0.2"]) and math.isnan(t.loc[3, "OR"])
    assert math.isnan(t.loc[3, "R_inf"])                                           # no planted chain of length 3
