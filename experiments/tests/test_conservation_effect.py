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


def test_shuffled_copy_permutes_amounts_and_keeps_everything_else(tmp_path):
    import importlib
    ce = importlib.import_module("chain_conservation_effect")
    lines = ["# directed transaction graph: v ID LABEL / e SOURCE TARGET LABEL AMOUNT"]
    lines += [f"v {i} {i % 2}" for i in range(30)]
    amounts = [10.5 + 3 * i for i in range(40)] + [910.28999999999996]
    lines += [f"e {i % 30} {(i * 7 + 1) % 30} {i % 3} {a!r}" for i, a in enumerate(amounts)]
    src = tmp_path / "in" / "T_S.tg"
    src.parent.mkdir()
    src.write_text("\n".join(lines) + "\n")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = ce.shuffled_copy(src, 7, out_dir)
    before, after = src.read_text().splitlines(), out.read_text().splitlines()
    assert len(before) == len(after)
    assert [l for l in before if not l.startswith("e ")] == [l for l in after if not l.startswith("e ")]
    eb = [l.split(" ") for l in before if l.startswith("e ")]
    ea = [l.split(" ") for l in after if l.startswith("e ")]
    assert [x[:4] for x in eb] == [x[:4] for x in ea]                          # sources, targets, labels kept
    assert sorted(float(x[4]) for x in eb) == sorted(float(x[4]) for x in ea)  # the same amounts
    moved = sum(float(x[4]) != float(y[4]) for x, y in zip(eb, ea))
    assert moved > len(eb) // 2                                              # and they did move
    again = tmp_path / "again"
    again.mkdir()
    assert ce.shuffled_copy(src, 7, again).read_text() == out.read_text()    # same seed, same file
