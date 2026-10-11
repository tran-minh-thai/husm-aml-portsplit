"""Hand-computed cases for the closed-chain returned sets and their scores."""
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import chain_cycle_effect as cce  # noqa: E402


def test_closed_sets_take_origin_and_intermediate_and_ignore_open(tmp_path):
    p = tmp_path / "roles.tsv"
    p.write_text("kind\tlength\trole\taccount\n"
                 "closed\t3\torigin\t1\nclosed\t3\tintermediate\t2\nclosed\t3\tintermediate\t3\n"
                 "open\t3\tintermediate\t9\nclosed\t4\tintermediate\t5\n")
    assert cce.closed_sets(p) == {3: {1, 2, 3}, 4: {5}}


def test_planted_cycles_use_every_account_of_closed_chains_only():
    planted = pd.DataFrame({"kind": ["closed", "open", "closed"], "length": [3, 3, 4],
                            "accounts": [(1, 2, 3), (7, 8, 9, 10), (4, 5, 6, 7)]})
    assert cce.planted_cycle_accounts(planted) == {3: {1, 2, 3}, 4: {4, 5, 6, 7}}


def test_effect_scores():
    sets = {"0.2": {3: {1, 2, 3, 4}}, "inf": {3: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}}
    marked = {1, 2, 3}
    planted = {3: {1, 2, 3, 11}}
    t = cce.effect(sets, marked, planted).set_index("length")
    assert t.loc[3, "P_0.2"] == 0.75 and t.loc[3, "P_inf"] == 0.3
    assert t.loc[3, "R_0.2"] == 0.75 and t.loc[3, "R_inf"] == 0.75
    assert math.isclose(t.loc[3, "OR"], (0.75 / 0.25) / (0.3 / 0.7))
    assert t.loc[4, "I_0.2"] == 0 and math.isnan(t.loc[4, "OR"])
