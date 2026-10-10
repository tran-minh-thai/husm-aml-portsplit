"""Hand-built cases for planted chains, background chains and the amount rules."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inchusm_aml.eval import planted_chains as pc  # noqa: E402


def ptx(rows):
    return pd.DataFrame(rows, columns=["attempt", "type", "src", "dst", "time", "amount"])


def as_set(df):
    return {(r.kind, r.length, r.accounts, r.amounts) for r in df.itertuples()}


def test_cycle_gives_open_chains_that_avoid_the_origin_and_one_closed_chain():
    p = ptx([(0, "CYCLE", "a", "b", 1, 100.0), (0, "CYCLE", "b", "c", 2, 90.0), (0, "CYCLE", "c", "a", 3, 81.0)])
    got = as_set(pc.planted_chains(p, 5))
    assert got == {("open", 2, ("a", "b", "c"), (100.0, 90.0)),
                   ("open", 2, ("b", "c", "a"), (90.0, 81.0)),
                   ("closed", 3, ("a", "b", "c"), (100.0, 90.0, 81.0))}


def test_no_chain_through_an_account_that_splits_or_merges():
    p = ptx([(1, "SCATTER-GATHER", "s", "v", 1, 10.0), (1, "SCATTER-GATHER", "v", "w1", 2, 5.0),
             (1, "SCATTER-GATHER", "v", "w2", 3, 5.0)])
    assert pc.planted_chains(p, 5).empty


def test_stack_chain_lengths_are_capped():
    p = ptx([(2, "STACK", "a", "b", 1, 10.0), (2, "STACK", "b", "c", 2, 10.0), (2, "STACK", "c", "d", 3, 10.0),
             (2, "STACK", "d", "e", 4, 10.0)])
    got = pc.planted_chains(p, 3)
    assert sorted(got["length"].tolist()) == [2, 2, 2, 3, 3]
    assert set(got["kind"]) == {"open"}


def test_rules_at_their_boundaries():
    assert pc.pairwise([100, 125, 156.25], 0.25)          # 125 = 1.25 * 100, 156.25 = 1.25 * 125
    assert not pc.pairwise([100, 125.0001], 0.25)
    assert pc.anchored([100, 75, 125], 0.25)
    assert not pc.anchored([100, 125, 156.25], 0.25)       # drift: fine pairwise, not anchored
    assert pc.pairwise([100, 90, 81, 72.9], 0.2) and not pc.anchored([100, 90, 81, 72.9], 0.2)


def test_background_chains_use_background_transactions_and_distinct_accounts():
    # 3 -> 2 makes walks 2 -> 3 -> 2 and 3 -> 2 -> 3 possible; both repeat an account and are discarded
    edges = pd.DataFrame({"src": [1, 2, 3, 2, 9, 4, 3], "dst": [2, 3, 1, 4, 2, 5, 2],
                          "amount": [1.0, 2.0, 3.0, 4.0, 99.0, 6.0, 7.0], "laundering": [0, 0, 0, 0, 1, 0, 0]})
    chains, discarded = pc.background_chains(edges, 2, 300, seed=3)
    allowed = {(1.0, 2.0), (1.0, 4.0), (2.0, 3.0), (3.0, 1.0), (4.0, 6.0), (7.0, 4.0)}
    assert len(chains) == 300 and set(chains) <= allowed and set(chains) == allowed
    assert discarded > 0                                   # walks starting 4 -> 5 have nowhere to go
