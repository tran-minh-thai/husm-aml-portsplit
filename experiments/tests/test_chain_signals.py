"""Hand-computed cases for structuring ratios, dwell time and night share."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import chain_signals as cs  # noqa: E402


def test_band_ratio_counts_half_open_bands():
    r = cs.band_ratio(np.array([8999.99, 9000, 9999.99, 10000, 10999.99, 11000]))
    assert (r["below"], r["above"]) == (2, 2) and r["ratio"] == 1.0
    assert np.isnan(cs.band_ratio(np.array([9500.0]))["ratio"])


def test_structuring_splits_laundering_cash_and_background():
    e = pd.DataFrame({"amount": [9500, 9600, 10500, 9500, 10500, 10600], "laundering": [1, 1, 1, 0, 0, 0]})
    fmt = np.array(["Cash", "ACH", "Cash", "Cash", "Cash", "ACH"])
    st = cs.structuring(e, fmt).set_index(["subset", "group"])
    assert st.loc[("every format", "laundering"), "ratio"] == 2.0
    assert st.loc[("every format", "background"), "ratio"] == 0.5
    assert st.loc[("cash", "laundering"), "ratio"] == 1.0
    assert st.loc[("cash", "background"), "ratio"] == 1.0


def test_dwell_takes_the_next_outgoing_transaction_of_the_same_account():
    # account 1 receives at 10 and 30, sends at 15 and 50 -> gaps 5 and 20, median 12.5
    # account 2 receives at 40, sends only at 20 (earlier) -> no dwell; account 3 sends at the same minute it receives
    src = np.array([0, 0, 1, 1, 2, 3, 0])
    dst = np.array([1, 1, 9, 9, 9, 9, 2])
    time = np.array([10, 30, 15, 50, 20, 7, 40])
    src = np.append(src, 5)
    dst = np.append(dst, 3)
    time = np.append(time, 7)
    d = cs.dwell_minutes(src, dst, time)
    assert d.loc[1] == 12.5
    assert 2 not in d.index
    assert d.loc[3] == 0


def test_night_share_counts_both_ends():
    src, dst = np.array([1, 2]), np.array([2, 3])
    time = np.array([5 * 60 + 59, 6 * 60])          # 05:59 is night, 06:00 is not
    n = cs.night_share(src, dst, time)
    assert n.loc[1] == 1.0 and n.loc[2] == 0.5 and n.loc[3] == 0.0
