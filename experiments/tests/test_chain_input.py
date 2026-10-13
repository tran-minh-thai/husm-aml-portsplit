"""Hand-built cases for the chain-input exporter."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inchusm_aml.data import chain_input as ci  # noqa: E402

HEADER = ("step,type,amount,nameOrig,bankOrig,daysInBankOrig,phoneChangesOrig,oldbalanceOrig,newbalanceOrig,"
          "nameDest,bankDest,daysInBankDest,phoneChangesDest,oldbalanceDest,newbalanceDest,isSAR,alertID,modelType")


def gentex(tmp_path: Path, rows: list[tuple]) -> Path:
    lines = [HEADER]
    for orig, bank_o, dest, bank_d, amount, sar in rows:
        lines.append(f"1,TRANSFER,{amount},{orig},{bank_o},0,0,0,0,{dest},{bank_d},0,0,0,0,{sar},{0 if sar else -1},"
                     f"{3 if sar else 0}")
    p = tmp_path / "tx_log.csv"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_amlgentex_drops_special_and_self_and_takes_the_most_frequent_bank(tmp_path):
    p = gentex(tmp_path, [
        (-2, "source", 1, "b", 50.0, 0),     # income source: dropped
        (1, "b", 2, "a", 10.0, 0),
        (2, "a", 3, "a", 9.0, 1),            # laundering: accounts 2 and 3
        (2, "c", 1, "b", 8.0, 0),            # account 2 seen in bank a twice, c once -> a
        (3, "a", 3, "a", 7.0, 0),            # self-transaction: dropped
        (4, "z", 1, "b", 6.0, 0),
        (1, "y", 4, "x", 5.0, 0),            # account 4: banks z and x once each -> tie -> x
    ])
    acc, edges, laundering, info = ci.amlgentex(p, "A")
    assert info["special_rows_dropped"] == 1 and info["self_transactions_dropped"] == 1
    assert list(laundering) == [2, 3]
    names = info["account_labels"]
    bank = {int(r.id): names[str(r.label)] for r in acc.itertuples()}
    assert bank == {1: "b", 2: "a", 3: "a", 4: "x"}
    assert info["accounts_with_more_than_one_bank"] == 3    # accounts 1 (b, y), 2 (a, c), 4 (z, x)
    assert len(edges) == 5 and set(edges["label"]) == {0}
    acc_s, _, _, _ = ci.amlgentex(p, "S")
    assert set(acc_s["label"]) == {0}


def test_write_tg_round_trip_keeps_amounts_exactly(tmp_path):
    acc = pd.DataFrame({"id": [7, 9], "label": [1, 2]})
    amounts = np.array([0.1, 1e-7 + 3.0, 123456789.123456789, 2.0 / 3.0])
    edges = pd.DataFrame({"src": [7, 9, 7, 9], "dst": [9, 7, 9, 7], "label": [0, 1, 0, 1], "amount": amounts})
    out = tmp_path / "g.tg"
    ci.write_tg(acc, edges, out)
    lines = out.read_text().splitlines()
    assert lines[1:3] == ["v 7 1", "v 9 2"]
    back = np.array([float(l.split()[4]) for l in lines[3:]])
    assert np.array_equal(back, amounts)


def test_write_tg_refuses_non_positive_amounts(tmp_path):
    acc = pd.DataFrame({"id": [1, 2], "label": [0, 0]})
    edges = pd.DataFrame({"src": [1], "dst": [2], "label": [0], "amount": [0.0]})
    with pytest.raises(ValueError):
        ci.write_tg(acc, edges, tmp_path / "g.tg")


def test_files_end_lines_with_newline_only_even_where_the_system_uses_crlf(tmp_path, monkeypatch):
    import os
    import pandas as pd
    from inchusm_aml.data import chain_input as ci
    monkeypatch.setattr(os, "linesep", "\r\n")                     # what pandas sees on Windows
    accounts = pd.DataFrame({"id": [0, 1], "label": [0, 0]})
    edges = pd.DataFrame({"src": [0], "dst": [1], "label": [0], "amount": [100.5]})
    ci.write_tg(accounts, edges, tmp_path / "g.tg")
    ci.write_ids([3, 17], tmp_path / "m.txt")
    assert b"\r" not in (tmp_path / "g.tg").read_bytes()
    assert (tmp_path / "g.tg").read_bytes().endswith(b"e 0 1 0 100.5\n")
    assert (tmp_path / "m.txt").read_bytes() == b"3\n17\n"
