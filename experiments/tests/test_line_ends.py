"""Files whose bytes matter are written with "\n" line ends, also where the system writes text with "\r\n".

A file matters when its sha256 is recorded (an input named with --data, the chain graph whose hash the export
record keeps) or when it is compared byte for byte. The fixture below makes this process write text the way
Windows does. Setting os.linesep alone is not enough: pandas reads it, but open() and Path.write_text translate
"\n" in the C io layer, which uses "\r\n" only when built for Windows, so on macOS and Linux they would keep
writing "\n" and a test relying on os.linesep alone could not fail for them. The first test checks that the
fixture really produces "\r\n" for every writing path used here.
"""
from __future__ import annotations

import builtins
import io
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

from inchusm_aml import miner_bridge  # noqa: E402
from inchusm_aml.data import chain_input  # noqa: E402

import e5a_ellipticpp_utility as e5a  # noqa: E402


@pytest.fixture
def windows_text(monkeypatch):
    """os.linesep is "\r\n", and a text-mode open() for writing with newline=None translates "\n" to "\r\n"."""
    real_open = io.open

    def open_as_on_windows(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in mode and newline is None and any(c in mode for c in "wax+"):
            newline = "\r\n"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(os, "linesep", "\r\n")
    monkeypatch.setattr(io, "open", open_as_on_windows)
    monkeypatch.setattr(builtins, "open", open_as_on_windows)


def test_the_fixture_writes_crlf_through_every_path_the_writers_use(tmp_path, windows_text):
    (tmp_path / "a.txt").write_text("x\ny\n", encoding="ascii")
    with open(tmp_path / "b.txt", "w", encoding="utf-8") as f:
        f.write("x\ny\n")
    pd.DataFrame({"a": [1, 2]}).to_csv(tmp_path / "c.csv", index=False)
    assert (tmp_path / "a.txt").read_bytes() == b"x\r\ny\r\n"
    assert (tmp_path / "b.txt").read_bytes() == b"x\r\ny\r\n"
    assert (tmp_path / "c.csv").read_bytes() == b"a\r\n1\r\n2\r\n"
    # and an explicit choice still wins, so the fixture does not hide a correct fix
    (tmp_path / "d.txt").write_text("x\n", encoding="ascii", newline="\n")
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "e.csv", index=False, lineterminator="\n")
    assert (tmp_path / "d.txt").read_bytes() == b"x\n"
    assert (tmp_path / "e.csv").read_bytes() == b"a\n1\n"


def test_export_lg_writes_lf(tmp_path, windows_text):
    labels = pd.Series([1, 2, 1], index=["a", "b", "c"])
    edges = pd.DataFrame({"src": ["a", "b"], "dst": ["b", "c"], "elabel": [0, 3], "weight": [10.5, 2.0]})
    out = tmp_path / "g.lg"
    miner_bridge.export_lg(labels, edges, out)
    data = out.read_bytes()
    assert b"\r" not in data
    assert data == b"v 0 1\nv 1 2\nv 2 1\ne 0 1 0 10.5\ne 1 2 3 2\n"


def test_export_snapshot_stream_writes_lf(tmp_path, windows_text):
    labels = pd.Series([0, 1], index=[5, 6])
    edges = pd.DataFrame({"src": [5], "dst": [6]})
    miner_bridge.export_snapshot_stream([(labels, edges), (labels, edges)], tmp_path / "snaps")
    files = sorted((tmp_path / "snaps").glob("snap_*.lg"))
    assert [f.name for f in files] == ["snap_01.lg", "snap_02.lg"]
    for f in files:
        assert f.read_bytes() == b"v 0 0\nv 1 1\ne 0 1 0\n"


def test_export_utilities_writes_lf(tmp_path, windows_text):
    out = tmp_path / "u.util.tsv"
    miner_bridge.export_utilities({2: 0.25, 0: 1.0, 1: 12.5}, out)
    data = out.read_bytes()
    assert b"\r" not in data
    assert data == b"0 1.000000\n1 12.500000\n2 0.250000\n"


def test_e5a_pattern_table_is_written_with_lf(tmp_path, windows_text):
    hus = pd.DataFrame({"code": ["v0-v1", "v1-v2"], "utility": [3.5, 1.25], "cmni": [2, 1]})
    out = tmp_path / "e5a_ellipticpp_hus.csv"
    e5a.save_hus(hus, out)
    data = out.read_bytes()
    assert b"\r" not in data
    assert data == b"code,utility,cmni\nv0-v1,3.5,2\nv1-v2,1.25,1\n"


def test_chain_graph_is_written_with_lf_through_the_file_handle_too(tmp_path, windows_text):
    """write_tg passes lineterminator="\n" to pandas, but pandas writes into a text-mode handle, which on Windows
    would translate that "\n" again unless the handle is opened with newline="\n"."""
    accounts = pd.DataFrame({"id": [0, 1], "label": [0, 0]})
    edges = pd.DataFrame({"src": [0], "dst": [1], "label": [0], "amount": [100.5]})
    out = tmp_path / "g.tg"
    chain_input.write_tg(accounts, edges, out)
    data = out.read_bytes()
    assert b"\r" not in data
    assert data.endswith(b"\nv 0 0\nv 1 0\ne 0 1 0 100.5\n")
    chain_input.write_ids([3, 17], tmp_path / "m.txt")
    assert (tmp_path / "m.txt").read_bytes() == b"3\n17\n"
