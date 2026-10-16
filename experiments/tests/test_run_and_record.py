"""Tests of experiments/provenance/run_and_record.py: run-id reservation, the code hash, the
engine-jar staleness refusal and the data checksums."""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys
import threading
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "provenance"))
import run_and_record as rr  # noqa: E402


def test_an_id_taken_under_any_parent_is_skipped_and_nothing_is_left_behind(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    (b / "20261001T000000").mkdir(parents=True)
    made = rr.new_record_dirs([a, b], "20261001T000000")
    assert [p.name for p in made] == ["20261001T000000-2", "20261001T000000-2"]
    assert not (a / "20261001T000000").exists(), "the id taken under b was created under a"


def test_runs_started_in_the_same_second_get_distinct_ids(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ids, errors = [], []
    start = threading.Barrier(12)

    def reserve():
        try:
            start.wait()
            made = rr.new_record_dirs([a, b], "20261001T000000")
            ids.append((made[0].name, made[1].name))
        except Exception as e:  # noqa: BLE001 - every failure is the finding
            errors.append(repr(e))

    threads = [threading.Thread(target=reserve) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len({x for x, _ in ids}) == 12
    assert all(x == y for x, y in ids)
    assert len(list(a.iterdir())) == 12 and len(list(b.iterdir())) == 12


def _fake_repo(root: pathlib.Path) -> None:
    (root / "experiments" / "src").mkdir(parents=True)
    (root / "experiments" / "src" / "m.py").write_bytes(b"x = 1\ny = 2\n")
    java = root / rr.JAVA_SOURCES / "p"
    java.mkdir(parents=True)
    (java / "A.java").write_bytes(b"class A {}\n")
    (root / "inchusm-miner" / "pom.xml").write_bytes(b"<project/>\n")


def test_code_hash_ignores_line_endings_but_not_content(tmp_path):
    _fake_repo(tmp_path)
    h_lf, n = rr.code_hash(tmp_path)
    assert n == 3
    (tmp_path / "experiments" / "src" / "m.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    assert rr.code_hash(tmp_path)[0] == h_lf, "CRLF checkout changed the hash"
    (tmp_path / "experiments" / "src" / "m.py").write_bytes(b"x = 1\ny = 3\n")
    assert rr.code_hash(tmp_path)[0] != h_lf, "a content change did not change the hash"


def test_code_hash_sees_a_new_file_and_a_rename(tmp_path):
    _fake_repo(tmp_path)
    h0, _ = rr.code_hash(tmp_path)
    (tmp_path / "experiments" / "src" / "m.py").rename(tmp_path / "experiments" / "src" / "n.py")
    h1, _ = rr.code_hash(tmp_path)
    assert h1 != h0
    (tmp_path / "experiments" / "src" / "z.py").write_bytes(b"")
    assert rr.code_hash(tmp_path)[0] != h1


def test_jar_older_than_a_source_is_reported_stale(tmp_path):
    _fake_repo(tmp_path)
    jar = tmp_path / rr.JAR
    jar.parent.mkdir(parents=True)
    jar.write_bytes(b"jar")
    old = time.time() - 100
    os.utime(jar, (old, old))
    sha, stale = rr.jar_state(tmp_path)
    assert sha == hashlib.sha256(b"jar").hexdigest()
    assert stale == [(rr.JAVA_SOURCES / "p" / "A.java").as_posix()]
    os.utime(jar, None)                     # rebuilt now: newer than every source
    assert rr.jar_state(tmp_path)[1] == []


def test_absent_jar_is_recorded_as_absent(tmp_path):
    _fake_repo(tmp_path)
    assert rr.jar_state(tmp_path) == (None, [])


def test_data_record_hashes_and_refuses_missing(tmp_path):
    f = tmp_path / "d.csv"
    f.write_bytes(b"a,b\n1,2\n")
    rec = rr.data_record([str(f)], tmp_path)
    assert rec == [{"path": "d.csv", "bytes": 8, "sha256": hashlib.sha256(b"a,b\n1,2\n").hexdigest()}]
    with pytest.raises(FileNotFoundError, match="no such file"):
        rr.data_record([str(tmp_path / "missing.csv")], tmp_path)


def test_keep_awake_requests_and_releases_system_wakefulness():
    calls = []

    class Kernel:
        def SetThreadExecutionState(self, flags):
            calls.append(flags)

    assert rr.keep_awake(True, Kernel()) == rr.ES_CONTINUOUS | rr.ES_SYSTEM_REQUIRED
    assert rr.keep_awake(False, Kernel()) == rr.ES_CONTINUOUS
    assert calls == [0x80000001, 0x80000000]
    if os.name != "nt":
        assert rr.keep_awake(True) is None


def test_timing_kind_names_measurement_and_probe_trees(tmp_path):
    root = tmp_path
    assert rr.timing_kind(root / "results" / "q", root) == "measurement"
    assert rr.timing_kind(root / "results-probe-timing" / "q", root).startswith("probe")
    assert rr.timing_kind(root / "elsewhere" / "q", root) == "outside the result trees"
    assert rr.timing_kind(tmp_path.parent / "q", root) == "outside the result trees"


def test_probe_timing_pairs_only_with_a_probe_record(tmp_path):
    root = tmp_path
    probe_timing = root / "results-probe-timing" / "q"
    assert rr.probe_pairing_refusal(root / "results-probe" / "q", probe_timing, root) is None
    assert "pairs only" in rr.probe_pairing_refusal(root / "results-invariant" / "q", probe_timing, root)
    assert "pairs only" in rr.probe_pairing_refusal(tmp_path.parent / "q", probe_timing, root)
    # a measurement pairing is not this check's business
    assert rr.probe_pairing_refusal(root / "results-invariant" / "q", root / "results" / "q", root) is None


def test_untracked_probe_timing_records_do_not_make_the_tree_dirty():
    assert "results-probe-timing" in rr.RESULT_TREES
