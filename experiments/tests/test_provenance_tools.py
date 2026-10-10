"""Tests of machine_info.py, verify_datasets.py and compare_records.py."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "provenance"))
import compare_records as cr  # noqa: E402
import machine_info as mi  # noqa: E402
import run_and_record as rr  # noqa: E402
import verify_datasets as vd  # noqa: E402


# ---------------------------------------------------------------- machine_info

def test_declaration_is_accepted_by_the_wrapper(tmp_path):
    text = mi.render(rr.host_id(), {"cpu": "X", "ram_gb": None}, ["gpu: none"])
    assert mi.declared_ids(text) == [rr.host_id()]
    assert "# ram_gb: unavailable" in text and "# error: gpu: none" in text
    (tmp_path / "results").mkdir()
    mi.write(text, rr.host_id(), path=tmp_path / "results" / rr.MEASUREMENT_MACHINE_FILE)
    assert rr.measurement_tree_refusal(tmp_path / "results" / "q", tmp_path) is None


def test_wrapper_refuses_timing_from_another_machine(tmp_path):
    (tmp_path / "results").mkdir()
    mi.write(mi.render("0123456789ab", {}, []), "0123456789ab",
             path=tmp_path / "results" / rr.MEASUREMENT_MACHINE_FILE)
    assert "not the declared measurement machine" in rr.measurement_tree_refusal(tmp_path / "results" / "q", tmp_path)


def test_first_declaration_is_not_overwritten_by_another_machine(tmp_path):
    p = tmp_path / "MEASUREMENT_MACHINE.txt"
    mi.write(mi.render("aaaaaaaaaaaa", {}, []), "aaaaaaaaaaaa", path=p)
    with pytest.raises(SystemExit, match="refusing to write"):
        mi.write(mi.render("bbbbbbbbbbbb", {}, []), "bbbbbbbbbbbb", path=p)
    mi.write(mi.render("aaaaaaaaaaaa", {"cpu": "new"}, []), "aaaaaaaaaaaa", path=p)   # same machine: refresh
    mi.write(mi.render("bbbbbbbbbbbb", {}, []), "bbbbbbbbbbbb", path=p, replace=True)
    assert mi.declared_ids(p.read_text()) == ["bbbbbbbbbbbb"]


def test_java_properties_are_parsed():
    text = ("Property settings:\n    java.home = /x\n    java.runtime.version = 17.0.12+7\n"
            "    java.vendor = Eclipse Adoptium\n    java.vm.name = OpenJDK 64-Bit Server VM\n")
    p = mi.parse_java_properties(text)
    assert p["java.runtime.version"] == "17.0.12+7" and p["java.vendor"] == "Eclipse Adoptium"


# ------------------------------------------------------------- verify_datasets

def _manifest(tmp_path, files: dict[str, bytes], entries=None) -> pathlib.Path:
    for rel, data in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(data)
    entries = entries if entries is not None else files
    m = tmp_path / "m.sha256"
    m.write_text("# header\n" + "".join(f"{hashlib.sha256(d).hexdigest()}  {len(d)}  {r}\n"
                                        for r, d in entries.items()))
    return m


def test_verify_passes_and_fails(tmp_path, capsys):
    m = _manifest(tmp_path, {"datasets/a.csv": b"1,2\n", "datasets/b.txt": b"x"})
    assert vd.verify(m, tmp_path) == 0
    (tmp_path / "datasets/b.txt").write_bytes(b"y")                     # same size, other bytes
    assert vd.verify(m, tmp_path) == 1
    assert "HASH" in capsys.readouterr().out
    (tmp_path / "datasets/b.txt").unlink()
    assert vd.verify(m, tmp_path) == 1
    assert "MISSING" in capsys.readouterr().out


def test_verify_reports_size_and_unlisted(tmp_path, capsys):
    m = _manifest(tmp_path, {"datasets/a.csv": b"1,2\n", "datasets/extra.csv": b"z"},
                  entries={"datasets/a.csv": b"1,2\n"})
    assert vd.verify(m, tmp_path) == 0
    assert "UNLISTED datasets/extra.csv" in capsys.readouterr().out
    (tmp_path / "datasets/a.csv").write_bytes(b"1,2,3\n")
    assert vd.verify(m, tmp_path) == 1
    assert "SIZE" in capsys.readouterr().out


def test_verify_without_manifest_is_not_a_pass(tmp_path):
    assert vd.verify(tmp_path / "absent.sha256", tmp_path) == 2


# ------------------------------------------------------------- compare_records

def _record(q: pathlib.Path, run_id: str, files: dict[str, bytes], cmd=("py", "x.py", "--d", "A"),
            host="h1") -> None:
    d = q / run_id
    d.mkdir(parents=True)
    (d / "run_meta.json").write_text(json.dumps({"command": list(cmd), "host_id": host, "platform": "p"}))
    for name, data in files.items():
        (d / name).write_bytes(data)


def test_identical_up_to_line_endings_and_masked_paths(tmp_path):
    q = tmp_path / "q"
    _record(q, "20261010T000000", {"t.csv": b"a,b\n1,2\n",
                                   "stdout.txt": b"Done -> <repository root>/r/q/20261010T000000/t.csv\n"},
            cmd=("/home/u/.venvs/v/bin/python", "experiments/scripts/x.py", "--d", "A"))
    _record(q, "20261011T000000", {"t.csv": b"a,b\r\n1,2\r\n",
                                   "stdout.txt": b"Done -> <repository root>\\r\\q\\20261011T000000\\t.csv\r\n"},
            cmd=("C:\\venv\\python.exe", "experiments\\scripts\\x.py", "--d", "A"), host="h2")
    assert cr.compare_question(q) == (1, 0)
    assert cr.main([str(q)]) == 0


def test_one_changed_cell_is_caught(tmp_path, capsys):
    q = tmp_path / "q"
    _record(q, "20261010T000000", {"t.csv": b"a,b\n1,0.30000000000000004\n"})
    _record(q, "20261011T000000", {"t.csv": b"a,b\n1,0.3\n"})
    assert cr.compare_question(q) == (1, 1)
    assert "1 of 4 cells differ" in capsys.readouterr().out


def test_missing_file_and_groups_by_command(tmp_path, capsys):
    q = tmp_path / "q"
    _record(q, "20261010T000000", {"t.csv": b"a\n1\n", "u.tsv": b"x\n"})
    _record(q, "20261011T000000", {"t.csv": b"a\n1\n"})
    _record(q, "20261012T000000", {"t.csv": b"a\n9\n"}, cmd=("py", "x.py", "--d", "B"))
    assert cr.compare_question(q) == (1, 1)
    out = capsys.readouterr().out
    assert "MISSING  u.tsv" in out and "only one record" in out


def test_nothing_compared_is_not_a_pass(tmp_path):
    q = tmp_path / "q"
    _record(q, "20261010T000000", {"t.csv": b"a\n1\n"})
    assert cr.main([str(q)]) == 1


def test_json_lines_compare_values_not_formatting(tmp_path, capsys):
    q = tmp_path / "q"
    _record(q, "20261010T000000", {"p.jsonl": b'{"utility": 0.002, "cmni": 3}\n{"utility": 2.0E23, "cmni": 1}\n'})
    _record(q, "20261011T000000", {"p.jsonl": b'{"utility": 2.0E-3, "cmni": 3}\r\n{"utility": 2e23, "cmni": 1}\r\n'})
    assert cr.compare_question(q) == (1, 0)
    q2 = tmp_path / "q2"
    _record(q2, "20261010T000000", {"p.jsonl": b'{"utility": 0.002, "cmni": 3}\n'})
    _record(q2, "20261011T000000", {"p.jsonl": b'{"utility": 0.0020000000000000005, "cmni": 3}\n'})
    assert cr.compare_question(q2) == (1, 1)
    assert "1 of 1 records differ" in capsys.readouterr().out
