"""The Large-data fetcher keeps only files whose size and sha256 match the manifest."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provenance"))

import fetch_amlworld_large as fal  # noqa: E402

GOOD = b"Timestamp,From Bank\n1,2\n"
ROW = (hashlib.sha256(GOOD).hexdigest(), len(GOOD), "datasets/amlworld/X_Trans.csv")


def writer(content):
    calls = []

    def get(name, target, auth):
        calls.append(name)
        target.write_bytes(content)
    return get, calls


def test_manifest_lists_the_four_large_files():
    rows = fal.read_manifest(fal.MANIFEST)
    assert [r[2] for r in rows] == ["datasets/amlworld/HI-Large_Patterns.txt", "datasets/amlworld/LI-Large_Patterns.txt",
                                    "datasets/amlworld/HI-Large_Trans.csv", "datasets/amlworld/LI-Large_Trans.csv"]
    assert [r[1] for r in rows] == [13807473, 1934790, 17052760651, 16742513790]


def test_a_matching_download_gets_its_final_name(tmp_path, monkeypatch):
    monkeypatch.setattr(fal, "credentials", lambda: "x")
    get, calls = writer(GOOD)
    assert fal.fetch([ROW], tmp_path, False, get) == []
    assert (tmp_path / ROW[2]).read_bytes() == GOOD and calls == ["X_Trans.csv"]
    get, calls = writer(GOOD)
    assert fal.fetch([ROW], tmp_path, False, get) == [] and calls == []      # present and matching: not fetched again


def test_a_corrupt_download_is_kept_aside_and_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(fal, "credentials", lambda: "x")
    get, _ = writer(GOOD[:-1] + b"3")                                       # same size, other bytes
    problems = fal.fetch([ROW], tmp_path, False, get)
    assert len(problems) == 1 and "sha256" in problems[0]
    assert not (tmp_path / ROW[2]).exists() and (tmp_path / (ROW[2] + ".part")).exists()


def test_check_only_reports_a_missing_file_without_downloading(tmp_path):
    get, calls = writer(GOOD)
    problems = fal.fetch([ROW], tmp_path, True, get)
    assert problems == ["X_Trans.csv: missing"] and calls == []
