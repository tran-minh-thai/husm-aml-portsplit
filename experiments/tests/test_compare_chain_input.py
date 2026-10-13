"""The chain-input comparison reports line counts and the first differing lines."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))

import compare_chain_input as cci  # noqa: E402


def test_first_differences_reports_changed_and_missing_lines(tmp_path):
    a, b = tmp_path / "a.tg", tmp_path / "b.tg"
    a.write_bytes(b"v 0 0\nv 1 0\ne 0 1 0 100\ne 1 0 0 90\n")
    b.write_bytes(b"v 0 0\nv 1 0\ne 0 1 0 100.00000000000001\n")
    na, nb, diffs = cci.first_differences(a, b, 5)
    assert (na, nb) == (4, 3)
    assert diffs == [(3, repr("e 0 1 0 100\n"), repr("e 0 1 0 100.00000000000001\n")),
                     (4, repr("e 1 0 0 90\n"), repr("<end of file>"))]
    assert cci.first_differences(a, a, 5)[2] == []
    assert cci.first_differences(b, a, 5)[:2] == (3, 4)                  # the first file is the shorter one
    assert len(cci.first_differences(a, b, 1)[2]) == 1
