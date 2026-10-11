"""Hand-computed cases for pattern purity, the paired comparison and the top-ten expectation."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import chain_pattern_purity as cpp  # noqa: E402

HEADER = "kind\tlength\taccount_labels\ttransaction_labels\tsupport\timage_counts\tmarked_counts\n"


def write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text(HEADER + "".join("\t".join(map(str, r)) + "\n" for r in rows))
    return p


def test_purity_uses_only_intermediate_positions(tmp_path):
    p = write(tmp_path, "a.tsv", [("open", 3, "0,0,0,0", "0,0,0", 10, "50,20,10,40", "50,10,10,40"),
                                  ("open", 1, "0,0", "0", 99, "99,99", "0,0"),
                                  ("closed", 3, "0,0,0", "0,0,0", 10, "10,10,10", "0,0,0")])
    t = cpp.purity_table(p)
    assert list(t["length"]) == [3]                         # length 1 has no intermediate account; closed left out
    assert t["purity"].iloc[0] == pytest.approx(20 / 30)    # positions 1 and 2 only: (10 + 10) / (20 + 10)


def test_missing_marked_counts_are_refused(tmp_path):
    p = write(tmp_path, "b.tsv", [("open", 2, "0,0,0", "0,0", 10, "10,10,10", "")])
    with pytest.raises(SystemExit):
        cpp.purity_table(p)


def test_paired_difference_is_taken_on_the_same_pattern():
    a = pd.DataFrame({"length": [4, 4], "account_labels": ["x", "y"], "transaction_labels": ["0", "0"],
                      "support": [10, 10], "image_mid": [10, 10], "marked_mid": [9, 1], "purity": [0.9, 0.1]})
    b = a.assign(purity=[0.2, 0.1], marked_mid=[2, 1])
    b = pd.concat([b, b.iloc[:1].assign(account_labels="z", purity=0.0)])
    j, missing = cpp.paired(a, b.iloc[::-1].reset_index(drop=True), 4)
    assert missing == 0
    assert dict(zip(j["account_labels"], j["difference"].round(6))) == {"x": 0.7, "y": 0.0}
    _, missing = cpp.paired(a, b[b["account_labels"] != "y"], 4)
    assert missing == 1


def test_top_ranks_by_length_then_support_and_takes_the_expectation_over_ties():
    t = pd.DataFrame({"length": [5, 4, 4, 4, 3], "support": [10, 50, 20, 20, 99],
                      "image_mid": [10, 10, 10, 30, 10], "marked_mid": [10, 0, 10, 0, 10],
                      "purity": [1.0, 0.0, 1.0, 0.0, 1.0]})
    r = cpp.top(t, 3)          # length 5 (1.0), then support 50 (0.0), then one of two tied at 20 (1.0 or 0.0)
    assert r["patterns"] == 3
    assert r["mean_purity"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert r["image_mid"] == pytest.approx(10 + 10 + 0.5 * 40)
    assert r["weighted_purity"] == pytest.approx((10 + 0 + 0.5 * 10) / 40)
    assert cpp.top(t, 99)["patterns"] == 5
