"""The three gate measurements on hand-built data with hand-computed answers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inchusm_aml.data.amlworld import load_pattern_transactions  # noqa: E402
from inchusm_aml.eval import flow_gates as fg  # noqa: E402


def ptx_frame(rows):
    """rows: (attempt, type, src, dst, time, amount)."""
    return pd.DataFrame(rows, columns=["attempt", "type", "src", "dst", "time", "amount"])


def edges_frame(rows):
    """rows: (src, dst, amount, laundering)."""
    return pd.DataFrame(rows, columns=["src", "dst", "amount", "laundering"])


# ------------------------------------------------------------------ amount gate

def test_amount_gate_hand_case():
    # background amounts 1, 2, 3, 4 (plus a self-transfer of 100 that must be ignored)
    edges = edges_frame([(1, 2, 1, 0), (2, 3, 2, 0), (3, 4, 3, 0), (4, 5, 4, 0), (6, 6, 100, 0),
                         (10, 11, 5, 1), (12, 11, 1, 1)])
    ptx = ptx_frame([(0, "FAN-IN", 10, 11, 0, 5.0), (0, "FAN-IN", 12, 11, 1, 1.0),
                     (1, "STACK", 20, 21, 0, 3.0), (1, "STACK", 21, 22, 1, 4.0)])
    g = fg.amount_gate(edges, ptx).set_index("type")
    assert g.attrs["n_background"] == 4
    # FAN-IN: P(5 > bg) = 1, P(1 > bg) = 0 + 1/2 * 1/4 = 0.125 -> mean 0.5625; bottleneck 1 -> 0.125
    assert g.loc["FAN-IN", "auc_vs_background"] == pytest.approx(0.5625)
    assert g.loc["FAN-IN", "bottleneck_pct_median"] == pytest.approx(0.125)
    # STACK: 3 -> (2 + 0.5)/4 = 0.625, 4 -> (3 + 0.5)/4 = 0.875 -> 0.75; bottleneck 3 -> 0.625
    assert g.loc["STACK", "auc_vs_background"] == pytest.approx(0.75)
    assert g.loc["STACK", "bottleneck_pct_median"] == pytest.approx(0.625)
    # all labelled laundering in the edges: 5 -> 1.0, 1 -> 0.125
    assert g.loc[fg.ALL_LAUNDERING, "auc_vs_background"] == pytest.approx(0.5625)
    assert g.loc["STACK", "n_attempts"] == 1 and g.loc["FAN-IN", "n_transactions"] == 2


def test_percentile_rejects_empty_reference():
    with pytest.raises(ValueError, match="empty"):
        fg.percentile_in([1.0], np.array([]))


# ------------------------------------------------------------------- order gate

ORDER_PTX = ptx_frame([
    # attempt 0, FAN-IN: no pass-through account
    (0, "FAN-IN", 1, 3, 10, 1.0), (0, "FAN-IN", 2, 3, 11, 1.0),
    # attempt 1, STACK a->b (t5), b->c (t7), c->d (t7): b in 5 out 7; c in 7 out 7 (tie)
    (1, "STACK", 10, 11, 5, 1.0), (1, "STACK", 11, 12, 7, 1.0), (1, "STACK", 12, 13, 7, 1.0),
    # attempt 2, CYCLE o->p (t1), p->o (t3): o sends at 1, receives at 3
    (2, "CYCLE", 20, 21, 1, 1.0), (2, "CYCLE", 21, 20, 3, 1.0),
    # attempt 3, CYCLE o->p (1), p->q (2), q->o (3), o->r (5): o receives 3, sends 1 and 5
    (3, "CYCLE", 30, 31, 1, 1.0), (3, "CYCLE", 31, 32, 2, 1.0), (3, "CYCLE", 32, 30, 3, 1.0),
    (3, "CYCLE", 30, 33, 5, 1.0),
    # a self-transfer inside an attempt is ignored (it would make 13 a pass-through account)
    (1, "STACK", 13, 13, 0, 1.0),
])


def test_passthrough_accounts_hand_case():
    acc = fg.passthrough_accounts(ORDER_PTX).set_index(["attempt", "account"])
    assert set(acc.index) == {(1, 11), (1, 12), (2, 20), (2, 21), (3, 30), (3, 31), (3, 32)}
    b, c = acc.loc[(1, 11)], acc.loc[(1, 12)]
    assert b.all_in_before_out_le and b.all_in_before_out_lt and b.gap_min == 2
    assert c.all_in_before_out_le and not c.all_in_before_out_lt and c.tie and c.gap_min == 0
    o2 = acc.loc[(2, 20)]
    assert not o2.all_in_before_out_le and not o2.some_in_before_out_le and o2.gap_min == -2
    o3 = acc.loc[(3, 30)]
    assert not o3.all_in_before_out_le and o3.some_in_before_out_le and o3.span_min == 2


def test_order_gate_hand_case():
    g = fg.order_gate(ORDER_PTX).set_index("type")
    assert g.loc["FAN-IN", "n_attempts"] == 1 and g.loc["FAN-IN", "n_with_passthrough"] == 0
    assert np.isnan(g.loc["FAN-IN", "frac_attempts_all_in_before_out_le"])
    s = g.loc["STACK"]
    assert s.n_with_passthrough == 1 and s.n_passthrough_accounts == 2
    assert s.frac_attempts_all_in_before_out_le == 1.0 and s.frac_attempts_all_in_before_out_lt == 0.0
    assert s.frac_attempts_some_in_before_out_le == 1.0 and s.frac_accounts_tie == 0.5
    c = g.loc["CYCLE"]
    assert c.n_attempts == 2 and c.n_with_passthrough == 2
    assert c.frac_attempts_all_in_before_out_le == 0.0
    assert c.frac_attempts_some_in_before_out_le == 0.5


# --------------------------------------------------------------- reduction gate

def _cap(edges, cap):
    deg = pd.concat([edges["src"], edges["dst"]]).value_counts()
    hot = set(deg[deg > cap].index)
    return edges[~(edges["src"].isin(hot) | edges["dst"].isin(hot))].reset_index(drop=True)


H, A, B, C, D, E = 1, 2, 3, 4, 5, 6
RED_EDGES = edges_frame([(H, A, 1, 0), (H, B, 1, 0), (H, C, 10, 0),
                         (A, B, 5, 1), (B, D, 6, 1), (C, D, 2, 1), (E, D, 8, 1)])
RED_PTX = ptx_frame([(0, "STACK", A, B, 0, 5.0), (0, "STACK", B, D, 1, 6.0),
                     (1, "FAN-IN", C, D, 0, 2.0), (1, "FAN-IN", E, D, 1, 8.0)])


def test_reduction_gate_threshold_hand_case():
    # amounts sorted 1 1 2 5 6 8 10; target 0.5 -> index 3 -> tau 5 -> keeps 5 6 8 10
    r = fg.reduction_gate(RED_EDGES, RED_PTX, caps=[], removal_fractions=[0.5], apply_hub_cap=_cap)
    t = r[r["reduction"] == "amount_threshold"].iloc[0]
    assert t.parameter == 5 and t.n_transactions == 4 and t.removed_fraction == pytest.approx(3 / 7)
    assert t.n_accounts == 6 and t.max_degree == 2          # edges A-B, B-D, E-D, H-C
    assert t.frac_attempts_intact == 0.5 and t.intact_STACK == 1.0 and t["intact_FAN-IN"] == 0.0
    assert t.frac_laundering_transactions_kept == 0.75        # 5, 6, 8 of 5, 6, 2, 8
    none = r[r["reduction"] == "none"].iloc[0]
    assert none.n_transactions == 7 and none.frac_attempts_intact == 1.0


def test_reduction_gate_degree_cap_hand_case():
    # degrees: H3 A2 B3 C2 D3 E1. cap 2 removes H, B, D -> no edge left; cap 3 removes nothing
    r = fg.reduction_gate(RED_EDGES, RED_PTX, caps=[2, 3], removal_fractions=[], apply_hub_cap=_cap)
    c2 = r[(r["reduction"] == "degree_cap") & (r["parameter"] == 2)].iloc[0]
    assert c2.n_transactions == 0 and c2.frac_attempts_intact == 0.0
    c3 = r[(r["reduction"] == "degree_cap") & (r["parameter"] == 3)].iloc[0]
    assert c3.n_transactions == 7 and c3.frac_attempts_intact == 1.0 and c3.max_degree == 3


# ----------------------------------------------------- pattern transaction parser

def test_load_pattern_transactions(tmp_path):
    p = tmp_path / "X_Patterns.txt"
    p.write_text(
        "BEGIN LAUNDERING ATTEMPT - FAN-IN:  Max 2-degree Fan-In\n"
        "2022/09/01 00:10,001,A,002,B,105.40,Yen,105.40,Yen,ACH,1\n"
        "2022/09/01 00:12,003,C,002,B,10.00,US Dollar,10.00,US Dollar,ACH,1\n"
        "END LAUNDERING ATTEMPT - FAN-IN\n\n"
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 2 hops\n"
        "2022/09/01 01:00,002,B,009,Z,1.00,US Dollar,1.00,US Dollar,ACH,1\n"
        "END LAUNDERING ATTEMPT - CYCLE\n", encoding="utf-8")
    code = {"001_A": 0, "002_B": 1, "003_C": 2}
    df = load_pattern_transactions(p, code, {"US Dollar": 1.0, "Yen": 105.4})
    assert list(df["attempt"]) == [0, 0, 1] and list(df["type"]) == ["FAN-IN", "FAN-IN", "CYCLE"]
    assert list(df["src"]) == [0, 2, 1] and list(df["dst"]) == [1, 1, -1]   # Z is unknown
    assert df["amount"].tolist() == pytest.approx([1.0, 10.0, 1.0])
    assert df["time"].iloc[1] - df["time"].iloc[0] == 2


def test_load_pattern_transactions_refuses_unknown_currency(tmp_path):
    p = tmp_path / "X_Patterns.txt"
    p.write_text("BEGIN LAUNDERING ATTEMPT - STACK:\n"
                 "2022/09/01 00:10,001,A,002,B,1,Ruble,1,Ruble,ACH,1\nEND LAUNDERING ATTEMPT\n")
    with pytest.raises(ValueError, match="Ruble"):
        load_pattern_transactions(p, {}, {"US Dollar": 1.0})
