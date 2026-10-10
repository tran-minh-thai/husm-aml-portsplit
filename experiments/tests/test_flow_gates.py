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


# ------------------------------------------------------------------ second round

R2 = ptx_frame([
    # 0 CYCLE o->p (1), p->q (2), q->o (3): pairs op-pq ok, pq-qo ok, qo-op reversed -> V = 1 (a1 and a2)
    (0, "CYCLE", 1, 2, 1, 1.0), (0, "CYCLE", 2, 3, 2, 1.0), (0, "CYCLE", 3, 1, 3, 1.0),
    # 1 GATHER-SCATTER leaf L->C (1), C->L (2): pair at C ok, pair at L reversed; both back-and-forth
    (1, "GATHER-SCATTER", 10, 11, 1, 1.0), (1, "GATHER-SCATTER", 11, 10, 2, 1.0),
    # 2 STACK a->b (5), b->c (5) tie ok, c->d (4) reversed -> V = 1
    (2, "STACK", 20, 21, 5, 1.0), (2, "STACK", 21, 22, 5, 1.0), (2, "STACK", 22, 23, 4, 1.0),
    # 3 FAN-IN: no head-to-tail pair; duration 3
    (3, "FAN-IN", 30, 32, 0, 1.0), (3, "FAN-IN", 31, 32, 3, 1.0),
    # 4 STACK e->f (1), f->g (2): V = 0
    (4, "STACK", 40, 41, 1, 1.0), (4, "STACK", 41, 42, 2, 1.0),
    # a self-transfer is ignored (it would add pairs at 41)
    (4, "STACK", 41, 41, 0, 1.0),
])


def test_head_to_tail_pairs_hand_case():
    p = fg.head_to_tail_pairs(R2)
    assert len(p) == 3 + 2 + 2 + 0 + 1
    gs = p[p["attempt"] == 1]
    assert gs["back_and_forth"].all() and sorted(gs["in_order"]) == [False, True]
    assert p.loc[(p["attempt"] == 2) & (p["t1"] == 5) & (p["t2"] == 5), "in_order"].item()


def test_order_tolerance_gate_hand_case():
    g = fg.order_tolerance_gate(R2).set_index("type")
    c = g.loc["CYCLE"]
    assert c.n_pairs_a1 == 3 and c.in_order_a1 == pytest.approx(2 / 3) and c.in_order_a2 == pytest.approx(2 / 3)
    assert c.attempts_V_le_0_a1 == 0.0 and c.attempts_V_le_1_a1 == 1.0 and c.attempts_V_le_0_a2 == 0.0
    gs = g.loc["GATHER-SCATTER"]
    assert gs.n_pairs_a1 == 2 and gs.in_order_a1 == 0.5 and gs.n_pairs_a2 == 0 and np.isnan(gs.in_order_a2)
    assert gs.attempts_V_le_0_a1 == 0.0 and gs.attempts_V_le_0_a2 == 1.0 and gs.n_back_and_forth == 2
    s = g.loc["STACK"]
    assert s.n_attempts == 2 and s.n_pairs_a1 == 3 and s.in_order_a1 == pytest.approx(2 / 3)
    assert s.attempts_V_le_0_a1 == 0.5 and s.attempts_V_le_1_a1 == 1.0 and s.attempts_V_max_a1 == 1
    f = g.loc["FAN-IN"]
    assert f.n_attempts_with_pair == 0 and np.isnan(f.in_order_a1) and np.isnan(f.attempts_V_le_0_a1)


def test_parallel_transactions_each_form_pairs():
    p = ptx_frame([(0, "STACK", 1, 2, 1, 1.0), (0, "STACK", 1, 2, 3, 1.0), (0, "STACK", 2, 3, 2, 1.0)])
    g = fg.order_tolerance_gate(p).set_index("type").loc["STACK"]
    assert g.n_pairs_a1 == 2 and g.in_order_a1 == 0.5 and g.attempts_V_le_0_a1 == 0.0


def test_window_gate_hand_case():
    w = fg.window_gate(R2, deltas=(1, 2), unit=1).set_index("type")
    assert w.loc["CYCLE", "duration_days_median"] == 2 and w.loc["CYCLE", "within_1d"] == 0.0
    assert w.loc["CYCLE", "within_2d"] == 1.0 and w.loc["CYCLE", "pair_gap_days_median"] == 1
    assert w.loc["STACK", "within_1d"] == 1.0 and w.loc["FAN-IN", "within_2d"] == 0.0
    w2 = fg.window_gate(R2.assign(time=R2["time"] * 1440), deltas=(1,), unit=1440).set_index("type")
    assert w2.loc["STACK", "within_1d"] == 1.0 and w2.loc["CYCLE", "duration_days_max"] == 2


# ------------------------------------------------------------------ AMLGentex loader

def test_load_amlgentex(tmp_path):
    from inchusm_aml.data.amlgentex import load_amlgentex
    f = tmp_path / "tx_log.csv"
    f.write_text(
        "step,type,amount,nameOrig,bankOrig,nameDest,isSAR,alertID,modelType\n"
        "0,INITALBALANCE,10,-2,source,5,0,-1,11\n"      # from the income source: dropped
        "1,TRANSFER,7,5,c0,-1,0,-1,0\n"                 # to the spending sink: dropped
        "2,TRANSFER,3,5,c0,6,0,-1,0\n"                  # background
        "3,TRANSFER,4,6,c0,7,1,-1,0\n"                  # flagged account, no attempt
        "4,TRANSFER,9,7,c0,8,1,12,7\n"                  # attempt 12, scatter-gather
        "5,TRANSFER,8,8,c0,9,1,12,7\n", encoding="utf-8")
    edges, ptx = load_amlgentex(f)
    assert len(edges) == 4 and edges.attrs["rows_special"] == 2 and edges.attrs["sar_without_alert"] == 1
    assert list(edges["laundering"]) == [0, 1, 1, 1]
    assert list(ptx["attempt"]) == [12, 12] and set(ptx["type"]) == {"SCATTER-GATHER"}
    assert list(ptx["time"]) == [4, 5]


def test_load_amlgentex_refuses_unknown_code(tmp_path):
    from inchusm_aml.data.amlgentex import load_amlgentex
    f = tmp_path / "tx_log.csv"
    f.write_text("step,type,amount,nameOrig,nameDest,isSAR,alertID,modelType\n"
                 "1,TRANSFER,1,1,2,1,0,42\n", encoding="utf-8")
    with pytest.raises(ValueError, match="42"):
        load_amlgentex(f)


# ------------------------------------------------------------------- third round

def test_deviation_and_auc_lower_hand_case():
    assert fg.deviation([100, 100], [100, 50]) == pytest.approx([0.0, np.log(2)])
    ref = np.sort(np.array([0.0, 1.0, 2.0, 3.0]))
    # 0.0 vs ref: below 0 of 4, equal 1 -> P(x < ref) + 1/2 P(=) = 3/4 + 1/8 = 0.875
    # 2.5 vs ref: greater refs: 3.0 -> 1/4
    assert fg.auc_lower([0.0], ref) == pytest.approx(0.875)
    assert fg.auc_lower([2.5], ref) == pytest.approx(0.25)
    assert fg.auc_lower([0.0, 2.5], ref) == pytest.approx((0.875 + 0.25) / 2)


def test_conservation_row_hand_case():
    # ratios 1.0, 0.5, 1.15, and one non-positive amount excluded
    row = fg._conservation_row("X", [100, 100, 100, 0], [100, 50, 115, 10], np.sort(np.array([0.1, 1.0])))
    assert row["n_pairs"] == 3 and row["n_nonpositive_excluded"] == 1
    assert row["within_0.05"] == pytest.approx(1 / 3) and row["within_0.2"] == pytest.approx(2 / 3)
    assert row["within_0.5"] == pytest.approx(1.0)                 # 0.5 is the boundary, included
    assert row["frac_decreasing"] == pytest.approx(2 / 3)          # 1.0 (equal) and 0.5
    assert row["ratio_median"] == pytest.approx(1.0)
    # deviations 0, ln2 = 0.693, ln1.15 = 0.140 vs ref 0.1, 1.0: P(<) = 1, 1/2, 1/2 -> 2/3
    assert row["auc_conservation"] == pytest.approx(2 / 3)


def _bg_graph():
    # v = 10 receives from 1, 2, 3 (background) and from 9 (laundering: must never be used);
    # v sends to 20 (background) and back to 1 (background): pairs (1->10, 10->1) are back-and-forth
    return edges_frame([(1, 10, 1.0, 0), (2, 10, 2.0, 0), (3, 10, 3.0, 0), (9, 10, 9.0, 1),
                        (10, 20, 5.0, 0), (10, 1, 7.0, 0), (30, 31, 1.0, 0)])


def test_background_sampler_pairs_are_valid_and_uniform():
    s = fg.sample_background_pairs(_bg_graph(), 30000, seed=7)
    assert len(s) == 30000
    assert (s["v"] != s["u"]).all() and (s["w"] != s["u"]).all()       # no back-and-forth
    assert set(s["u"]) <= {1, 2, 3} and 9 not in set(s["u"])          # laundering never sampled
    assert set(s["v"]) == {10}
    to20 = s[s["w"] == 20]["u"].value_counts(normalize=True)
    assert set(to20.index) == {1, 2, 3} and (to20 - 1 / 3).abs().max() < 0.02
    to1 = s[s["w"] == 1]["u"].value_counts(normalize=True)          # 1 is excluded as sender here
    assert set(to1.index) == {2, 3} and (to1 - 1 / 2).abs().max() < 0.02
    assert s.attrs["rejected_no_incoming"] > 0 and s.attrs["rejected_back_and_forth"] > 0


def test_background_sampler_is_reproducible():
    a = fg.sample_background_pairs(_bg_graph(), 500, seed=3)
    b = fg.sample_background_pairs(_bg_graph(), 500, seed=3)
    assert a.equals(b)


def test_conservation_gate_skips_back_and_forth():
    ptx = ptx_frame([(0, "STACK", 1, 2, 0, 100.0), (0, "STACK", 2, 3, 1, 100.0), (0, "STACK", 3, 4, 2, 50.0),
                     (1, "GATHER-SCATTER", 5, 6, 0, 10.0), (1, "GATHER-SCATTER", 6, 5, 1, 10.0)])
    g = fg.conservation_gate(_bg_graph(), ptx, 1000, seed=1).set_index("group")
    assert g.loc["STACK", "n_pairs"] == 2 and "GATHER-SCATTER" not in g.index
    assert g.loc[fg.ALL_PAIRS_LABEL, "n_pairs"] == 2 and g.loc[fg.BACKGROUND_LABEL, "n_pairs"] == 1000


# ------------------------------------------------------------------- fourth round

ROLE_PTX = ptx_frame([
    # attempt 0 (STACK): a->v 100, v->c 100: v is a chain node (ratio 1)
    (0, "STACK", 1, 2, 0, 100.0), (0, "STACK", 2, 3, 1, 100.0),
    # attempt 1 (GATHER-SCATTER): x->h 50, y->h 50, h->z 80: h merges (k_in 2, k_out 1, ratio 0.8);
    # h->s 30, h->t 30 from attempt 2 must not count here
    (1, "GATHER-SCATTER", 10, 12, 0, 50.0), (1, "GATHER-SCATTER", 11, 12, 0, 50.0), (1, "GATHER-SCATTER", 12, 13, 1, 80.0),
    # attempt 2 (SCATTER-GATHER): g->h 60, h->s 30, h->t 30: h splits (k_in 1, k_out 2, ratio 1.0)
    (2, "SCATTER-GATHER", 20, 12, 0, 60.0), (2, "SCATTER-GATHER", 12, 21, 1, 30.0), (2, "SCATTER-GATHER", 12, 22, 1, 30.0),
    # attempt 3 (CYCLE): p->q 10, q->p 10: both chain nodes, both pairs back-and-forth -> no chain pair
    (3, "CYCLE", 30, 31, 0, 10.0), (3, "CYCLE", 31, 30, 1, 10.0),
])


def test_node_roles_hand_case():
    r = fg.node_roles(ROLE_PTX).set_index(["attempt", "account"])
    assert r.loc[(0, 2), "role"] == "chain"
    h1 = r.loc[(1, 12)]
    assert h1.role == "split_merge" and h1.k_in == 2 and h1.k_out == 1 and h1.sum_out / h1.sum_in == pytest.approx(0.8)
    h2 = r.loc[(2, 12)]
    assert h2.role == "split_merge" and h2.k_in == 1 and h2.k_out == 2 and h2.sum_out == 60
    assert r.loc[(3, 30), "role"] == "chain" and r.loc[(3, 31), "role"] == "chain"
    assert (1, 10) not in r.index                                   # a pure sender is no role


def test_chain_pairs_skip_back_and_forth():
    cp = fg.chain_pairs(ROLE_PTX)
    assert len(cp) == 1 and cp.iloc[0]["v"] == 2 and cp.iloc[0]["a1"] == 100 and cp.iloc[0]["a2"] == 100


def _matched_graph():
    # account 5: background in 1, 2, 3 (amounts 10, 20, 30), out 4 (40), 6 (60); laundering in from 9
    # account 7: background in 1 (5), out 8 (5) -> one in, one out
    return edges_frame([(1, 5, 10.0, 0), (2, 5, 20.0, 0), (3, 5, 30.0, 0), (9, 5, 999.0, 1),
                        (5, 4, 40.0, 0), (5, 6, 60.0, 0), (1, 7, 5.0, 0), (7, 8, 5.0, 0)])


def test_matched_background_deviations():
    devs, unmatched = fg.matched_background_deviations(_matched_graph(), [(3, 2), (2, 2), (4, 1)], 200, seed=5)
    # (3, 2): only account 5 qualifies, all its background transactions: |ln(100 / 60)|
    assert len(devs[0]) == 200 and np.allclose(devs[0], abs(np.log(100 / 60)))
    # (2, 2): account 5, any 2 of {10, 20, 30} in: sums 30, 40, 50 vs out 100
    assert set(np.round(devs[1], 6)) <= {round(abs(np.log(100 / s)), 6) for s in (30, 40, 50)}
    assert len(set(np.round(devs[1], 6))) == 3
    # (4, 1): no account has 4 background incoming transactions (the laundering one does not count)
    assert unmatched == 1 and len(devs[2]) == 0
    again, _ = fg.matched_background_deviations(_matched_graph(), [(2, 2)], 50, seed=5)
    again2, _ = fg.matched_background_deviations(_matched_graph(), [(2, 2)], 50, seed=5)
    assert np.array_equal(again[0], again2[0])


def test_role_gate_tables():
    tab_i, tab_ii = fg.role_gate(_matched_graph(), ROLE_PTX, 500, seed_pairs=1, replicates=20, seed_matched=2)
    ti = tab_i.set_index("type")
    assert list(ti.index) == ["STACK"] and ti.loc["STACK", "n_chain_pairs"] == 1
    assert tab_i.attrs["n_one_in_one_out_background_accounts"] == 1   # account 7 only
    tii = tab_ii.set_index("type")
    assert tii.loc["GATHER-SCATTER", "n_split_merge_nodes"] == 1 and tii.loc["GATHER-SCATTER", "ratio_median"] == pytest.approx(0.8)
    assert tii.loc["SCATTER-GATHER", "ratio_median"] == pytest.approx(1.0)
    # the split node (ratio 1, deviation 0) beats every matched replicate of account 5 -> AUC 1
    assert tii.loc["SCATTER-GATHER", "auc_ii"] == pytest.approx(1.0)
