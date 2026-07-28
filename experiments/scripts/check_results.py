"""Self-grade PASS/FAIL of the DETERMINISTIC values after a consolidated run.

Checks only values that must reproduce exactly on any machine: exactness columns, pattern and
state counts, recovery rates, and the fixed-seed F1 figures. It does NOT check timing -- that
varies by machine and by load, and the paper deliberately reports no speedup ratio for the
window modes. Reference values match docs/reproduce/paper3.md and the tables of the paper.

Run after run_paper3.bat:
    "%USERPROFILE%\\.venvs\\husm-aml-portsplit\\Scripts\\python.exe" experiments\\scripts\\check_results.py
No external libraries needed (reads CSV with the standard library only).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

RESULTS = Path(__file__).resolve().parents[1] / "results"
results: list[tuple[str, bool, str]] = []


def rows(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def record(label: str, ok: bool, detail: str) -> None:
    results.append((label, ok, detail))


def near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def check_window(csv_name: str, mode: str, exp_rows: int, exp_peak: int,
                 exp_final: int, exp_high: int) -> None:
    """One mode of E6: exactness after every snapshot plus the state curve."""
    r = rows(csv_name)
    if r is None:
        record(f"E6 {mode}", False, f"missing {csv_name}")
        return
    r = [x for x in r if x.get("mode") == mode]
    if not r:
        record(f"E6 {mode}", False, f"no rows with mode={mode}")
        return
    tracked = [int(x["tracked"]) for x in r]
    n_ok = sum(1 for x in r if x["match"] == "OK")
    peak, final, high = max(tracked), tracked[-1], int(r[-1]["high_total"])
    ok = (len(r) == exp_rows and n_ok == exp_rows and peak == exp_peak
          and final == exp_final and high == exp_high)
    record(f"E6 {mode}", ok,
           f"exact {n_ok}/{len(r)} (need {exp_rows}) | peak {peak}/{exp_peak} | "
           f"final {final}/{exp_final} | high {high}/{exp_high}")


def check_e5b() -> None:
    r = rows("e5b_bhus_stream.csv")
    if r is None:
        record("E5b BHUS-inc", False, "missing e5b_bhus_stream.csv")
        return
    last = r[-1]
    got = (int(last["tracked"]), int(last["high_total"]),
           int(last["max_bhus"]), int(last["min_bhus"]))
    exp = (542, 153, 65, 13)
    record("E5b BHUS-inc", got == exp,
           f"(tracked, high, max_bhus, min_bhus) = {got}, need {exp}")


def check_e5c() -> None:
    r = rows("e5c_inchusm_mu.csv")
    if r is None:
        record("E5c IncHUSM-mu", False, "missing e5c_inchusm_mu.csv")
        return
    last = r[-1]
    got = (int(last["tracked"]), int(last["high_total"]), int(last["large"]))
    exp = (107, 30, 14)
    record("E5c IncHUSM-mu", got == exp,
           f"(tracked, high, large) = {got}, need {exp}")


def check_recovery() -> None:
    """Recovery over the eight typologies at four scales."""
    exp = {
        "HI-Small": {"FAN-IN": 1.00, "FAN-OUT": 1.00, "GATHER-SCATTER": 1.00,
                     "BIPARTITE": 0.71, "SCATTER-GATHER": 0.70, "CYCLE": 0.63,
                     "RANDOM": 1.00, "STACK": 1.00},
        "LI-Small": {"FAN-IN": 1.00, "FAN-OUT": 1.00, "GATHER-SCATTER": 1.00,
                     "BIPARTITE": 1.00, "SCATTER-GATHER": 0.62, "CYCLE": 0.50,
                     "RANDOM": 1.00, "STACK": 1.00},
        "HI-Medium": {"FAN-IN": 1.00, "FAN-OUT": 0.99, "GATHER-SCATTER": 1.00,
                      "BIPARTITE": 0.69, "SCATTER-GATHER": 0.66, "CYCLE": 0.54,
                      "RANDOM": 0.77, "STACK": 0.52},
        "LI-Medium": {"FAN-IN": 1.00, "FAN-OUT": 0.99, "GATHER-SCATTER": 1.00,
                      "BIPARTITE": 0.79, "SCATTER-GATHER": 0.78, "CYCLE": 0.46,
                      "RANDOM": 0.84, "STACK": 0.43},
    }
    for ds, want in exp.items():
        r = rows(f"e3plus_recovery8_{ds}.csv")
        if r is None:
            record(f"E3+ {ds}", False, "not run (Medium needs --include-medium)")
            continue
        got = {x["type"]: float(x["recovery_rate"]) for x in r}
        bad = [f"{t}:{got.get(t, '-'):.2f}!={v:.2f}" if t in got else f"{t}:missing"
               for t, v in want.items() if t not in got or not near(got[t], v, 0.005)]
        record(f"E3+ {ds}", not bad, "8/8 typologies match" if not bad else "; ".join(bad))


def check_e4() -> None:
    """The detection table needs all three rows, and the neural rows only exist when the
    machine has torch_geometric -- without it the driver skips them and writes RF alone,
    which silently leaves the paper's GNN numbers unbacked by any artifact."""
    r = rows("e4plus_gnn_drift.csv")
    if r is None:
        record("E4+ detection", False, "missing e4plus_gnn_drift.csv")
        return
    exp = {"rf": 0.826, "sage": 0.692, "gcn": 0.531}
    got, bad = {}, []
    for x in r:
        key = str(x.get("method", "")).lower()
        for k in exp:
            if k in key:
                got[k] = float(x.get("f1_illicit") or x.get("f1") or 0)
    missing = [k for k in exp if k not in got]
    if missing == ["sage", "gcn"]:
        record("E4+ detection", False,
               "Random forest only: the neural rows are absent, so the GraphSAGE and GCN "
               "reproduced here -- install torch_geometric and re-run e4plus_gnn_drift.py")
        return
    for k, v in exp.items():
        if k not in got:
            bad.append(f"{k}:missing")
        elif not near(got[k], v, 0.01):
            bad.append(f"{k}:{got[k]:.3f}!={v:.3f}")
    record("E4+ detection", not bad,
           "RF/SAGE/GCN match" if not bad else "; ".join(bad))


def check_e7() -> None:
    """E7 is the NEGATIVE result: mining must add nothing over degree."""
    r = rows("e7_actor_hus_detection.csv")
    if r is None:
        record("E7 actor-level (negative)", False, "missing e7_actor_hus_detection.csv")
        return
    by = {x["tag"]: x for x in r}
    bad = []
    base = by.get("base")
    if base is None or not near(float(base["f1"]), 0.371, 0.01):
        bad.append("base != 0.371")
    d = by.get("delta_hus_over_deg")
    if d is None or abs(float(d["f1"])) > 0.01:
        bad.append("HUS-over-degree delta is not ~0 (the negative result changed!)")
    df = by.get("delta_degfull_over_base")
    if df is None or not near(float(df["f1"]), 0.183, 0.02):
        bad.append("full-graph degree gain != +0.183")
    record("E7 actor-level (negative)", not bad,
           "null vs degree kept; hub-cap contrast kept" if not bad else "; ".join(bad))


def check_e8() -> None:
    """E8 carries the paper's claim that the published utility protocols are near-degenerate.

    Every value here is deterministic: the Gaussian table is a fixed seed, the re-scoring is
    arithmetic on the mined set, and the Citeseer runs share a threshold.
    """
    r = rows("e8_utility_contrast.csv")
    if r is None:
        record("E8 utility contrast", False, "missing e8_utility_contrast.csv")
        return
    by = {x["arm"]: x for x in r}
    bad = []
    cite = by.get("citeseer/gaussian")
    if cite is None:
        bad.append("citeseer arm absent (needs datasets/citeseer/citeseer.lg)")
    else:
        # The headline: HUSM's own protocol on HUSM's own graph nearly reproduces plain FSM.
        # These are MEDIANS over the ten seeds, not seed 42 -- seed 42 alone gives 0.047, the
        # lowest of the ten, and reporting it would understate the utility factor.
        if not near(float(cite.get("jaccard_vs_fsm") or 0), 0.825, 0.005):
            bad.append(f"citeseer median Jaccard vs FSM = {cite.get('jaccard_vs_fsm')} != 0.825")
        if not near(float(cite["var_share_ubar"]), 0.080, 0.005):
            bad.append(f"citeseer utility carries {cite['var_share_ubar']} of the variance,"
                       f" not 0.080")
    for arm, share, forced in [("ellipticpp/monetary", 0.334, 0.000),
                               ("ellipticpp/gaussian (re-scored)", 0.010, 0.696)]:
        x = by.get(arm)
        if x is None:
            bad.append(f"{arm}: missing")
            continue
        if not near(float(x["var_share_ubar"]), share, 0.005):
            bad.append(f"{arm}: var share {x['var_share_ubar']} != {share}")
        if not near(float(x["forced_pairs"]), forced, 0.005):
            bad.append(f"{arm}: forced pairs {x['forced_pairs']} != {forced}")
    record("E8 utility contrast", not bad,
           "Citeseer ~= FSM (median Jaccard 0.825 over 10 seeds); monetary revives the "
           "utility dimension" if not bad else "; ".join(bad))


def main() -> None:
    # E6 -- exactness and bounded state, both streams
    check_window("e6_sliding_window_elliptic.csv", "add-only", 49, 266, 266, 83)
    check_window("e6_sliding_window_elliptic.csv", "W=10", 49, 175, 95, 32)
    check_window("e6_sliding_window_elliptic.csv", "W=5", 49, 117, 76, 13)
    check_window("e6_sliding_window_actor.csv", "W=5", 49, 215, 132, 29)
    check_e5b()
    check_e5c()
    check_recovery()
    check_e4()
    check_e7()
    check_e8()

    width = max(len(n) for n, _, _ in results)
    print("=" * 78)
    print("this work -- deterministic values (timing is NOT checked)")
    print("=" * 78)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    n_ok = sum(1 for _, ok, _ in results if ok)
    print("-" * 78)
    print(f"{n_ok}/{len(results)} checks passed")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
