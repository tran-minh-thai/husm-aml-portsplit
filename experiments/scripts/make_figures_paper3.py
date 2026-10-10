"""Figures for the experiments, drawn only from result files (no value is typed in here).

  Figure_4_utility_dimension.pdf : (a) share of pattern pairs whose order the weight term
                               reverses relative to support, per measure (HUSM on the log
                               and the raw bucket table, and the bottleneck-flow measure),
                               with the Citeseer Gaussian protocol as reference;
                               (b) recovery of the multi-hop typologies by measure.
  Figure_3_recovery_hubcap.pdf   : recovery of the flow-defined typologies against the degree cap,
                               truth-neighbourhood vs full graph.

Panel (a) needs the order statistics that the recovery drivers write since they report the
reversed-pair share (column `reversed_pairs`) and that e8 writes as `order_reversed_frac`.
Result files written before those columns existed carry an older, covariance-free variance
ratio under similar names; the script refuses them rather than plotting a different quantity.

Usage:
  python scripts/make_figures_paper3.py --results <dir with e9_*.csv> --e8 <e8_utility_contrast.csv>
Figures go to --out, by default the record directory when run through
experiments/provenance/run_and_record.py, else experiments/figures.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
DEFAULT_RESULTS = HERE.parents[1] / "results"
DEFAULT_OUT = (Path(os.environ["RUN_RECORD_DIR"]) if os.environ.get("RUN_RECORD_DIR")
               else HERE.parents[1] / "figures")

# colourblind-safe, muted (Okabe-Ito subset)
C_LOG, C_RAW, C_FLOW = "#999999", "#0072B2", "#D55E00"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})


def read(path: Path, needed: list[str]) -> pd.DataFrame:
    """Read a result CSV, refusing it when a needed column is absent."""
    if not path.is_file():
        raise SystemExit(f"missing result file {path}")
    frame = pd.read_csv(path)
    absent = [c for c in needed if c not in frame.columns]
    if absent:
        raise SystemExit(f"{path} has no column(s) {absent}: it was written before the order "
                         f"statistics existed; re-run its driver")
    return frame


def piv(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.pivot_table(index="type", columns="hub_cap", values="recovery_rate")


def at_cap0(frame: pd.DataFrame, column: str, path: Path) -> float:
    vals = frame.loc[frame["hub_cap"] == 0, column].dropna().unique()
    if len(vals) != 1:
        raise SystemExit(f"{path}: expected one {column} value at hub_cap 0, found {list(vals)}")
    return float(vals[0])


def fig_utility_dimension(res: Path, e8: Path, out: Path) -> None:
    f_log = res / "e9_portsplit_recovery_HI-Small_truth-neighborhood_log.csv"
    f_raw = res / "e9_portsplit_recovery_HI-Small_truth-neighborhood_raw.csv"
    f_flow = res / "e9_flow_recovery_HI-Small_truth-neighborhood.csv"
    need = ["type", "hub_cap", "recovery_rate", "reversed_pairs"]
    log, raw, flow = read(f_log, need), read(f_raw, need), read(f_flow, need)
    ref = read(e8, ["arm", "order_reversed_frac"]).set_index("arm")
    if "citeseer/gaussian" not in ref.index:
        raise SystemExit(f"{e8} has no citeseer/gaussian row")
    gauss = 100 * float(ref.loc["citeseer/gaussian", "order_reversed_frac"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    measures = ["HUSM\n(log table)", "HUSM\n(raw table)", r"$u_{\mathrm{flow}}$" + "\n(bottleneck)"]
    share = [100 * at_cap0(d, "reversed_pairs", p)
             for d, p in ((log, f_log), (raw, f_raw), (flow, f_flow))]
    bars = axL.bar(measures, share, color=[C_LOG, C_RAW, C_FLOW], width=0.62)
    axL.axhline(gauss, ls="--", lw=1, color="black")
    axL.text(2.42, gauss, f"Citeseer Gaussian\nreference {gauss:.1f}%", ha="right", va="bottom",
             fontsize=7.5)
    for b, v in zip(bars, share):
        axL.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    axL.set_ylabel("Pattern pairs reversed\nrelative to support (%)")
    axL.set_ylim(0, 1.15 * max(share + [gauss]))
    axL.set_title("(a) Reordering by the weight term", fontsize=9, loc="left")

    plog, praw, pflow = piv(log), piv(raw), piv(flow)
    typ = ["STACK", "CYCLE", "RANDOM"]
    lab = ["Stack", "Cycle", "Random"]
    x = np.arange(len(typ)); w = 0.26
    axR.bar(x - w, [plog.loc[t, 0] for t in typ], w, label="HUSM (log)", color=C_LOG)
    axR.bar(x, [praw.loc[t, 0] for t in typ], w, label="HUSM (raw)", color=C_RAW)
    axR.bar(x + w, [pflow.loc[t, 0] for t in typ], w, label=r"$u_{\mathrm{flow}}$", color=C_FLOW)
    axR.set_xticks(x); axR.set_xticklabels(lab)
    axR.set_ylabel("Recovery rate")
    axR.set_ylim(0, 1.0)
    axR.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=1)
    axR.set_title("(b) Multi-hop recovery by measure", fontsize=9, loc="left")

    fig.tight_layout()
    fig.savefig(out / "Figure_4_utility_dimension.pdf")
    plt.close(fig)
    print("wrote Figure_4_utility_dimension.pdf")


def fig_recovery_hubcap(res: Path, out: Path) -> None:
    need = ["type", "hub_cap", "recovery_rate"]
    tn = piv(read(res / "e9_portsplit_recovery_HI-Small_truth-neighborhood_raw.csv", need))
    fu = piv(read(res / "e9_portsplit_recovery_HI-Small_full_raw.csv", need))
    flow_typ = ["GATHER-SCATTER", "SCATTER-GATHER", "FAN-IN", "FAN-OUT"]
    caps = [0, 200, 50]
    xlab = ["no cap", "200", "50"]
    tn_mean = [np.mean([tn.loc[t, c] for t in flow_typ]) for c in caps]
    fu_mean = [np.mean([fu.loc[t, c] for t in flow_typ]) if c in fu.columns else np.nan
               for c in caps]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = np.arange(len(caps))
    ax.plot(x, tn_mean, "o-", color=C_RAW, label="Truth-neighbourhood")
    ax.plot(x, fu_mean, "s--", color=C_FLOW, label="Full graph")
    ax.set_xticks(x); ax.set_xticklabels(xlab)
    ax.set_xlabel("Degree cap")
    ax.set_ylabel("Mean recovery of\nflow-defined typologies")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Recovery against the degree cap", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out / "Figure_3_recovery_hubcap.pdf")
    plt.close(fig)
    print("wrote Figure_3_recovery_hubcap.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                    help="directory holding the e9 recovery CSVs")
    ap.add_argument("--e8", type=Path, default=DEFAULT_RESULTS / "e8_utility_contrast.csv",
                    help="e8_utility_contrast.csv with order statistics")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_utility_dimension(args.results, args.e8, args.out)
    fig_recovery_hubcap(args.results, args.out)


if __name__ == "__main__":
    main()
