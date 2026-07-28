"""Publication figures for the experiments (English labels; figures stay English so a
translation of the text does not require redrawing). Reads the deterministic result CSVs and
writes vector PDFs into bai_bao3/. Distinct in content from the motif paper's figures: the axis here is
the UTILITY MEASURE (log / raw / bottleneck-flow), which is unique to this paper.

  Figure_4_utility_dimension.pdf : (a) share of score variance carried by the utility dimension
                               across measures (the R-ceiling and its escape);
                               (b) recovery of the multi-hop typologies by measure.
  Figure_3_recovery_hubcap.pdf   : recovery of the flow-defined typologies against the degree cap,
                               truth-neighbourhood vs full graph (the tractability-signal tension).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parents[3] / "bai_bao3"

# colourblind-safe, muted (Okabe-Ito subset)
C_LOG, C_RAW, C_FLOW = "#999999", "#0072B2", "#D55E00"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})


def piv(fn):
    return pd.read_csv(RES / fn).pivot_table(index="type", columns="hub_cap", values="recovery_rate")


def fig_utility_dimension():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # (a) variance share carried by the utility dimension (HI-Small, cap 0)
    measures = ["HUSM\n(log table)", "HUSM\n(raw table)", r"$u_{\mathrm{flow}}$" + "\n(bottleneck)"]
    share = [2.1, 11.1, 59.2]
    bars = axL.bar(measures, share, color=[C_LOG, C_RAW, C_FLOW], width=0.62)
    axL.axhline(8.0, ls="--", lw=1, color="black")
    axL.text(2.42, 9.0, "Gaussian\nreference 8.0%", ha="right", va="bottom", fontsize=7.5)
    for b, v in zip(bars, share):
        axL.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=8)
    axL.set_ylabel("Score variance from the\nutility dimension (%)")
    axL.set_ylim(0, 66)
    axL.set_title("(a) The utility dimension escapes the ceiling", fontsize=9, loc="left")

    # (b) multi-hop recovery by measure (HI-Small, cap 0)
    log, raw, flow = piv("e9_portsplit_recovery_HI-Small_truth-neighborhood.csv"), \
        piv("e9_portsplit_recovery_HI-Small_truth-neighborhood_raw.csv"), \
        piv("e9_flow_recovery_HI-Small_truth-neighborhood.csv")
    typ = ["STACK", "CYCLE", "RANDOM"]
    lab = ["Stack", "Cycle", "Random"]
    import numpy as np
    x = np.arange(len(typ)); w = 0.26
    axR.bar(x - w, [log.loc[t, 0] for t in typ], w, label="HUSM (log)", color=C_LOG)
    axR.bar(x, [raw.loc[t, 0] for t in typ], w, label="HUSM (raw)", color=C_RAW)
    axR.bar(x + w, [flow.loc[t, 0] for t in typ], w, label=r"$u_{\mathrm{flow}}$", color=C_FLOW)
    axR.set_xticks(x); axR.set_xticklabels(lab)
    axR.set_ylabel("Recovery rate")
    axR.set_ylim(0, 1.0)
    axR.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=1)
    axR.set_title("(b) Multi-hop recovery by measure", fontsize=9, loc="left")

    fig.tight_layout()
    fig.savefig(OUT / "Figure_4_utility_dimension.pdf")
    plt.close(fig)
    print("wrote Figure_4_utility_dimension.pdf")


def fig_recovery_hubcap():
    import numpy as np
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    tn = piv("e9_portsplit_recovery_HI-Small_truth-neighborhood_raw.csv")
    fu = piv("e9_portsplit_recovery_HI-Small_full_raw.csv")
    flow_typ = ["GATHER-SCATTER", "SCATTER-GATHER", "FAN-IN", "FAN-OUT"]
    caps = [0, 200, 50]
    xlab = ["no cap", "200", "50"]
    tn_mean = [np.mean([tn.loc[t, c] for t in flow_typ]) for c in caps]
    fu_mean = [np.mean([fu.loc[t, c] for t in flow_typ]) if c in fu.columns else np.nan
               for c in caps]
    x = np.arange(len(caps))
    ax.plot(x, tn_mean, "o-", color=C_RAW, label="Truth-neighbourhood")
    ax.plot(x, fu_mean, "s--", color=C_FLOW, label="Full graph")
    ax.set_xticks(x); ax.set_xticklabels(xlab)
    ax.set_xlabel("Degree cap")
    ax.set_ylabel("Mean recovery of\nflow-defined typologies")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Hub cap isolates the degree effect", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_3_recovery_hubcap.pdf")
    plt.close(fig)
    print("wrote Figure_3_recovery_hubcap.pdf")


if __name__ == "__main__":
    fig_utility_dimension()
    fig_recovery_hubcap()
