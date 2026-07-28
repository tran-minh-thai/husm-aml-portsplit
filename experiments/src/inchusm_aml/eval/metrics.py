"""The project's standard metric: illicit-class F1 + per-step + stable/collapse zones.

Protocol: do NOT use accuracy/micro-F1; every detection result must include per-step
F1 and split the 35-42 / 43-49 zones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def illicit_report(y_true, y_pred, y_score=None) -> dict:
    out = {
        "f1_illicit": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    if y_score is not None:
        out["pr_auc"] = average_precision_score(y_true, y_score)
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    return out


def per_step_f1(steps, y_true, y_pred) -> pd.DataFrame:
    df = pd.DataFrame({"step": steps, "y": y_true, "p": y_pred})
    rows = []
    for s, g in df.groupby("step"):
        rows.append(
            {
                "step": s,
                "n": len(g),
                "illicit_pct": 100.0 * (g["y"] == 1).mean(),
                "f1": f1_score(g["y"], g["p"], pos_label=1, zero_division=0),
                "precision": precision_score(g["y"], g["p"], pos_label=1, zero_division=0),
                "recall": recall_score(g["y"], g["p"], pos_label=1, zero_division=0),
            }
        )
    return pd.DataFrame(rows).set_index("step")


def zone_summary(steps, y_true, y_pred) -> dict:
    steps = np.asarray(steps)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    for name, lo, hi in [("stable_35_42", 35, 42), ("collapse_43_49", 43, 49)]:
        m = (steps >= lo) & (steps <= hi)
        out[name] = f1_score(y_true[m], y_pred[m], pos_label=1, zero_division=0)
    return out


def drift_retention(zones: dict) -> float:
    """Fraction of the stable-zone F1 retained across the step-43 shift (collapse / stable).

    Near 1 means the method is robust to the distribution shift; near 0 means it collapses.
    A single scalar for the drift-robustness comparison in E4+.
    """
    stable = zones.get("stable_35_42", 0.0)
    collapse = zones.get("collapse_43_49", 0.0)
    return collapse / stable if stable > 0 else float("nan")
