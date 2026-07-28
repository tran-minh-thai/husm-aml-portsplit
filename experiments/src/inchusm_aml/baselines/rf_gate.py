"""E0/E4  -  RF gate baseline, strict inductive.

Reference target: RF raw 165-d, illicit F1 = 0.821 +/- 0.003, CI [0.80-0.84]
(Maganti 2026, Table 6). LR raw ~ 0.53.

E4 reuses this same function with extra_features = per-node structural motif flags
(fan-in/fan-out/gather-scatter, smurf-context degrees, two-step reach, cycle flags),
computed per time step by features/motif_features.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from ..data.elliptic import EllipticData
from ..eval.metrics import illicit_report, per_step_f1, zone_summary
from ..protocol import DEFAULT_SEEDS, bootstrap_ci_f1, summarize_seeds


def _f1_optimal_threshold(y_true, scores, grid=200) -> float:
    """Threshold maximizing the illicit-class F1  -  call ONLY on TRAIN labels/scores (OOB)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, grid):
        f1 = f1_score(y_true, (scores >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def run_rf_gate(
    data: EllipticData,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_estimators: int = 100,
    extra_features: pd.DataFrame | None = None,
    model: str = "rf",
    class_weight: str | None = None,      # 'balanced_subsample' handles the 2% imbalance
    tune_threshold: bool = False,          # F1-optimal threshold, chosen on OOB train (no leakage)
    max_depth: int | None = None,
) -> dict:
    """Run the baseline on the standard temporal split; return metrics + per-step table.

    extra_features: DataFrame index=txId  -  appended to the 165 raw features (E4).
                    CALLER'S OBLIGATION: every column must be computed only from data <= t.
    class_weight/tune_threshold/max_depth: VALID optimization levers (applied to EVERY
                    variant for a fair comparison); the threshold is tuned only on the
                    train OOB => strict-inductive is preserved, the test is never seen.
    """
    train, test = data.split_labeled()
    cols = data.feature_cols

    x_train = train[cols].to_numpy()
    x_test = test[cols].to_numpy()
    if extra_features is not None:
        x_train = np.hstack(
            [x_train, extra_features.reindex(train["txId"]).fillna(0.0).to_numpy()]
        )
        x_test = np.hstack(
            [x_test, extra_features.reindex(test["txId"]).fillna(0.0).to_numpy()]
        )
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    steps_test = test["step"].to_numpy()

    if model == "lr":
        # Scaler fit TRAIN-ONLY (avoid leakage via batch statistics)
        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)

    per_seed_f1: dict[int, float] = {}
    per_seed_pred: dict[int, np.ndarray] = {}
    per_seed_score: dict[int, np.ndarray] = {}
    per_seed_threshold: dict[int, float] = {}
    for seed in seeds:
        if model == "rf":
            clf = RandomForestClassifier(
                n_estimators=n_estimators, n_jobs=-1, random_state=seed,
                class_weight=class_weight, max_depth=max_depth,
                oob_score=tune_threshold, bootstrap=True,
            )
        elif model == "lr":
            clf = LogisticRegression(
                max_iter=2000, random_state=seed, class_weight=class_weight
            )
        else:
            raise ValueError(model)
        clf.fit(x_train, y_train)
        score = clf.predict_proba(x_test)[:, 1]
        if tune_threshold and model == "rf":
            oob = clf.oob_decision_function_[:, 1]
            ok = ~np.isnan(oob)
            thr = _f1_optimal_threshold(y_train[ok], oob[ok])
            pred = (score >= thr).astype(int)
        else:
            thr = 0.5
            pred = clf.predict(x_test)
        per_seed_threshold[seed] = thr
        per_seed_pred[seed] = pred
        per_seed_score[seed] = score
        per_seed_f1[seed] = illicit_report(y_test, pred)["f1_illicit"]

    summary = summarize_seeds(per_seed_f1)
    med = summary["median_seed"]
    report = illicit_report(y_test, per_seed_pred[med], per_seed_score[med])
    ci = bootstrap_ci_f1(y_test, per_seed_pred[med])

    return {
        "model": model,
        "n_features": x_train.shape[1],
        **summary,
        "median_seed_report": report,
        "f1_ci95": ci,
        "per_step": per_step_f1(steps_test, y_test, per_seed_pred[med]),
        "zones": zone_summary(steps_test, y_test, per_seed_pred[med]),
        "per_seed_f1": per_seed_f1,
        "median_threshold": per_seed_threshold[med],
    }
