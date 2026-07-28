"""Strict-inductive protocol + statistical reporting.

Invariant principles:
1. All statistics (scaler, utility threshold, discretized label table) are fit only on
   step <= 34 data, then frozen and applied to the test set.
2. 10 independent seeds, report mean+/-std.
3. Bootstrap 95% CI (resample test rows) on the median seed.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

DEFAULT_SEEDS = tuple(range(10))


def bootstrap_ci_f1(y_true, y_pred, n_resamples: int = 10_000, seed: int = 0,
                    alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the illicit-class F1 over test rows."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        stats[i] = f1_score(y_true[idx], y_pred[idx], pos_label=1, zero_division=0)
    return (
        float(np.quantile(stats, alpha / 2)),
        float(np.quantile(stats, 1 - alpha / 2)),
    )


def summarize_seeds(per_seed_f1: dict[int, float]) -> dict:
    vals = np.array(list(per_seed_f1.values()))
    seeds = list(per_seed_f1.keys())
    median_seed = seeds[int(np.argsort(vals)[len(vals) // 2])]
    return {
        "f1_mean": float(vals.mean()),
        "f1_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "median_seed": median_seed,
    }
