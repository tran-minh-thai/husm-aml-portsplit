"""Build a frozen external-utility table from real monetary features.

The miner scores a pattern by image-based support times the mean external utility of its
vertex labels (the same definition as the earlier work). Here the external utility of a
label is derived from real transaction value on the Elliptic++ actor graph, rather than a
value drawn from a synthetic distribution.

Design (kept explicit so it is reproducible and leakage-free):
- The monetary signal per wallet is log(1 + btc_transacted_total) scaled by (1 + fee share),
  so both transaction volume and fee behavior contribute.
- Vertex labels are quantile buckets of that signal; the bucket edges are estimated on the
  training window and then frozen, so no test-period information enters the utility.
- The external utility of a bucket is its mean signal on the training window (a real
  monetary quantity), not an ordinal placeholder such as i+1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class UtilityBuckets:
    """Frozen quantile edges plus the {bucket label -> external utility} table."""

    edges: list[float]                 # inner quantile edges (length n_buckets - 1)
    label_utility: dict[int, float]    # bucket label -> representative monetary utility


def monetary_signal(wallets: pd.DataFrame) -> pd.Series:
    """Real-value signal per wallet: log(1 + btc transacted) scaled by (1 + mean fee share)."""
    btc = np.log1p(wallets["btc_transacted_total"].to_numpy(dtype=np.float64))
    fee = wallets["fees_as_share_mean"].fillna(0.0).to_numpy(dtype=np.float64)
    return pd.Series(btc * (1.0 + fee), index=wallets.index)


def assign_bucket(signal: pd.Series, edges: list[float]) -> pd.Series:
    """Digitize a signal into bucket labels 0..len(edges) using the frozen edges."""
    return pd.Series(np.digitize(signal.to_numpy(), edges), index=signal.index).astype(int)


def fit_monetary_utility(
    wallets: pd.DataFrame, train_mask: pd.Series, n_buckets: int = 4
) -> UtilityBuckets:
    """Fit the monetary-utility buckets on the training wallets and freeze them.

    Parameters
    - wallets: per-wallet table including the monetary columns.
    - train_mask: boolean selector for the training window (edges fit only on these rows).
    - n_buckets: number of quantile buckets (discretized vertex labels).
    """
    signal = monetary_signal(wallets)
    train_signal = signal[train_mask.to_numpy()]
    qs = np.linspace(0.0, 1.0, n_buckets + 1)[1:-1]      # inner quantiles only
    edges = list(np.quantile(train_signal.to_numpy(), qs))

    labels = assign_bucket(signal, edges)
    train_labels = labels[train_mask.to_numpy()]
    label_utility: dict[int, float] = {}
    for b in range(n_buckets):
        vals = train_signal[train_labels == b]
        # a bucket empty on train falls back to the bucket index so utilities stay increasing
        label_utility[b] = float(vals.mean()) if len(vals) else float(b)
    return UtilityBuckets(edges=edges, label_utility=label_utility)
