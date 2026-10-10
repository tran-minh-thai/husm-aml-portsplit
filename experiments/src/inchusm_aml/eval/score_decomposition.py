"""How much of a pattern score's spread comes from support and how much from the weight term.

Every score this project studies has the form score(P) = support(P) * weight(P): HUSM's utility
(CMNI times the mean label utility) and the bottleneck-flow measure (CMNI times the bottleneck
amount). On the log scale Z = X + Y with X = log support and Y = log weight, and

    Var(Z) = Var(X) + Var(Y) + 2 Cov(X, Y).

The ratio Var(Y) / Var(Z) ignores the covariance: it need not lie in [0, 1], and it does not add
up with Var(X) / Var(Z) to one. This module reports the three variance terms themselves and, as
a summary, the Shapley allocation of the two-player game v(S) = Var(sum of the terms in S):

    share_support = (Var(X) + Cov(X, Y)) / Var(Z),   share_weight = (Var(Y) + Cov(X, Y)) / Var(Z),

which always sum to one and split the covariance equally. A share can be negative when the
covariance is negative and larger in magnitude than that term's own variance; the raw terms are
reported so the reader can see when that happens. All variances are population variances
(ddof = 0); the ratios do not depend on that choice.
"""

from __future__ import annotations

import numpy as np


def variance_decomposition(support, weight) -> dict[str, float]:
    """Decompose Var(log(support * weight)) over the patterns given.

    support and weight are positive arrays of equal length (one entry per pattern). Returns
    n, var_log_support, var_log_weight, cov_log, var_log_score, share_support, share_weight,
    and ratio_weight_no_cov (the earlier Var(Y)/Var(Z), kept only for comparison). Shares are
    NaN, with degenerate = 1, when fewer than two patterns are given or Var(Z) is zero.
    """
    s = np.asarray(support, dtype=float)
    w = np.asarray(weight, dtype=float)
    if s.shape != w.shape:
        raise ValueError(f"support has shape {s.shape} but weight has shape {w.shape}")
    if np.any(s <= 0) or np.any(w <= 0):
        raise ValueError(f"support and weight must be positive: {int(np.sum(s <= 0))} support and "
                         f"{int(np.sum(w <= 0))} weight entries are not")
    n = int(s.size)
    out = {"n": n, "var_log_support": float("nan"), "var_log_weight": float("nan"),
           "cov_log": float("nan"), "var_log_score": float("nan"), "share_support": float("nan"),
           "share_weight": float("nan"), "ratio_weight_no_cov": float("nan"), "degenerate": 1}
    if n < 2:
        return out
    x, y = np.log(s), np.log(w)
    vx, vy = float(np.var(x)), float(np.var(y))
    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    vz = float(np.var(x + y))
    out.update(var_log_support=vx, var_log_weight=vy, cov_log=cov, var_log_score=vz)
    # Var(Z) is a difference of floating sums; treat it as zero when it is lost in their rounding
    if vz <= 1e-12 * max(vx + vy, 1e-300):
        return out
    out.update(share_support=(vx + cov) / vz, share_weight=(vy + cov) / vz,
               ratio_weight_no_cov=vy / vz, degenerate=0)
    return out


def order_statistics(support, weight, R: float) -> dict[str, float]:
    """How often the weight term actually reorders patterns relative to support.

    Over all unordered pairs {P, Q} with different support, a pair is
      * locked when the larger support exceeds the smaller by more than the factor R
        (no weight in a table of spread R can reverse it), and
      * reversed when score = support * weight orders it the other way round from support.
    Pairs with equal support are not comparable by support and are counted separately; pairs
    with equal score are not reversed. Every reversed pair must be unlocked, so
    `violations` (locked and reversed) must be 0 for every R that bounds the weights' spread:
    a nonzero count means either R is wrong or the score is not support * weight.

    Returns pairs, comparable_pairs, locked_frac, reversed_frac (both over comparable pairs),
    reversed_among_unlocked, and violations.
    """
    s = np.asarray(support, dtype=float)
    w = np.asarray(weight, dtype=float)
    if s.shape != w.shape:
        raise ValueError(f"support has shape {s.shape} but weight has shape {w.shape}")
    if not R >= 1.0:
        raise ValueError(f"R must be >= 1, got {R}")
    n = int(s.size)
    i, j = np.triu_indices(n, k=1)
    ds = s[i] - s[j]
    dz = s[i] * w[i] - s[j] * w[j]
    comparable = ds != 0
    hi = np.maximum(s[i], s[j])
    lo = np.minimum(s[i], s[j])
    locked = comparable & (hi > R * lo)
    reversed_ = comparable & (np.sign(dz) == -np.sign(ds))
    nc = int(comparable.sum())
    unlocked = comparable & ~locked
    return {
        "pairs": int(i.size),
        "comparable_pairs": nc,
        "locked_frac": float(locked.sum() / nc) if nc else float("nan"),
        "reversed_frac": float(reversed_.sum() / nc) if nc else float("nan"),
        "reversed_among_unlocked": float((reversed_ & unlocked).sum() / unlocked.sum())
        if unlocked.any() else float("nan"),
        "violations": int((reversed_ & locked).sum()),
    }
