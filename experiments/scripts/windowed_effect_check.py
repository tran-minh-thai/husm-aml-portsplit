"""Check windowed_effect.py with one window over the whole data set against the records of the full chain input.

Compares the pooled rows of windowed_effect.csv with conservation_effect.csv (open chains, chain_conservation_effect.py)
and cycle_effect.csv (closed chains, chain_cycle_effect.py), value by value: planted accounts, returned set sizes,
marked accounts or shares, recoveries and odds ratios. Exits non-zero on any difference or when nothing is compared.

Run: python experiments/scripts/windowed_effect_check.py windowed_effect.csv <open record dir> <closed record dir>
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

OPEN = (("planted_intermediates", "planted"), ("I_0.2", "I_0.2"), ("I_inf", "I_inf"), ("marked_0.2", "marked_0.2"),
        ("marked_inf", "marked_inf"), ("R_0.2", "R_0.2"), ("R_inf", "R_inf"), ("OR", "OR"))
CLOSED = (("planted_accounts", "planted"), ("I_0.2", "I_0.2"), ("I_inf", "I_inf"), ("P_0.2", "P_0.2"), ("P_inf", "P_inf"),
          ("R_0.2", "R_0.2"), ("R_inf", "R_inf"), ("OR", "OR"))


def same(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if not (pd.isna(a) or pd.isna(b)) and float(a) == float(b):     # equal values, infinite ones included
        return True
    if pd.isna(a) or pd.isna(b) or math.isinf(float(a)) or math.isinf(float(b)):
        return False                                                  # unequal, one of them missing or infinite
    return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(a)))


def main() -> None:
    w = pd.read_csv(sys.argv[1])
    if set(w["window"].astype(str)) != {"0", "pooled"}:
        sys.exit(f"{sys.argv[1]} is not a run with one window: windows {sorted(set(w['window'].astype(str)))}")
    w = w[w["window"].astype(str) == "pooled"]
    compared, bad = 0, []
    for kind, path, pairs in (("open", Path(sys.argv[2]) / "conservation_effect.csv", OPEN),
                              ("closed", Path(sys.argv[3]) / "cycle_effect.csv", CLOSED)):
        ref = pd.read_csv(path)
        for _, r in ref.iterrows():
            x = w[(w["kind"] == kind) & (w["length"] == r["length"])]
            if len(x) != 1:
                bad.append(f"{kind} length {r['length']}: {len(x)} pooled rows")
                continue
            for a, b in pairs:
                compared += 1
                if not same(r[a], x.iloc[0][b]):
                    bad.append(f"{kind} length {r['length']} {a}: record {r[a]}, windowed {x.iloc[0][b]}")
    print(f"values compared: {compared}; differences: {len(bad)}")
    for b in bad:
        print("  " + b)
    if compared == 0 or bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
