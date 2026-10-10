"""Measure the candidate label attributes of one data set against the label-choice criteria.

For every candidate account attribute and transaction attribute: number of values, effective
number exp(H), eta squared of ln(amount) on the attribute, for account attributes the share of
accounts that carry exactly one value, and (reported only) how much the attribute says about the
laundering flag. Self-transactions are dropped first, as the miner drops them. Writes
label_roles.csv and prints which attributes pass K1 (eta squared <= 0.10), K2 (>= 99% of accounts
single-valued; account attributes only) and K3 (effective number <= 20).

Run: python experiments/probes/label_roles.py --source amlworld --dataset HI-Small
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent))

from round2_measurements import OUT  # noqa: E402

K1_MAX_ETA2, K2_MIN_SINGLE, K3_MAX_EFFECTIVE = 0.10, 0.99, 20.0
LIFT_MIN_ROWS = 100


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def eta2(values: pd.Series, groups: pd.Series) -> float:
    y = np.log(values.to_numpy(dtype=float))
    g = pd.Series(y).groupby(groups.to_numpy())
    total = float(((y - y.mean()) ** 2).sum())
    between = float((g.count() * (g.mean() - y.mean()) ** 2).sum())
    return between / total if total > 0 else 0.0


def leakage(labels: pd.Series, flag: pd.Series) -> tuple[float, float, float]:
    """(I(label; flag) / H(flag), smallest and largest lift over values with >= LIFT_MIN_ROWS rows)."""
    tab = pd.crosstab(labels.to_numpy(), flag.to_numpy())
    joint = tab.to_numpy(dtype=float)
    pxy = joint / joint.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())
    hy = entropy(joint.sum(axis=0))
    rows = joint.sum(axis=1)
    pos = joint[:, list(tab.columns).index(1)] if 1 in tab.columns else np.zeros(len(rows))
    overall = pos.sum() / rows.sum()
    keep = rows >= LIFT_MIN_ROWS
    lift = (pos[keep] / rows[keep]) / overall if overall > 0 and keep.any() else np.array([np.nan])
    return (mi / hy if hy > 0 else 0.0), float(np.nanmin(lift)), float(np.nanmax(lift))


def load(source: str, dataset: str):
    """(transactions with src, dst, amount, laundering and the candidate columns, attribute specs)."""
    if source == "amlworld":
        from inchusm_aml.data.amlworld import load_transactions
        path = ROOT / "datasets" / "amlworld" / f"{dataset}_Trans.csv"
        tx, code = load_transactions(path)
        raw = pd.read_csv(path, usecols=[1, 9], names=["from_bank", "format"], header=0,
                          dtype={"from_bank": str, "format": str})
        if len(raw) != len(tx):
            raise RuntimeError(f"row counts differ: {len(raw)} raw rows, {len(tx)} loaded")
        tx["format"] = raw["format"].to_numpy()
        bank_of = np.empty(len(code), dtype=object)
        for key, c in code.items():
            bank_of[c] = key.split("_", 1)[0]
        tx["bank_src"] = bank_of[tx["src"].to_numpy()]
        tx["bank_dst"] = bank_of[tx["dst"].to_numpy()]
        return tx, {"transaction": ["format"], "account": ["bank"]}
    path = ROOT / "datasets" / "amlgentex" / dataset / "tx_log.csv"
    df = pd.read_csv(path, usecols=["type", "amount", "nameOrig", "nameDest", "bankOrig", "bankDest", "isSAR"])
    df = df[~df["nameOrig"].isin((-2, -1)) & ~df["nameDest"].isin((-2, -1))]
    tx = pd.DataFrame({"src": df["nameOrig"].to_numpy(), "dst": df["nameDest"].to_numpy(),
                       "amount": df["amount"].to_numpy(dtype=float), "laundering": df["isSAR"].to_numpy(),
                       "type": df["type"].astype(str).to_numpy(), "bank_src": df["bankOrig"].astype(str).to_numpy(),
                       "bank_dst": df["bankDest"].astype(str).to_numpy()})
    return tx, {"transaction": ["type"], "account": ["bank"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["amlworld", "amlgentex"], required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    tx, attrs = load(args.source, args.dataset)
    n_rows = len(tx)
    tx = tx[tx["src"] != tx["dst"]]
    nonpos = int((tx["amount"] <= 0).sum())
    tx = tx[tx["amount"] > 0].reset_index(drop=True)
    print(f"{args.source} {args.dataset}: {n_rows:,} transactions, {n_rows - len(tx) - nonpos:,} self-transactions "
          f"and {nonpos:,} non-positive amounts dropped, {len(tx):,} kept")

    rows = []
    for a in attrs["transaction"]:
        counts = tx[a].value_counts().to_numpy()
        nmi, lo, hi = leakage(tx[a], tx["laundering"])
        rows.append({"attribute": a, "level": "transaction", "values": len(counts),
                     "effective_number": float(np.exp(entropy(counts))), "eta2_log_amount": eta2(tx["amount"], tx[a]),
                     "single_valued_share": np.nan, "nmi_laundering": nmi, "lift_min": lo, "lift_max": hi})
    for a in attrs["account"]:
        pairs = pd.concat([pd.DataFrame({"acct": tx["src"], "v": tx[f"{a}_src"]}),
                           pd.DataFrame({"acct": tx["dst"], "v": tx[f"{a}_dst"]})]).drop_duplicates()
        per_acct = pairs.groupby("acct")["v"].nunique()
        single = float((per_acct == 1).mean())
        first = pairs.drop_duplicates("acct").set_index("acct")["v"]
        counts = first.value_counts().to_numpy()
        flagged = pd.Series(0, index=first.index)
        laund = tx[tx["laundering"] == 1]
        flagged[flagged.index.isin(set(laund["src"]) | set(laund["dst"]))] = 1
        nmi, lo, hi = leakage(first, flagged)
        rows.append({"attribute": a, "level": "account", "values": len(counts),
                     "effective_number": float(np.exp(entropy(counts))),
                     "eta2_log_amount": eta2(tx["amount"], tx[f"{a}_src"]), "single_valued_share": single,
                     "nmi_laundering": nmi, "lift_min": lo, "lift_max": hi})
    out = pd.DataFrame(rows)
    out["K1"] = out["eta2_log_amount"] <= K1_MAX_ETA2
    out["K2"] = (out["level"] == "transaction") | (out["single_valued_share"] >= K2_MIN_SINGLE)
    out["K3"] = out["effective_number"] <= K3_MAX_EFFECTIVE
    out["passes"] = out["K1"] & out["K2"] & out["K3"]
    out.to_csv(OUT / "label_roles.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"(lift over values with >= {LIFT_MIN_ROWS} rows; nmi = I(attribute; laundering flag) / H(flag))")
    for level in ("account", "transaction"):
        ok = out[(out["level"] == level) & out["passes"]]["attribute"].tolist()
        print(f"labeling A, {level} label: {', '.join(ok) if ok else 'none passes -> one shared label'}")


if __name__ == "__main__":
    main()
