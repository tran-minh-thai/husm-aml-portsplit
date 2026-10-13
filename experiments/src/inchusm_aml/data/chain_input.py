"""Build the chain miner's input (.tg) from AMLworld and AMLGentex, under one of two labelings.

Labeling S (structure only): one label for every account and one for every transaction.
Labeling A (attributes): the attributes that passed the label-choice criteria
  (eta squared of ln(amount) <= 0.10, single-valued for >= 99% of accounts, effective number <= 20):
  AMLworld   transaction label = payment format, account label shared (banks: thousands of values);
  AMLGentex  account label = bank, transaction label shared (one transaction type after the
             income source and spending sink are removed).
Payment currency is never a label: it encodes the scale of the amount.

Amounts: AMLworld paid amount in US Dollar (load_transactions), AMLGentex amount as logged.
Self-transactions are left out (no chain occurrence can use one). The laundering flag never enters
the graph; the accounts with at least one flagged transaction are written to a separate file, used
only to count them in the miner's output.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELINGS = ("S", "A")


def amlworld(path: Path, labeling: str):
    """(accounts DataFrame [id, label], transactions [src, dst, label, amount], laundering account ids, info)."""
    from inchusm_aml.data.amlworld import load_transactions
    tx, _ = load_transactions(path)
    fmt = pd.read_csv(path, usecols=[9], names=["format"], header=0, dtype={"format": str})["format"].to_numpy()
    if len(fmt) != len(tx):
        raise RuntimeError(f"{path.name}: {len(fmt)} payment formats for {len(tx)} transactions")
    flagged = tx["laundering"].to_numpy() == 1
    laundering = np.union1d(tx["src"].to_numpy()[flagged], tx["dst"].to_numpy()[flagged])
    keep = tx["src"].to_numpy() != tx["dst"].to_numpy()
    tx, fmt = tx[keep].reset_index(drop=True), fmt[keep]
    info = {"rows": int(len(keep)), "self_transactions_dropped": int((~keep).sum())}
    if labeling == "A":
        names = sorted(set(fmt))
        code = {n: i for i, n in enumerate(names)}
        elab = np.array([code[f] for f in fmt], dtype=np.int64)
        info["transaction_labels"] = {str(i): n for n, i in code.items()}
    else:
        elab = np.zeros(len(tx), dtype=np.int64)
    ids = np.union1d(tx["src"].to_numpy(), tx["dst"].to_numpy())
    accounts = pd.DataFrame({"id": ids, "label": 0})
    edges = pd.DataFrame({"src": tx["src"].to_numpy(), "dst": tx["dst"].to_numpy(), "label": elab,
                          "amount": tx["amount"].to_numpy(dtype=float)})
    return accounts, edges, np.intersect1d(laundering, ids), info


def amlgentex(path: Path, labeling: str):
    df = pd.read_csv(path, usecols=["nameOrig", "nameDest", "bankOrig", "bankDest", "amount", "isSAR"])
    special = df["nameOrig"].isin((-2, -1)) | df["nameDest"].isin((-2, -1))
    df = df[~special]
    info = {"rows": int(len(special)), "special_rows_dropped": int(special.sum())}
    flagged = df["isSAR"].to_numpy() == 1
    laundering = np.union1d(df["nameOrig"].to_numpy()[flagged], df["nameDest"].to_numpy()[flagged])
    keep = df["nameOrig"].to_numpy() != df["nameDest"].to_numpy()
    info["self_transactions_dropped"] = int((~keep).sum())
    df = df[keep]
    ids = np.union1d(df["nameOrig"].to_numpy(), df["nameDest"].to_numpy())
    if labeling == "A":
        pairs = pd.concat([pd.DataFrame({"id": df["nameOrig"].to_numpy(), "bank": df["bankOrig"].astype(str).to_numpy()}),
                           pd.DataFrame({"id": df["nameDest"].to_numpy(), "bank": df["bankDest"].astype(str).to_numpy()})])
        counts = pairs.groupby(["id", "bank"]).size().reset_index(name="n")
        info["accounts_with_more_than_one_bank"] = int((counts.groupby("id").size() > 1).sum())
        # the bank seen most often for the account; ties go to the smallest bank name
        mode = counts.sort_values(["id", "n", "bank"], ascending=[True, False, True]).drop_duplicates("id")
        names = sorted(mode["bank"].unique())
        code = {n: i for i, n in enumerate(names)}
        label = mode.set_index("id")["bank"].map(code).reindex(ids).to_numpy()
        info["account_labels"] = {str(i): n for n, i in code.items()}
    else:
        label = np.zeros(len(ids), dtype=np.int64)
    accounts = pd.DataFrame({"id": ids, "label": label})
    edges = pd.DataFrame({"src": df["nameOrig"].to_numpy(), "dst": df["nameDest"].to_numpy(), "label": 0,
                          "amount": df["amount"].to_numpy(dtype=float)})
    return accounts, edges, np.intersect1d(laundering, ids), info


def write_tg(accounts: pd.DataFrame, edges: pd.DataFrame, path: Path) -> None:
    """Accounts first, then transactions; amounts with 17 significant digits (exact for a double).

    Lines end in "\n" on every system: pandas otherwise ends them with os.linesep, "\r\n" on Windows, and the
    same input would then hash differently on the two machines.
    """
    if (edges["amount"] <= 0).any() or not np.isfinite(edges["amount"]).all():
        raise ValueError(f"{path.name}: non-positive or non-finite amounts")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# directed transaction graph: v ID LABEL / e SOURCE TARGET LABEL AMOUNT\n")
        accounts.assign(tag="v")[["tag", "id", "label"]].to_csv(f, sep=" ", header=False, index=False,
                                                                lineterminator="\n")
        edges.assign(tag="e")[["tag", "src", "dst", "label", "amount"]].to_csv(
            f, sep=" ", header=False, index=False, float_format="%.17g", lineterminator="\n")


def write_ids(ids, path: Path) -> None:
    """One integer per line, "\n" line ends on every system (numpy.savetxt writes in text mode: "\r\n" on Windows)."""
    path.write_bytes("".join(f"{int(v)}\n" for v in ids).encode("ascii"))


def build(source: str, dataset: str, labeling: str, root: Path, out_dir: Path) -> dict:
    if labeling not in LABELINGS:
        raise ValueError(f"labeling must be one of {LABELINGS}")
    if source == "amlworld":
        accounts, edges, laundering, info = amlworld(root / "datasets" / "amlworld" / f"{dataset}_Trans.csv", labeling)
    else:
        accounts, edges, laundering, info = amlgentex(root / "datasets" / "amlgentex" / dataset / "tx_log.csv", labeling)
    out_dir.mkdir(parents=True, exist_ok=True)
    tg = out_dir / f"{dataset}_{labeling}.tg"
    write_tg(accounts, edges, tg)
    marked = out_dir / f"{dataset}_laundering_accounts.txt"
    write_ids(laundering, marked)
    import hashlib
    digest = hashlib.sha256()
    with open(tg, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    info["graph_sha256"] = digest.hexdigest()
    info.update({"source": source, "dataset": dataset, "labeling": labeling, "accounts": int(len(accounts)),
                 "transactions": int(len(edges)), "laundering_accounts": int(len(laundering)),
                 "account_label_values": int(accounts["label"].nunique()),
                 "transaction_label_values": int(edges["label"].nunique()),
                 "graph_file": tg.name, "laundering_file": marked.name})
    (out_dir / f"{dataset}_{labeling}.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info
