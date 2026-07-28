"""Load IBM AMLworld (Altman 2023)  -  transactions + ground-truth laundering attempts.

Format (Kaggle 2025-07 release):
- {HI,LI}-{Small,..}_Trans.csv: Timestamp, From Bank, Account, To Bank, Account,
  Amount Received, Receiving Currency, Amount Paid, Payment Currency,
  Payment Format, Is Laundering.
- _Patterns.txt: blocks "BEGIN LAUNDERING ATTEMPT - <TYPE>[: description]" ...
  "END LAUNDERING ATTEMPT"; each line in a block is a transaction in the same format.

Account key = "<bank>_<account>" (safe if account numbers collide across banks).
Time is reduced to MINUTES (int64) so every deltaT constraint is integer arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# AMLworld's 8 typologies; names normalized per Patterns.txt
KNOWN_TYPES = (
    "FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER",
    "CYCLE", "RANDOM", "BIPARTITE", "STACK",
)


def load_transactions(path: str | Path, chunksize: int = 4_000_000):
    """Read *_Trans.csv -> (edges DataFrame [src, dst, amount, time(min), laundering], code_map).

    MEMORY-EFFICIENT (2026-07-02): read in CHUNKS and encode the "bank_acct" account
    into int32 (instead of holding 31M-176M strings) -> Medium/Large run on modest RAM.
    src/dst = int32, time = int32 (minutes), laundering = int8. `code_map` (str->int32)
    is returned so `load_patterns` encodes the ground-truth accounts via the SAME table.

    This is a BIJECTIVE RELABELING: all recovery/precision results are unchanged from the
    string version (identity is preserved), so the Small figures stay the same.
    """
    code: dict[str, int] = {}
    src_p, dst_p, amt_p, time_p, laund_p = [], [], [], [], []
    reader = pd.read_csv(
        path,
        usecols=[0, 1, 2, 3, 4, 7, 10],
        names=["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amount", "laundering"],
        header=0,
        dtype={"from_bank": str, "from_acct": str, "to_bank": str, "to_acct": str},
        chunksize=chunksize,
    )
    for chunk in reader:
        fk = (chunk["from_bank"] + "_" + chunk["from_acct"]).to_numpy()
        tk = (chunk["to_bank"] + "_" + chunk["to_acct"]).to_numpy()
        src_p.append(_encode(fk, code))
        dst_p.append(_encode(tk, code))
        amt_p.append(chunk["amount"].to_numpy(dtype=np.float64))
        # cast to ns EXPLICITLY: pandas >=3.0 defaults to datetime64[us], so
        # .astype("int64") directly would yield microseconds -> time off by 1000x.
        t = (
            pd.to_datetime(chunk["ts"], format="%Y/%m/%d %H:%M")
            .astype("datetime64[ns]")
            .astype("int64")
            .to_numpy()
        )
        time_p.append((t // 60_000_000_000).astype(np.int32))
        laund_p.append(chunk["laundering"].to_numpy(dtype=np.int8))
    out = pd.DataFrame(
        {
            "src": np.concatenate(src_p),
            "dst": np.concatenate(dst_p),
            "amount": np.concatenate(amt_p),
            "time": np.concatenate(time_p),
            "laundering": np.concatenate(laund_p),
        }
    )
    return out, code


def _encode(keys: np.ndarray, code: dict[str, int]) -> np.ndarray:
    """Encode an array of 'bank_acct' strings -> int32 via the global code table (add on first sight)."""
    local, uniq = pd.factorize(keys)
    gids = np.empty(len(uniq), dtype=np.int32)
    for i, u in enumerate(uniq):
        g = code.get(u)
        if g is None:
            g = len(code)
            code[u] = g
        gids[i] = g
    return gids[local].astype(np.int32)


@dataclass
class Attempt:
    """One laundering attempt from Patterns.txt (ground truth)."""

    type: str
    srcs: list[str] = field(default_factory=list)
    dsts: list[str] = field(default_factory=list)

    @property
    def accounts(self) -> set[str]:
        return set(self.srcs) | set(self.dsts)

    def fan_out_center(self) -> str:
        return pd.Series(self.srcs).mode().iat[0]

    def fan_in_center(self) -> str:
        return pd.Series(self.dsts).mode().iat[0]

    def gather_scatter_candidates(self) -> set:
        """Every account that is both a source and a destination in the attempt (center candidates).
        Returns a SET (not the mode) => recovery is independent of ID encoding and does not depend on
        tie-breaking; an attempt counts as recovered if ANY candidate center is captured."""
        return set(self.srcs) & set(self.dsts)

    def scatter_gather_candidates(self) -> tuple[set, set]:
        """(pure-source set, pure-destination set) of smurf type-2  -  used for encoding-independent
        pair matching (recovered if a pair (u,w) exists in the mined set)."""
        return set(self.srcs) - set(self.dsts), set(self.dsts) - set(self.srcs)


def load_patterns(path: str | Path, code: dict[str, int] | None = None) -> list[Attempt]:
    """Read Patterns.txt. If `code` is given (from load_transactions), encode the
    ground-truth accounts into int32 via the SAME table => identity matches the encoded edges.
    Accounts absent from the transactions -> code -1 (matches nothing)."""

    def enc(key: str):
        return key if code is None else code.get(key, -1)

    attempts: list[Attempt] = []
    current: Attempt | None = None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            head = line.split("-", 1)[1].strip()          # "FAN-OUT:  Max 16-degree ..."
            type_name = head.split(":")[0].strip().upper()
            current = Attempt(type=type_name)
        elif line.startswith("END LAUNDERING ATTEMPT"):
            if current is not None:
                attempts.append(current)
            current = None
        elif current is not None and line:
            tok = line.split(",")
            if len(tok) >= 5:
                current.srcs.append(enc(f"{tok[1]}_{tok[2]}"))
                current.dsts.append(enc(f"{tok[3]}_{tok[4]}"))
    return attempts
