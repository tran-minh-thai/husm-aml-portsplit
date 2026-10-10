"""Load IBM AMLworld (Altman 2023)  -  transactions + ground-truth laundering attempts.

Format (Kaggle 2025-07 release):
- {HI,LI}-{Small,..}_Trans.csv: Timestamp, From Bank, Account, To Bank, Account,
  Amount Received, Receiving Currency, Amount Paid, Payment Currency,
  Payment Format, Is Laundering.
- _Patterns.txt: blocks "BEGIN LAUNDERING ATTEMPT - <TYPE>[: description]" ...
  "END LAUNDERING ATTEMPT"; each line in a block is a transaction in the same format.

Account key = "<bank>_<account>" (safe if account numbers collide across banks).
Time is reduced to MINUTES (int64) so every deltaT constraint is integer arithmetic.

Currency. Amounts are denominated in 15 currencies whose typical transaction sizes differ by
about six orders of magnitude (median paid amount: 0.07 in Bitcoin, about 900 in US Dollar,
about 95,000 in Yen), so comparing two raw amounts in different currencies is meaningless.
US Dollar and Euro carry most transactions, so per-account totals are affected less than
single transactions: on the HI-Small truth neighborhood, 26.5% of accounts change amount
quartile after conversion (7.2% on LI-Small).

`load_transactions` converts every amount to US Dollar by default, using exchange rates
estimated from the file itself: each cross-currency row reports the same
transfer in both currencies, so received/paid on rows where one side is US Dollar gives that
currency's rate. Amounts are rounded (0.000001 for Bitcoin, 0.01 otherwise), and on a row of a
few cents the rounding alone moves the ratio by up to about 19%, so only rows where both sides
are at least RATE_MIN_STEPS rounding steps are used (rounding error <= 0.5%). On those rows the
generator's rates are fixed: on HI-Small and LI-Small every such row of every currency lies
within 0.73% of its currency's median, and the two files give the same rates.
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


USD = "US Dollar"
# A row pairing a currency with US Dollar is used for its rate only if both amounts are at least
# RATE_MIN_STEPS rounding steps (step = 10**-d, d = the fewest decimals that represent every
# amount recorded in that currency; see _decimals).
# The estimate is accepted only if at least RATE_MIN_ROWS such rows exist and at least
# RATE_MIN_AGREE of them lie within RATE_TOL (relative) of the median.
RATE_MIN_STEPS = 100
RATE_MIN_ROWS = 20
RATE_MIN_AGREE = 0.90
RATE_TOL = 0.02


def load_transactions(path: str | Path, chunksize: int = 4_000_000, currency: str = "usd"):
    """Read *_Trans.csv -> (edges DataFrame [src, dst, amount, time(min), laundering, currency], code_map).

    MEMORY-EFFICIENT (2026-07-02): read in CHUNKS and encode the "bank_acct" account
    into int32 (instead of holding 31M-176M strings) -> Medium/Large run on modest RAM.
    src/dst = int32, time = int32 (minutes), laundering = int8. `code_map` (str->int32)
    is returned so `load_patterns` encodes the ground-truth accounts via the SAME table.

    `amount` is the paid amount. With currency="usd" (default) it is converted to US Dollar
    (see the module docstring); with currency="raw" it is left in the payment currency, which
    reproduces the behavior before the conversion was added and exists only to measure what
    the conversion changes. `currency` (int8) indexes `out.attrs["currencies"]`; the estimated
    rates (units per US Dollar) and their agreement are in `out.attrs["usd_rates"]` and
    `out.attrs["rate_agreement"]`.
    """
    if currency not in ("usd", "raw"):
        raise ValueError(f"currency must be 'usd' or 'raw', got {currency!r}")
    code: dict[str, int] = {}
    cur_code: dict[str, int] = {}
    src_p, dst_p, amt_p, cur_p, time_p, laund_p = [], [], [], [], [], []
    usd_pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    steps: dict[str, float] = {}
    reader = pd.read_csv(
        path,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
        names=["ts", "from_bank", "from_acct", "to_bank", "to_acct",
               "received", "recv_cur", "amount", "pay_cur", "laundering"],
        header=0,
        dtype={"from_bank": str, "from_acct": str, "to_bank": str, "to_acct": str,
               "recv_cur": str, "pay_cur": str},
        chunksize=chunksize,
    )
    for chunk in reader:
        fk = (chunk["from_bank"] + "_" + chunk["from_acct"]).to_numpy()
        tk = (chunk["to_bank"] + "_" + chunk["to_acct"]).to_numpy()
        src_p.append(_encode(fk, code))
        dst_p.append(_encode(tk, code))
        amt_p.append(chunk["amount"].to_numpy(dtype=np.float64))
        cur_p.append(_encode(chunk["pay_cur"].to_numpy(), cur_code).astype(np.int8))
        _collect_usd_pairs(chunk, usd_pairs, steps)
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
    currencies = [c for c, _ in sorted(cur_code.items(), key=lambda kv: kv[1])]
    cur = np.concatenate(cur_p)
    amount = np.concatenate(amt_p)
    rates, agreement = estimate_usd_rates(usd_pairs, steps, currencies, path)
    if currency == "usd":
        per_code = np.array([rates[c] for c in currencies], dtype=np.float64)
        amount = amount / per_code[cur]
    out = pd.DataFrame(
        {
            "src": np.concatenate(src_p),
            "dst": np.concatenate(dst_p),
            "amount": amount,
            "time": np.concatenate(time_p),
            "laundering": np.concatenate(laund_p),
            "currency": cur,
        }
    )
    out.attrs["currencies"] = currencies
    out.attrs["usd_rates"] = rates
    out.attrs["rate_agreement"] = agreement
    out.attrs["amount_unit"] = "US Dollar" if currency == "usd" else "payment currency (raw)"
    return out, code


def _collect_usd_pairs(chunk: pd.DataFrame, acc: dict[str, list[tuple[np.ndarray, np.ndarray]]],
                      steps: dict[str, float]) -> None:
    """Record (amount in currency c, amount in US Dollar) for every row pairing c with US Dollar,
    and update each currency's rounding step from the decimals its amounts use."""
    recv = chunk["received"].to_numpy(dtype=np.float64)
    paid = chunk["amount"].to_numpy(dtype=np.float64)
    rc = chunk["recv_cur"].to_numpy()
    pc = chunk["pay_cur"].to_numpy()
    for amt, cur in ((recv, rc), (paid, pc)):
        for c in np.unique(cur):
            step = 10.0 ** -_decimals(amt[cur == c])
            steps[str(c)] = min(steps.get(str(c), np.inf), step)
    ok = (recv > 0) & (paid > 0) & (rc != pc)
    # received in c, paid in US Dollar  /  received in US Dollar, paid in c
    for mask, cur_col, c_amt, usd_amt in ((ok & (pc == USD), rc, recv, paid),
                                          (ok & (rc == USD), pc, paid, recv)):
        if not mask.any():
            continue
        cc, ca, ua = cur_col[mask], c_amt[mask], usd_amt[mask]
        for c in np.unique(cc):
            sel = cc == c
            acc.setdefault(str(c), []).append((ca[sel], ua[sel]))


MAX_DECIMALS = 8


def _decimals(x: np.ndarray) -> int:
    """Fewest decimals d (0..MAX_DECIMALS) such that every amount in x is a multiple of 10**-d.

    Raises ValueError if no such d exists, since the rounding step is then unknown."""
    for d in range(MAX_DECIMALS + 1):
        scaled = x * 10.0 ** d
        # a parsed decimal times 10**d is within a few ulps of an integer; allow 8
        tol = 8 * np.finfo(np.float64).eps * np.maximum(np.abs(scaled), 1.0)
        if np.all(np.abs(scaled - np.rint(scaled)) <= tol):
            return d
    raise ValueError(f"amounts are not multiples of 10**-{MAX_DECIMALS}; rounding step unknown")


def estimate_usd_rates(usd_pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]],
                       steps: dict[str, float], currencies: list[str],
                       source: str | Path) -> tuple[dict[str, float], dict[str, dict]]:
    """Median rate (units per US Dollar) for every payment currency, with an agreement check.

    Only rows where both amounts are at least RATE_MIN_STEPS rounding steps are used. Raises
    ValueError naming the file, the currency and the observed values when a currency has too
    few such rows or they disagree, rather than converting with a guess.
    """
    rates: dict[str, float] = {}
    agreement: dict[str, dict] = {}
    usd_step = steps.get(USD, np.inf)
    for c in currencies:
        if c == USD:
            rates[c] = 1.0
            agreement[c] = {"rows": None, "rows_used": None, "within_tol": None, "max_dev": None}
            continue
        parts = usd_pairs.get(c, [])
        ca = np.concatenate([p[0] for p in parts]) if parts else np.empty(0)
        ua = np.concatenate([p[1] for p in parts]) if parts else np.empty(0)
        big = (ca >= RATE_MIN_STEPS * steps.get(c, np.inf)) & (ua >= RATE_MIN_STEPS * usd_step)
        n_all, n = int(len(ca)), int(big.sum())
        if n < RATE_MIN_ROWS:
            raise ValueError(
                f"{source}: currency {c!r} is paired with {USD!r} on {n_all} rows, {n} of them "
                f"with both amounts >= {RATE_MIN_STEPS} rounding steps (need >= {RATE_MIN_ROWS}); "
                f"its US Dollar rate cannot be estimated")
        lr = np.log(ca[big] / ua[big])
        med = float(np.median(lr))
        dev = np.abs(np.exp(lr - med) - 1.0)
        within = float(np.mean(dev <= RATE_TOL))
        if within < RATE_MIN_AGREE:
            raise ValueError(
                f"{source}: currency {c!r}: only {within:.1%} of its {n} usable US Dollar rows lie "
                f"within {RATE_TOL:.0%} of the median rate {np.exp(med):.6g} "
                f"(need >= {RATE_MIN_AGREE:.0%}); rates are not fixed for this currency")
        rates[c] = float(np.exp(med))
        agreement[c] = {"rows": n_all, "rows_used": n, "within_tol": within,
                        "max_dev": float(dev.max())}
    return rates, agreement


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


def load_pattern_transactions(path: str | Path, code: dict[str, int], rates: dict[str, float]) -> pd.DataFrame:
    """Every transaction of every laundering attempt in Patterns.txt, one row each.

    Columns: attempt (0-based, in file order), type, src, dst (encoded with `code`, -1 when the
    account is absent from the transactions), time (minutes, as load_transactions), amount (paid
    amount converted to US Dollar with `rates`, i.e. load_transactions(...)[0].attrs["usd_rates"]
    of the matching *_Trans.csv), currency (payment currency name). Raises ValueError on a payment
    currency without a rate and on a malformed transaction line inside an attempt.
    """
    rows = []
    attempt, typ = -1, None
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            attempt += 1
            typ = line.split("-", 1)[1].strip().split(":")[0].strip().upper()
        elif line.startswith("END LAUNDERING ATTEMPT"):
            typ = None
        elif typ is not None and line:
            tok = line.split(",")
            if len(tok) < 9:
                raise ValueError(f"{path}:{n}: expected at least 9 fields, got {len(tok)}: {line!r}")
            cur = tok[8].strip()
            if cur not in rates:
                raise ValueError(f"{path}:{n}: payment currency {cur!r} has no US Dollar rate")
            rows.append((attempt, typ, code.get(f"{tok[1]}_{tok[2]}", -1), code.get(f"{tok[3]}_{tok[4]}", -1),
                         tok[0].strip(), float(tok[7]) / rates[cur], cur))
    out = pd.DataFrame(rows, columns=["attempt", "type", "src", "dst", "ts", "amount", "currency"])
    t = (pd.to_datetime(out["ts"], format="%Y/%m/%d %H:%M").astype("datetime64[ns]").astype("int64").to_numpy())
    out["time"] = (t // 60_000_000_000).astype(np.int32)
    return out.drop(columns="ts")[["attempt", "type", "src", "dst", "time", "amount", "currency"]]
