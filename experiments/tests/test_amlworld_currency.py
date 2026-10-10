"""Currency conversion in the AMLworld loader.

Builds a small *_Trans.csv with known exchange rates (Euro 0.8 and Yen 100 units per US
Dollar) and checks that every amount is converted to US Dollar, that the raw mode leaves
amounts untouched, and that the loader refuses to convert when a rate cannot be estimated
or the rate rows disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inchusm_aml.data.amlworld import _decimals, load_transactions  # noqa: E402

HEADER = ("Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
          "Amount Paid,Payment Currency,Payment Format,Is Laundering\n")
EUR, YEN, BTC, USD = "Euro", "Yen", "Bitcoin", "US Dollar"


def _row(src: str, dst: str, recv: float, rc: str, paid: float, pc: str, laund: int = 0) -> str:
    return f"2022/09/01 00:00,001,{src},002,{dst},{recv},{rc},{paid},{pc},ACH,{laund}\n"


def _rate_rows(n_eur: int = 25, n_yen: int = 25, eur_rate: float = 0.8) -> list[str]:
    # amounts are far above the rounding step so rounding does not disturb the ratio
    rows = [_row(f"E{i}", f"F{i}", round(10000.0 * eur_rate, 2), EUR, 10000.0, USD) for i in range(n_eur)]
    rows += [_row(f"Y{i}", f"Z{i}", 500.0, USD, 50000.0, YEN) for i in range(n_yen)]
    return rows


def _write(tmp_path: Path, rows: list[str]) -> Path:
    p = tmp_path / "T-Small_Trans.csv"
    p.write_text(HEADER + "".join(rows), encoding="utf-8")
    return p


def _payment(df, code, src: str) -> float:
    return float(df.loc[df["src"] == code[f"001_{src}"], "amount"].iat[0])


def test_amounts_are_converted_to_us_dollar(tmp_path):
    rows = _rate_rows() + [
        _row("A", "B", 80.0, EUR, 80.0, EUR),        # 80 Euro    = 100 USD
        _row("C", "D", 10000.0, YEN, 10000.0, YEN),  # 10,000 Yen = 100 USD
        _row("G", "H", 100.0, USD, 100.0, USD),      # 100 USD
    ]
    df, code = load_transactions(_write(tmp_path, rows))
    assert df.attrs["amount_unit"] == "US Dollar"
    assert df.attrs["usd_rates"][EUR] == pytest.approx(0.8)
    assert df.attrs["usd_rates"][YEN] == pytest.approx(100.0)
    for acct in ("A", "C", "G"):
        assert _payment(df, code, acct) == pytest.approx(100.0), acct
    # every rate row pays 10,000 USD (Euro rows) or 50,000 Yen = 500 USD (Yen rows)
    assert _payment(df, code, "E0") == pytest.approx(10000.0)
    assert _payment(df, code, "Y0") == pytest.approx(500.0)


def test_raw_mode_keeps_payment_currency(tmp_path):
    rows = _rate_rows() + [_row("A", "B", 80.0, EUR, 80.0, EUR),
                           _row("C", "D", 10000.0, YEN, 10000.0, YEN)]
    df, code = load_transactions(_write(tmp_path, rows), currency="raw")
    assert df.attrs["amount_unit"] == "payment currency (raw)"
    assert _payment(df, code, "A") == pytest.approx(80.0)
    assert _payment(df, code, "C") == pytest.approx(10000.0)
    names = df.attrs["currencies"]
    assert names[int(df.loc[df["src"] == code["001_C"], "currency"].iat[0])] == YEN


def test_refuses_currency_without_dollar_rate(tmp_path):
    rows = _rate_rows(n_yen=0) + [_row("C", "D", 10000.0, YEN, 10000.0, YEN)]
    with pytest.raises(ValueError, match="Yen.*0 rows"):
        load_transactions(_write(tmp_path, rows))


def test_refuses_disagreeing_rates(tmp_path):
    # 15 of 40 Euro rows at a 10% different rate: 62.5% agreement, below the 90% floor.
    # A row PAID in Euro is required, otherwise no Euro rate is needed and none is checked.
    rows = (_rate_rows(n_eur=25) + _rate_rows(n_eur=15, n_yen=0, eur_rate=0.88)
            + [_row("A", "B", 80.0, EUR, 80.0, EUR)])
    with pytest.raises(ValueError, match="Euro.*within"):
        load_transactions(_write(tmp_path, rows))


def test_rate_needed_only_for_payment_currencies(tmp_path):
    # Euro appears only as a receiving currency: amounts are paid amounts, so no Euro rate is
    # used and its disagreement must not block loading.
    rows = _rate_rows(n_eur=25) + _rate_rows(n_eur=15, n_yen=0, eur_rate=0.88)
    df, _ = load_transactions(_write(tmp_path, rows))
    assert EUR not in df.attrs["usd_rates"]


def test_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="currency must be"):
        load_transactions(_write(tmp_path, _rate_rows()), currency="eur")


def test_rounding_dominated_rows_are_excluded(tmp_path):
    # 30 large Bitcoin rows at the true rate, 40 rows of 0.000001 BTC for 0.01 USD whose ratio is
    # off by about 19% purely from rounding. Using all 70 rows would give 43% agreement and a
    # refusal; excluding rows below 100 rounding steps keeps the 30 exact rows.
    rate = 8.41624e-05
    rows = _rate_rows()
    rows += [_row(f"B{i}", f"C{i}", round(10000.0 * rate, 6), BTC, 10000.0, USD) for i in range(30)]
    rows += [_row(f"S{i}", f"T{i}", 0.000001, BTC, 0.01, USD) for i in range(40)]
    rows += [_row("P", "Q", 0.841624, BTC, 0.841624, BTC)]   # 0.841624 BTC = 10,000 USD
    df, code = load_transactions(_write(tmp_path, rows))
    assert df.attrs["usd_rates"][BTC] == pytest.approx(rate, rel=1e-6)
    assert df.attrs["rate_agreement"][BTC]["rows_used"] == 30
    assert _payment(df, code, "P") == pytest.approx(10000.0, rel=1e-6)


@pytest.mark.parametrize("values, expected", [
    ([0.07, 1.5, 3.0], 2), ([0.000001, 0.841624], 6), ([1.0, 2.0, 1e12], 0),
    ([1046302363293.48], 2),
])
def test_decimals(values, expected):
    import numpy as np
    assert _decimals(np.array(values)) == expected


def test_decimals_refuses_unknown_step():
    import numpy as np
    with pytest.raises(ValueError, match="rounding step unknown"):
        _decimals(np.array([1e-9]))
