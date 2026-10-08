"""The synthetic wholesale book: positions with the sensitivities a stress test needs.

Valuation conventions, stated so a risk reviewer can challenge them:
- Loans sit in the banking book at amortised cost; stress shows up as expected credit loss.
  EAD = drawn + CCF x undrawn (CCF 75%, foundation-IRB style). LGD 25% secured, 40% senior
  unsecured, 75% subordinated.
- Bonds and derivatives are marked to market; all MTM moves hit CET1 (as with AOCI included).
- Amounts are USD equivalents; INR positions also carry USD/INR translation risk.
- The as-of yield curve and rating spreads are an illustrative assumption, not market data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

from seismo.module_b import blotter as bl
from seismo.module_b.credit import NOTCH, PD_1Y, ecl, risk_weight, stage
from seismo.module_b.factors import SECTOR_FACTOR
from seismo.module_b.pricing import bond_risk, bs_greeks, cds_cs01, swap_dv01

AS_OF = bl.AS_OF
CURVE = [(0.25, 0.040), (2.0, 0.036), (5.0, 0.037), (10.0, 0.041), (30.0, 0.046)]  # illustrative
RATING_SPREAD = {"AAA": 0.003, "AA": 0.005, "A": 0.008, "BBB": 0.013, "BB": 0.025, "B": 0.040, "CCC": 0.080}
CCF = 0.75
DERIV_ADDON = {"IRS": 0.005, "FX_FWD": 0.01, "CDS": 0.05, "EQ_OPTION": 0.06}
EQUITY_RW = 2.5


def base_rate(years: float) -> float:
    if years <= CURVE[0][0]:
        return CURVE[0][1]
    for (t0, r0), (t1, r1) in zip(CURVE, CURVE[1:], strict=False):
        if years <= t1:
            return r0 + (r1 - r0) * (years - t0) / (t1 - t0)
    return CURVE[-1][1]


def rating_spread(rating: str | None) -> float:
    if rating is None:
        return RATING_SPREAD["BBB"]
    letters = rating.rstrip("+-")
    return RATING_SPREAD.get(letters, RATING_SPREAD["CCC"])


def years_to(d: date) -> float:
    return max(0.05, (d - AS_OF).days / 365.25)


@dataclass
class Position:
    pos_id: str
    asset_class: str          # Loans | Bonds | Derivatives | Equities
    instrument: str
    obligor_id: str
    name: str
    sector: str
    country: str
    rating: str | None
    orig_rating: str | None
    currency: str
    years: float
    bank: bool = False
    d: dict = field(default_factory=dict)   # instrument-specific terms and sensitivities

    @property
    def factor(self) -> str:
        if self.country == "IN":
            return "EQ:IN_BANKS" if self.bank else "EQ:IN"
        if self.bank and self.obligor_id in bl.REGIONAL_BANKS:
            return "EQ:BANKS"
        return SECTOR_FACTOR.get(self.sector, "EQ:MKT")


@dataclass
class Book:
    positions: list[Position]
    obligors: dict[str, bl.Obligor]

    def by_class(self, asset_class: str) -> list[Position]:
        return [p for p in self.positions if p.asset_class == asset_class]


def _value(p: Position) -> float:
    """Carrying value before stress (loans gross of allowance)."""
    if p.asset_class == "Loans":
        return p.d["drawn"]
    if p.asset_class == "Bonds":
        return p.d["face"] * p.d["price"] / 100
    if p.asset_class == "Equities":
        return p.d["value"]
    return p.d.get("mtm", 0.0)


def build_book(root: Path) -> Book:
    obligors, trades = bl.generate(root)
    by_id = {o.obligor_id: o for o in obligors}
    positions: list[Position] = []
    for pid, agg in sorted(bl.aggregate(trades).items()):
        inst = agg["instrument"]
        ob = by_id.get(agg["counterparty"])
        years = years_to(agg["maturity"])
        if inst in ("TERM_LOAN", "REVOLVER"):
            drawn = max(0.0, agg["drawn"])
            limit = max(agg["limit"], drawn)
            undrawn = limit - drawn if inst == "REVOLVER" else 0.0
            positions.append(Position(
                pid, "Loans", inst, ob.obligor_id, ob.name, ob.sector, ob.country, ob.rating, ob.orig_rating,
                agg["currency"], years, ob.bank,
                {"drawn": drawn, "undrawn": undrawn, "ead": drawn + CCF * undrawn,
                 "lgd": bl.COLLATERAL_LGD.get(agg["collateral"], 0.40), "collateral": agg["collateral"],
                 "rate": agg["rate"]},
            ))
        elif inst == "BOND":
            face = agg["face"]
            if face <= 0:
                continue
            sovereign = agg["counterparty"] in ("UST", "GSEC")
            if sovereign:
                name = "US Treasury" if agg["counterparty"] == "UST" else "Government of India G-sec"
                rating, sector, country = ("AA+", "Sovereign", "US") if agg["counterparty"] == "UST" else ("BBB-", "Sovereign", "IN")
                ytm = base_rate(years) if country == "US" else agg["rate"]
                obligor_id, bank, orig = agg["counterparty"], False, rating
            else:
                name, rating, sector, country = ob.name, ob.rating, ob.sector, ob.country
                ytm = base_rate(years) + rating_spread(rating)
                obligor_id, bank, orig = ob.obligor_id, ob.bank, ob.orig_rating
            price, mod, conv = bond_risk(agg["rate"], years, ytm)
            positions.append(Position(
                pid, "Bonds", "SOVEREIGN" if sovereign else "CORPORATE", obligor_id, name, sector, country,
                rating, orig, agg["currency"], years, bank,
                {"face": face, "coupon": agg["rate"], "ytm": ytm, "price": price, "mod_dur": mod,
                 "convexity": conv, "sovereign": sovereign, "lgd": 0.40},
            ))
        elif inst == "IRS":
            receive = agg["direction"] == "RECEIVE_FIXED"
            positions.append(Position(
                pid, "Derivatives", "IRS", ob.obligor_id, f"IRS {'receive' if receive else 'pay'} fixed {years:.0f}y",
                ob.sector, "US", None, None, "USD", years, False,
                {"notional": agg["notional"], "dv01": swap_dv01(agg["notional"], years, agg["rate"], receive),
                 "receive_fixed": receive, "mtm": 0.0},
            ))
        elif inst == "FX_FWD":
            pair = agg["extra"].get("pair", "USDINR")
            long_usd = agg["direction"] == "BUY_USD"
            # Long USD/INR gains when USDINR rises; long USD vs EUR gains when EURUSD falls.
            sign = (1 if long_usd else -1) * (1 if pair == "USDINR" else -1)
            positions.append(Position(
                pid, "Derivatives", "FX_FWD", ob.obligor_id, f"FX forward {pair} {'buy' if long_usd else 'sell'} USD",
                ob.sector, "US", None, None, "USD", years, False,
                {"notional": agg["notional"], "pair": pair, "sign": sign, "mtm": 0.0},
            ))
        elif inst == "CDS":
            ref = by_id[agg["extra"]["reference"]]
            bought = agg["direction"] == "BUY_PROTECTION"
            positions.append(Position(
                pid, "Derivatives", "CDS", ref.obligor_id, f"CDS on {ref.name} ({'bought' if bought else 'sold'} protection)",
                ref.sector, ref.country, ref.rating, ref.orig_rating, "USD", years, ref.bank,
                {"notional": agg["notional"], "spread": agg["rate"], "bought": bought,
                 "cs01": cds_cs01(agg["notional"], years, agg["rate"], 0.04, bought), "mtm": 0.0, "lgd": 0.6},
            ))
        elif inst == "EQ_OPTION":
            und = agg["extra"]["underlying"]
            call = agg["extra"]["type"] == "CALL"
            long = agg["direction"] == "LONG"
            strike = 100 * agg["rate"]
            g = bs_greeks(100.0, strike, years, 0.20, 0.04, call)
            units = agg["notional"] / 100.0 * (1 if long else -1)
            positions.append(Position(
                pid, "Derivatives", "EQ_OPTION", "DEALER",
                f"{'Long' if long else 'Short'} {und} {'call' if call else 'put'} K={agg['rate']:.0%}",
                "Index", "US", None, None, "USD", years, False,
                {"notional": agg["notional"], "underlying": und, "units": units, "call": call,
                 "strike": strike, "delta": g["delta"], "gamma": g["gamma"], "vega": g["vega"],
                 "mtm": units * g["price"]},
            ))
        elif inst == "EQUITY":
            positions.append(Position(
                pid, "Equities", "EQUITY", ob.obligor_id, ob.name, ob.sector, ob.country, ob.rating, ob.orig_rating,
                agg["currency"], 0.0, ob.bank, {"value": agg["notional"], "beta": 1.0},
            ))
    return Book(positions, by_id)


@lru_cache(maxsize=2)
def load_book(root: str) -> Book:
    return build_book(Path(root))


def book_summary(book: Book) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in book.positions:
        out[p.asset_class] = out.get(p.asset_class, 0.0) + _value(p)
    return out


def baseline_ecl(p: Position) -> tuple[int, float]:
    st = stage(p.orig_rating or p.rating, p.rating)
    return st, ecl(st, PD_1Y[p.rating], p.d["lgd"], p.d["ead"], p.years)


def rwa_position(p: Position, rating: str | None = None, defaulted: bool = False, value: float | None = None) -> float:
    rating = rating or p.rating
    if p.asset_class == "Loans":
        rw = 1.5 if defaulted else risk_weight(rating)
        return p.d["ead"] * rw
    if p.asset_class == "Bonds":
        v = _value(p) if value is None else value
        return v * (1.5 if defaulted else risk_weight(rating, sovereign=p.d["sovereign"]))
    if p.asset_class == "Equities":
        v = p.d["value"] if value is None else value
        return max(0.0, v) * EQUITY_RW
    return p.d["notional"] * DERIV_ADDON.get(p.instrument, 0.02)


__all__ = ["Book", "Position", "build_book", "load_book", "book_summary", "baseline_ecl",
           "rwa_position", "NOTCH", "_value"]
