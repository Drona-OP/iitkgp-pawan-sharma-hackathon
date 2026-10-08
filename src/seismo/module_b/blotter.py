"""A seeded synthetic trade blotter for a wholesale bank, aggregated into positions.

The brief asks Module B to "use the provided sample transaction data". The suggested public
datasets are retail (card and key-worker banking transactions), so Seismo states that mismatch
and builds the wholesale equivalent: a trade blotter (trade_id, trade_date, counterparty,
instrument, direction, notional, rate, maturity, collateral) that aggregates into positions the
way a bank's books do. Every obligor relationship and exposure is invented; public company names
appear only as obligors, and internal ratings are synthetic, not agency ratings.

The India desk (16 Nifty 50 names from data/universe_in.csv) is generated after the core book
with its own seed, so adding it leaves every earlier trade unchanged. Its Adani group lines
(Adani Enterprises and Adani Ports: loans, a dollar bond each, small equity stakes) are sized
by hand, about 4% of the book, so a governance shock to one group is material but survivable,
as India's large-exposure rules intend. Group membership drives contagion in the stress test.

    python -m seismo.module_b.blotter     # writes data/blotter/trade_blotter.csv
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from seismo.module_b.credit import NOTCH, SCALE

AS_OF = date(2026, 9, 30)
SEED = 2026
SEED_INDIA = 2027
GROUPS = {"ADANIENT.NS": "Adani", "ADANIPORTS.NS": "Adani"}
# Hand-sized Adani lines in USD millions: (term loan, revolver limit, USD bond face, equity stake)
ADANI_LINES = {"ADANIENT.NS": (420, 300, 150, 45), "ADANIPORTS.NS": (380, 200, 210, 35)}

# Obligors that are not in the index universe, all synthetic.
MIDCAP_NAMES_US = [
    "Meridian Logistics", "Bluewater Chemicals", "Granite Peak Homebuilders", "Northstar Freight",
    "Cobalt Ridge Mining", "Harborview Hospitals", "Silverline Software", "Prairie Grain Co",
    "Redwood Paper Mills", "Summit Auto Parts", "Ironclad Steelworks", "Crescent Hotels",
    "Lakeshore Utilities", "Pinnacle Retail Group", "Atlas Aviation Leasing", "Vantage Media",
    "Keystone Rail Services", "Evergreen Foods", "Beacon Semiconductor", "Tidewater Shipping",
]
MIDCAP_NAMES_IN = [
    "Deccan Infra Projects", "Ganga Textiles", "Konkan Power", "Sahyadri Cement",
    "Narmada Chemicals", "Malabar Spices Exports", "Vindhya Steel", "Coromandel Agro",
    "Aravalli Auto Components", "Kaveri Pharma", "Himalaya Hydro", "Thar Solar",
    "Brahmaputra Logistics", "Nilgiri Tea Estates", "Godavari Fertilisers",
]
SECTORS = ["Industrials", "Materials", "Consumer Discretionary", "Consumer Staples", "Energy",
           "Health Care", "Information Technology", "Utilities", "Real Estate", "Communication Services"]
REGIONAL_BANKS = {"SIVB", "FRC", "SBNY", "HNB"}

COLLATERAL_LGD = {"SECURED": 0.25, "SENIOR_UNSECURED": 0.40, "SUBORDINATED": 0.75}


@dataclass
class Obligor:
    obligor_id: str
    name: str
    sector: str
    country: str
    rating: str
    orig_rating: str
    listed: bool
    bank: bool = False
    in_index: bool = False
    group: str = ""


@dataclass
class Trade:
    trade_id: str
    trade_date: date
    counterparty: str
    instrument: str
    direction: str
    notional: float
    rate: float
    maturity: date
    collateral: str
    currency: str = "USD"
    facility_id: str = ""
    extra: dict = field(default_factory=dict)


def _rating_between(rng: random.Random, a: str, b: str) -> str:
    lo, hi = sorted((NOTCH[a], NOTCH[b]))
    return SCALE[rng.randint(lo, hi)]


def build_obligors(universe_rows: list[dict[str, str]], watch_rows: list[dict[str, str]], rng: random.Random) -> list[Obligor]:
    out: list[Obligor] = []
    for r in universe_rows:
        fin = r["sector"] == "Financials"
        rating = _rating_between(rng, "A-", "A+") if fin else _rating_between(rng, "BBB+", "AA-")
        out.append(Obligor(r["ticker"], r["name"], r["sector"], "US", rating, rating, True, fin, True))
    for r in watch_rows:
        rating = "BBB" if r["ticker"] in REGIONAL_BANKS else "BBB+"
        out.append(Obligor(r["ticker"], r["name"], r["sector"], "US" if r["ticker"] != "CS" else "CH",
                           rating, rating, True, True, False))
    for i, name in enumerate(MIDCAP_NAMES_US + MIDCAP_NAMES_IN):
        india = i >= len(MIDCAP_NAMES_US)
        orig = _rating_between(rng, "BBB", "BB-")
        drift = rng.choice([0, 0, 0, 0, 1, 2])
        current = SCALE[min(NOTCH[orig] + drift, NOTCH["B"])]
        out.append(Obligor(f"SYN{i + 1:02d}", f"{name} (synthetic)", rng.choice(SECTORS),
                           "IN" if india else "US", current, orig, False))
    return out


def generate_trades(obligors: list[Obligor], rng: random.Random) -> list[Trade]:
    trades: list[Trade] = []
    n = 0

    def tid() -> str:
        nonlocal n
        n += 1
        return f"T{n:05d}"

    def past(days: int) -> date:
        return AS_OF - timedelta(days=rng.randint(30, days))

    def future(years_lo: float, years_hi: float) -> date:
        return AS_OF + timedelta(days=int(365 * rng.uniform(years_lo, years_hi)))

    # Loans: 1-3 facilities per obligor, with drawdowns and repayments.
    for ob in obligors:
        scale = 1.0 if ob.in_index else 0.45
        for f in range(rng.randint(1, 3) if ob.in_index or ob.bank else rng.randint(1, 2)):
            fac = f"F-{ob.obligor_id}-{f + 1}"
            kind = rng.choice(["TERM_LOAN", "REVOLVER", "TERM_LOAN"])
            limit = round(rng.uniform(150, 600) * scale, 1) * 1e6
            collateral = rng.choice(["SENIOR_UNSECURED", "SENIOR_UNSECURED", "SECURED", "SUBORDINATED"]) \
                if not ob.in_index else rng.choice(["SENIOR_UNSECURED", "SENIOR_UNSECURED", "SECURED"])
            maturity = future(1.0, 7.0)
            ccy = "INR" if ob.country == "IN" else "USD"
            rate = round(rng.uniform(0.045, 0.095), 4)
            trades.append(Trade(tid(), past(1500), ob.obligor_id, kind, "COMMIT", limit, rate, maturity,
                                collateral, ccy, fac))
            draws = rng.randint(1, 4)
            drawn_share = rng.uniform(0.35, 0.95) if kind == "REVOLVER" else 1.0
            for _ in range(draws):
                trades.append(Trade(tid(), past(1200), ob.obligor_id, kind, "DRAW",
                                    round(limit * drawn_share / draws, -3), rate, maturity, collateral, ccy, fac))
            if rng.random() < 0.4:
                trades.append(Trade(tid(), past(400), ob.obligor_id, kind, "REPAY",
                                    round(limit * drawn_share * rng.uniform(0.05, 0.25), -3), rate, maturity,
                                    collateral, ccy, fac))

    # Bonds: corporates for index names and banks, plus Treasuries and Indian G-secs.
    issuers = [o for o in obligors if o.in_index or o.bank] + rng.sample([o for o in obligors if o.obligor_id.startswith("SYN")], 8)
    for ob in issuers:
        for _ in range(rng.randint(1, 2)):
            mat = future(1.5, 15.0)
            coupon = round(rng.uniform(0.025, 0.065) + (0.02 if NOTCH[ob.rating] >= NOTCH["BB+"] else 0.0), 4)
            face = round(rng.uniform(40, 220), 0) * 1e6
            trades.append(Trade(tid(), past(1800), ob.obligor_id, "BOND", "BUY", face, coupon, mat,
                                "SENIOR_UNSECURED", "INR" if ob.country == "IN" else "USD", f"B-{ob.obligor_id}-{mat:%Y}"))
            if rng.random() < 0.3:
                trades.append(Trade(tid(), past(300), ob.obligor_id, "BOND", "SELL", round(face * 0.3, -3),
                                    coupon, mat, "SENIOR_UNSECURED", "INR" if ob.country == "IN" else "USD",
                                    f"B-{ob.obligor_id}-{mat:%Y}"))
    for years in (2, 3, 5, 7, 10, 20, 30):
        mat = AS_OF + timedelta(days=365 * years)
        trades.append(Trade(tid(), past(900), "UST", "BOND", "BUY", round(rng.uniform(300, 900), 0) * 1e6,
                            round(rng.uniform(0.035, 0.045), 4), mat, "SOVEREIGN", "USD", f"UST-{years}Y"))
    for years in (5, 10):
        mat = AS_OF + timedelta(days=365 * years)
        trades.append(Trade(tid(), past(900), "GSEC", "BOND", "BUY", round(rng.uniform(150, 400), 0) * 1e6,
                            round(rng.uniform(0.065, 0.072), 4), mat, "SOVEREIGN", "INR", f"GSEC-{years}Y"))

    # Derivatives.
    for i in range(16):
        years = rng.choice([2, 3, 5, 7, 10, 30])
        direction = rng.choice(["PAY_FIXED", "RECEIVE_FIXED", "RECEIVE_FIXED"])
        trades.append(Trade(tid(), past(1000), rng.choice(obligors).obligor_id, "IRS", direction,
                            round(rng.uniform(200, 1200), 0) * 1e6, round(rng.uniform(0.03, 0.045), 4),
                            AS_OF + timedelta(days=365 * years), "CSA", "USD", f"IRS-{i + 1:02d}"))
    for i in range(10):
        pair = rng.choice(["USDINR", "USDINR", "EURUSD"])
        trades.append(Trade(tid(), past(200), rng.choice(obligors).obligor_id, "FX_FWD",
                            rng.choice(["BUY_USD", "SELL_USD"]), round(rng.uniform(50, 400), 0) * 1e6,
                            0.0, future(0.2, 1.0), "CSA", pair, f"FX-{i + 1:02d}", {"pair": pair}))
    cds_refs = [o for o in obligors if o.bank or o.in_index]
    for i in range(10):
        ref = rng.choice(cds_refs)
        trades.append(Trade(tid(), past(700), "DEALER", "CDS", rng.choice(["BUY_PROTECTION", "SELL_PROTECTION", "SELL_PROTECTION"]),
                            round(rng.uniform(25, 150), 0) * 1e6, round(rng.uniform(0.006, 0.02), 4),
                            future(2, 5), "CSA", "USD", f"CDS-{i + 1:02d}", {"reference": ref.obligor_id}))
    for i in range(6):
        factor = rng.choice(["EQ:MKT", "EQ:MKT", "EQ:IT", "EQ:FIN"])
        call = rng.random() < 0.4
        trades.append(Trade(tid(), past(150), "DEALER", "EQ_OPTION", "LONG" if rng.random() < 0.6 else "SHORT",
                            round(rng.uniform(50, 250), 0) * 1e6, round(rng.uniform(0.9, 1.05), 3),
                            future(0.1, 1.0), "CSA", "USD", f"OPT-{i + 1:02d}",
                            {"underlying": factor, "type": "CALL" if call else "PUT"}))

    # Listed equity holdings (strategic stakes and hedges).
    holders = [o for o in obligors if o.listed]
    for ob in rng.sample(holders, 20):
        trades.append(Trade(tid(), past(1000), ob.obligor_id, "EQUITY", "BUY",
                            round(rng.uniform(20, 120), 0) * 1e6, 0.0, AS_OF, "NONE", "USD", f"EQ-{ob.obligor_id}"))
    trades.sort(key=lambda t: (t.trade_date, t.trade_id))
    return trades


def write_blotter(trades: list[Trade], obligors: list[Obligor], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "trade_blotter.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["trade_id", "trade_date", "counterparty", "instrument", "direction", "notional",
                    "rate", "maturity", "collateral", "currency", "facility_id", "extra", "synthetic"])
        for t in trades:
            extra = ";".join(f"{k}={v}" for k, v in t.extra.items())
            w.writerow([t.trade_id, t.trade_date.isoformat(), t.counterparty, t.instrument, t.direction,
                        f"{t.notional:.0f}", t.rate, t.maturity.isoformat(), t.collateral, t.currency,
                        t.facility_id, extra, "true"])
    with (out_dir / "obligors.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["obligor_id", "name", "sector", "country", "internal_rating", "origination_rating",
                    "listed", "bank", "index_member", "group", "synthetic_exposure"])
        for o in obligors:
            w.writerow([o.obligor_id, o.name, o.sector, o.country, o.rating, o.orig_rating,
                        o.listed, o.bank, o.in_index, o.group, "true"])


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [r for r in csv.DictReader(fh)]


def india_desk(rows: list[dict[str, str]], first_trade: int) -> tuple[list[Obligor], list[Trade]]:
    """Obligors and trades for the 16 NSE names, on their own seed (the core book is untouched)."""
    rng = random.Random(SEED_INDIA)
    obligors: list[Obligor] = []
    for r in rows:
        fin = r["sector"] == "Financials"
        rating = "BBB-" if r["ticker"] in GROUPS else _rating_between(rng, "BBB-", "BBB+")
        obligors.append(Obligor(r["ticker"], r["name"], r["sector"], "IN", rating, rating, True, fin, False,
                                GROUPS.get(r["ticker"], "")))
    trades: list[Trade] = []
    n = first_trade

    def tid() -> str:
        nonlocal n
        n += 1
        return f"T{n:05d}"

    def past(days: int) -> date:
        return AS_OF - timedelta(days=rng.randint(30, days))

    def future(lo: float, hi: float) -> date:
        return AS_OF + timedelta(days=int(365 * rng.uniform(lo, hi)))

    for ob in obligors:
        if ob.obligor_id in ADANI_LINES:
            loan, revolver, bond, stake = ADANI_LINES[ob.obligor_id]
            facilities = [("TERM_LOAN", loan * 1e6, 1.0), ("REVOLVER", revolver * 1e6, 0.6)]
        else:
            loan, bond, stake = rng.uniform(120, 380), (rng.uniform(60, 160) if ob.bank or rng.random() < 0.3 else 0), 0
            facilities = [("TERM_LOAN", round(loan, 1) * 1e6, 1.0)]
        for i, (kind, limit, share) in enumerate(facilities):
            fac = f"F-{ob.obligor_id}-{i + 1}"
            mat = future(1.5, 6.0)
            rate = round(rng.uniform(0.078, 0.095), 4)
            collateral = "SENIOR_UNSECURED" if kind == "REVOLVER" else rng.choice(["SECURED", "SENIOR_UNSECURED"])
            trades.append(Trade(tid(), past(1500), ob.obligor_id, kind, "COMMIT", limit, rate, mat, collateral, "INR", fac))
            trades.append(Trade(tid(), past(1200), ob.obligor_id, kind, "DRAW", round(limit * share, -3), rate, mat,
                                collateral, "INR", fac))
        if bond:
            mat = future(3.0, 9.0)
            coupon = round(rng.uniform(0.04, 0.055), 4)
            trades.append(Trade(tid(), past(1500), ob.obligor_id, "BOND", "BUY", round(bond, 0) * 1e6, coupon, mat,
                                "SENIOR_UNSECURED", "USD", f"B-{ob.obligor_id}-{mat:%Y}"))
        if stake:
            trades.append(Trade(tid(), past(900), ob.obligor_id, "EQUITY", "BUY", stake * 1e6, 0.0, AS_OF, "NONE",
                                "INR", f"EQ-{ob.obligor_id}"))
    return obligors, trades


def generate(root: Path) -> tuple[list[Obligor], list[Trade]]:
    rng = random.Random(SEED)
    obligors = build_obligors(read_rows(root / "data" / "universe.csv"), read_rows(root / "data" / "watchlist.csv"), rng)
    trades = generate_trades(obligors, rng)
    india = root / "data" / "universe_in.csv"
    if india.exists():
        more_obligors, more_trades = india_desk(read_rows(india), len(trades))
        obligors += more_obligors
        trades = sorted(trades + more_trades, key=lambda t: (t.trade_date, t.trade_id))
    return obligors, trades


def aggregate(trades: list[Trade]) -> dict[str, dict]:
    """Net trades into positions keyed by facility or instrument id."""
    pos: dict[str, dict] = defaultdict(lambda: {"limit": 0.0, "drawn": 0.0, "face": 0.0, "notional": 0.0})
    for t in trades:
        p = pos[t.facility_id]
        p.update(instrument=t.instrument, counterparty=t.counterparty, rate=t.rate, maturity=t.maturity,
                 collateral=t.collateral, currency=t.currency, direction=p.get("direction", t.direction),
                 extra=t.extra, trades=p.get("trades", 0) + 1)
        if t.instrument in ("TERM_LOAN", "REVOLVER"):
            if t.direction == "COMMIT":
                p["limit"] += t.notional
            elif t.direction == "DRAW":
                p["drawn"] += t.notional
            elif t.direction == "REPAY":
                p["drawn"] -= t.notional
        elif t.instrument == "BOND":
            p["face"] += t.notional if t.direction == "BUY" else -t.notional
            p["direction"] = "LONG"
        else:
            p["notional"] += t.notional
            p["direction"] = t.direction
    return dict(pos)


def main() -> int:
    from seismo.config import load_settings

    root = load_settings().root
    obligors, trades = generate(root)
    write_blotter(trades, obligors, root / "data" / "blotter")
    print(f"Wrote {len(trades)} trades for {len(obligors)} obligors -> {len(aggregate(trades))} positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
