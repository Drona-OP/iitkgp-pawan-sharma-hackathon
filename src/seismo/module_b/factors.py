"""Risk factors and historical-analog shock vectors, computed from data, never typed by hand.

For each scenario window in config/scenarios.yaml, the shock to a factor is its largest move
from the window's base (the close on the start date) to any later close inside the window:

    price-like factors (equity ETFs, FX, oil, gold):  x_t / x_0 - 1, the t with the largest |move|
    rates and spreads (FRED, in percent):             (x_t - x_0) x 100, in basis points
    VIX:                                              x_t - x_0, in vol points

    python -m seismo.module_b.factors      # writes data/market/analog_shocks.csv

India: Nifty 50 and Nifty Bank are the equity factors for Indian obligors, the India 10-year
yield moves INR bonds when a monthly print falls in the window, and every NSE name in
data/universe_in.csv also gets its own observed move ("EQN:<ticker>"), so an Indian stake is
revalued by what that stock actually did in the analog window.

Credit spreads use ICE BofA option-adjusted spreads where FRED history covers the window and
Moody's Baa-10y as the long-history fallback (scaled 1x for IG/BBB and 3x for high yield, a
stated assumption). Missing sector ETFs (XLC before 2018, XLRE before 2015) fall back to SPY.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from seismo.config import load_settings
from seismo.scenarios import load_library

SECTOR_FACTOR = {
    "Information Technology": "EQ:IT", "Communication Services": "EQ:COMM",
    "Consumer Discretionary": "EQ:CD", "Financials": "EQ:FIN", "Health Care": "EQ:HC",
    "Energy": "EQ:EN", "Industrials": "EQ:IND", "Consumer Staples": "EQ:CS", "Utilities": "EQ:UT",
    "Materials": "EQ:MAT", "Real Estate": "EQ:RE",
}

# factor -> (kind, primary column, [(fallback column, scale), ...])
FACTORS: dict[str, tuple[str, str, list[tuple[str, float]]]] = {
    "EQ:IT": ("ret", "XLK", [("SPY", 1.0)]),
    "EQ:COMM": ("ret", "XLC", [("SPY", 1.0)]),
    "EQ:CD": ("ret", "XLY", [("SPY", 1.0)]),
    "EQ:FIN": ("ret", "XLF", [("SPY", 1.0)]),
    "EQ:HC": ("ret", "XLV", [("SPY", 1.0)]),
    "EQ:EN": ("ret", "XLE", [("SPY", 1.0)]),
    "EQ:IND": ("ret", "XLI", [("SPY", 1.0)]),
    "EQ:CS": ("ret", "XLP", [("SPY", 1.0)]),
    "EQ:UT": ("ret", "XLU", [("SPY", 1.0)]),
    "EQ:MAT": ("ret", "XLB", [("SPY", 1.0)]),
    "EQ:RE": ("ret", "XLRE", [("SPY", 1.0)]),
    "EQ:MKT": ("ret", "SPY", []),
    "EQ:BANKS": ("ret", "KRE", [("XLF", 1.0)]),
    "IR:3M": ("bp", "UST_3M", []),
    "IR:2Y": ("bp", "UST_2Y", []),
    "IR:5Y": ("bp", "UST_5Y", []),
    "IR:10Y": ("bp", "UST_10Y", []),
    "IR:30Y": ("bp", "UST_30Y", []),
    "CR:IG": ("bp", "IG_OAS", [("BAA_10Y", 1.0)]),
    "CR:BBB": ("bp", "BBB_OAS", [("BAA_10Y", 1.0)]),
    "CR:HY": ("bp", "HY_OAS", [("BAA_10Y", 3.0)]),
    "FX:DXY": ("ret", "DXY", []),
    "FX:USDINR": ("ret", "USDINR", [("USDINR_FRED", 1.0)]),
    "FX:EURUSD": ("ret", "EURUSD", []),
    "CM:BRENT": ("ret", "BRENT", [("BRENT_FRED", 1.0)]),
    "CM:GOLD": ("ret", "GOLD", []),
    "VOL:VIX": ("pts", "VIX", [("VIX_FRED", 1.0)]),
    "EQ:IN": ("ret", "NIFTY50", []),
    "EQ:IN_BANKS": ("ret", "NIFTYBANK", [("NIFTY50", 1.0)]),
    "IR:IN10Y": ("bp", "IN_10Y", []),
}
NAME_SUFFIX = ".NS"   # price columns that also get a name-level shock
TENORS = [(0.25, "IR:3M"), (2.0, "IR:2Y"), (5.0, "IR:5Y"), (10.0, "IR:10Y"), (30.0, "IR:30Y")]


@dataclass(frozen=True)
class Shock:
    scenario: str
    factor: str
    value: float          # return (fraction), bp or vol points by factor kind
    kind: str
    source: str
    peak_date: str


def _read_wide(path: Path):
    import pandas as pd

    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def load_market(market_dir: str | Path):
    """Prices and FRED series joined on date (forward-filled across holidays)."""
    import pandas as pd

    market_dir = Path(market_dir)
    frames = []
    for name in ("prices_daily.csv", "fred_daily.csv"):
        if (market_dir / name).exists():
            frames.append(_read_wide(market_dir / name))
    if not frames:
        raise FileNotFoundError(f"No market data in {market_dir}. Run scripts/fetch_data.py first.")
    return pd.concat(frames, axis=1).sort_index().ffill(limit=5)


def _excursion(series, start, end, kind: str):
    s = series.dropna()
    base_part = s[s.index <= start]
    window = s[(s.index > start) & (s.index <= end)]
    if base_part.empty or window.empty or (start - base_part.index[-1]).days > 7:
        return None
    x0 = float(base_part.iloc[-1])
    if kind == "ret":
        moves = window / x0 - 1.0
    elif kind == "bp":
        moves = (window - x0) * 100.0
    else:
        moves = window - x0
    idx = moves.abs().idxmax()
    return float(moves.loc[idx]), idx.strftime("%Y-%m-%d")


def compute_shocks(market, scenarios) -> list[Shock]:
    import pandas as pd

    out: list[Shock] = []
    for sid, sc in scenarios.items():
        start, end = pd.Timestamp(sc.start), pd.Timestamp(sc.end)
        for factor, (kind, col, fallbacks) in FACTORS.items():
            got = None
            for column, scale, label in [(col, 1.0, col)] + [(c, s, f"fallback:{c}" + (f"x{s:g}" if s != 1 else "")) for c, s in fallbacks]:
                if column in market.columns:
                    res = _excursion(market[column], start, end, kind)
                    if res is not None:
                        got = (res[0] * scale, label, res[1])
                        break
            if got is not None:
                out.append(Shock(sid, factor, round(got[0], 6), kind, got[1], got[2]))
        for column in sorted(c for c in market.columns if str(c).endswith(NAME_SUFFIX)):
            res = _excursion(market[column], start, end, "ret")
            if res is not None:
                out.append(Shock(sid, f"EQN:{column}", round(res[0], 6), "ret", column, res[1]))
    return out


def write_shocks(shocks: list[Shock], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["scenario", "factor", "value", "kind", "source", "peak_date"])
        for s in shocks:
            w.writerow([s.scenario, s.factor, s.value, s.kind, s.source, s.peak_date])


def load_shocks(path: str | Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["scenario"], {})[row["factor"]] = float(row["value"])
    return out


def shock_table(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def rate_at(shock: dict[str, float], years: float) -> float:
    """Linear interpolation of the rate shock (bp) across the Treasury curve."""
    pts = [(t, shock.get(f, 0.0)) for t, f in TENORS if f in shock]
    if not pts:
        return 0.0
    if years <= pts[0][0]:
        return pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:], strict=False):
        if years <= t1:
            return v0 + (v1 - v0) * (years - t0) / (t1 - t0)
    return pts[-1][1]


def spread_shock(shock: dict[str, float], rating: str | None) -> float:
    from seismo.module_b.credit import NOTCH

    if rating is None:
        return shock.get("CR:BBB", 0.0)
    n = NOTCH.get(rating, NOTCH["BBB"])
    if n <= NOTCH["A-"]:
        return shock.get("CR:IG", 0.0)
    if n <= NOTCH["BBB-"]:
        return shock.get("CR:BBB", 0.0)
    return shock.get("CR:HY", 0.0)


def main() -> int:
    settings = load_settings()
    lib = load_library(str(settings.path("scenarios.path")))
    market_dir = settings.root / "data" / "market"
    market = load_market(market_dir)
    shocks = compute_shocks(market, lib.scenarios)
    out = market_dir / "analog_shocks.csv"
    write_shocks(shocks, out)
    print(f"Wrote {len(shocks)} shocks for {len(lib.scenarios)} scenarios to {out}")
    for sid in lib.scenarios:
        row = {s.factor: s.value for s in shocks if s.scenario == sid}
        print(f"  {sid:22s} SPY {row.get('EQ:MKT', float('nan')):+.1%}  2Y {row.get('IR:2Y', float('nan')):+.0f}bp  "
              f"HY {row.get('CR:HY', float('nan')):+.0f}bp  VIX {row.get('VOL:VIX', float('nan')):+.1f}  "
              f"Nifty {row.get('EQ:IN', float('nan')):+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
