"""Synthetic market data for tests only: random walks with a crash injected in each analog window.

Real runs use data/market/*.csv from scripts/fetch_data.py; nothing here reaches the dashboard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLS = ["SPY", "KRE", "XLK", "XLC", "XLY", "XLF", "XLV", "XLE", "XLI", "XLP", "XLU", "XLB", "XLRE",
              "VIX", "BRENT", "GOLD", "DXY", "USDINR", "EURUSD", "NIFTY50", "NIFTYBANK", "ADANIENT.NS", "ADANIPORTS.NS"]
FRED_COLS = ["UST_3M", "UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y", "BAA_10Y", "IG_OAS", "BBB_OAS", "HY_OAS"]
WINDOWS = {  # start, end, equity move, 2y bp, spread bp, vix pts
    "2008-09-12": ("2008-10-10", -0.30, -80, 250, 40),
    "2020-02-19": ("2020-03-23", -0.33, -140, 300, 60),
    "2022-02-18": ("2022-03-08", -0.10, -20, 40, 15),
    "2022-06-09": ("2022-06-16", -0.08, 50, 30, 10),
    "2023-03-08": ("2023-03-17", -0.05, -100, 50, 10),
    "2025-01-24": ("2025-01-27", -0.04, -10, 5, 5),
    "2025-04-02": ("2025-04-08", -0.12, -30, 80, 25),
    "2023-01-24": ("2023-02-03", -0.03, 0, 0, 0),
}
# A group-specific crash on top of the window move (the Adani names in early 2023).
NAME_CRASH = {"2023-01-24": {"ADANIENT.NS": -0.55, "ADANIPORTS.NS": -0.35}}


def write_synthetic_market(out: Path, seed: int = 7) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2007-01-01", "2026-09-30")
    n = len(idx)
    prices = pd.DataFrame(index=idx)
    for c in PRICE_COLS:
        prices[c] = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    prices["VIX"] = 18 + rng.normal(0, 0.5, n).cumsum() * 0.01
    fred = pd.DataFrame(index=idx)
    for c in FRED_COLS:
        fred[c] = 3.0 + rng.normal(0, 0.01, n).cumsum() * 0.1
    for start, (end, eq, two_y, spr, vix) in WINDOWS.items():
        w = (idx > start) & (idx <= end)
        k = int(w.sum())
        ramp = np.linspace(0, 1, k)
        for c in PRICE_COLS:
            if c in ("VIX", "GOLD", "DXY", "USDINR", "EURUSD", "BRENT"):
                continue
            mult = 2.5 if c == "KRE" and start == "2023-03-08" else 1.0
            move = NAME_CRASH.get(start, {}).get(c, eq * mult)
            prices.loc[w, c] = prices.loc[w, c].to_numpy() * (1 + move * ramp)
        prices.loc[w, "VIX"] = prices.loc[w, "VIX"].to_numpy() + vix * ramp
        for c in ("UST_3M", "UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y"):
            scale = {"UST_3M": 0.8, "UST_2Y": 1.0, "UST_5Y": 0.8, "UST_10Y": 0.6, "UST_30Y": 0.4}[c]
            fred.loc[w, c] = fred.loc[w, c].to_numpy() + two_y * scale / 100 * ramp
        for c, s in (("BAA_10Y", 0.6), ("IG_OAS", 0.6), ("BBB_OAS", 1.0), ("HY_OAS", 3.0)):
            fred.loc[w, c] = fred.loc[w, c].to_numpy() + spr * s / 100 * ramp
    prices.index.name = fred.index.name = "date"
    prices.round(6).to_csv(out / "prices_daily.csv")
    fred.round(4).to_csv(out / "fred_daily.csv")
