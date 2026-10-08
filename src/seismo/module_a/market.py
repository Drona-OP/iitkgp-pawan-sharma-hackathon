"""Market inputs for Module A: benchmark caps, EWMA volatility and prices (data/market/*.csv).

Without downloaded data the module still runs, on equal weights and a flat 25% volatility, and
says so in every snapshot (``inputs: fallback``).
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path

EWMA_DECAY = 0.97
FALLBACK_VOL = 0.25


def load_caps(market_dir: Path) -> dict[str, float] | None:
    path = market_dir / "caps.csv"
    if not path.exists():
        return None
    out = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                v = float(row["market_cap_usd"])
            except (KeyError, ValueError):
                continue
            if math.isfinite(v) and v > 0:
                out[row["ticker"]] = v
    return out or None


def caps_from_shares(market_dir: Path, tickers: list[str], as_of) -> dict[str, float] | None:
    """Market cap on the replay date = shares outstanding (shares_in.csv) x the last close before it."""
    import pandas as pd

    path = market_dir / "shares_in.csv"
    px = closes(market_dir)
    if not path.exists() or px is None:
        return None
    shares = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                v = float(row["shares"])
            except (KeyError, ValueError):
                continue
            if math.isfinite(v) and v > 0:
                shares[row["ticker"]] = v
    cutoff = pd.Timestamp(as_of).tz_localize(None).normalize() if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of).normalize()
    before = px[px.index < cutoff]
    out = {}
    for t in tickers:
        if t in shares and t in before.columns and before[t].notna().any():
            out[t] = shares[t] * float(before[t].dropna().iloc[-1])
    return out or None


@lru_cache(maxsize=4)
def _prices(path: str):
    import pandas as pd

    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def closes(market_dir: Path):
    return _prices(str(market_dir / "prices_daily.csv"))


def opens(market_dir: Path):
    return _prices(str(market_dir / "prices_open.csv"))


def ewma_vol(market_dir: Path, tickers: list[str], as_of=None) -> dict[str, float] | None:
    """Annualised EWMA volatility (decay 0.97, as in the S&P DJI sentiment index), as of a date."""
    import numpy as np
    import pandas as pd

    px = closes(market_dir)
    if px is None:
        return None
    if as_of is not None:
        px = px[px.index < pd.Timestamp(as_of).tz_localize(None).normalize()]
    px = px.tail(260)
    out = {}
    for t in tickers:
        if t not in px.columns:
            continue
        r = np.log(px[t].dropna()).diff().dropna().to_numpy()
        if len(r) < 20:
            continue
        w = EWMA_DECAY ** np.arange(len(r))[::-1]
        var = float(np.sum(w * r * r) / np.sum(w))
        out[t] = math.sqrt(var * 252)
    return out or None
