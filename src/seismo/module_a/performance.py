"""Replay performance for Module A: Seismo vs capped cap-weight vs the naive tilt, with costs.

Look-ahead-safe by construction: a weights snapshot stamped at time t trades at the next
session open after t (variant: the next close), and earns that session's open-to-close return
plus close-to-close returns afterwards. Costs are 5 bp per unit of turnover. Needs
data/market/prices_daily.csv (and prices_open.csv for the open variant).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from seismo.module_a.market import closes, opens

# market -> (column whose rows define sessions, exchange time zone, open, last decision time)
MARKETS = {
    "US": ("SPY", "America/New_York", pd.Timedelta(hours=9, minutes=30), pd.Timedelta(hours=15, minutes=45)),
    "IN": ("NIFTY50", "Asia/Kolkata", pd.Timedelta(hours=9, minutes=15), pd.Timedelta(hours=15, minutes=15)),
}


def _sessions(px: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, column: str = "SPY") -> pd.DatetimeIndex:
    """Exchange sessions only (rows where the market's index traded; FX also prints on holidays)."""
    traded = px[column].notna() if column in px.columns else px.notna().any(axis=1)
    return px.index[traded & (px.index >= start.normalize()) & (px.index <= end.normalize())]


def replay_performance(snapshots: list[dict], market_dir: Path, cost_bps: float = 5.0,
                       pad_days: int = 3, trade_at: str = "open") -> pd.DataFrame | None:
    """Daily value paths (base 100) for 'Seismo', 'Benchmark' and 'Naive tilt'."""
    px_c = closes(market_dir)
    if px_c is None or not snapshots:
        return None
    column, tz, open_t, close_t = MARKETS[snapshots[0].get("market", "US")]
    px_o = opens(market_dir) if trade_at == "open" else None
    times = [pd.Timestamp(s["as_of"]).tz_convert(tz).tz_localize(None) for s in snapshots]
    start = min(times) - pd.Timedelta(days=pad_days)
    end = max(times) + pd.Timedelta(days=pad_days)
    days = _sessions(px_c, start, end, column)
    tickers = list(snapshots[0]["bench"])
    if len(days) < 2 or not set(tickers) <= set(px_c.columns):
        return None
    close = px_c.loc[days, tickers].ffill()
    open_ = px_o.loc[px_o.index.intersection(days), tickers].reindex(days) if px_o is not None else None

    def effective(day: pd.Timestamp, key: str) -> tuple[dict, int]:
        # Snapshot visible before this session's open (9:30 New York, 9:15 Mumbai) trades at the open.
        cutoff = day + (open_t if trade_at == "open" else close_t)
        idx = max((i for i, t in enumerate(times) if t <= cutoff), default=0)
        return snapshots[idx][key], idx

    out = {"Seismo": [100.0], "Benchmark": [100.0], "Naive tilt": [100.0]}
    prev = {"Seismo": None, "Benchmark": None, "Naive tilt": None}
    keys = {"Seismo": "weights", "Benchmark": "bench", "Naive tilt": "naive"}
    for i in range(1, len(days)):
        d0, d1 = days[i - 1], days[i]
        for name, key in keys.items():
            w, _ = effective(d1, key)
            if trade_at == "open" and open_ is not None and not open_.loc[d1].isna().any():
                # Old weights earn close->open; new weights earn open->close.
                w_old = prev[name] or w
                r_gap = sum(w_old[t] * (open_.loc[d1, t] / close.loc[d0, t] - 1) for t in tickers)
                r_day = sum(w[t] * (close.loc[d1, t] / open_.loc[d1, t] - 1) for t in tickers)
                r = (1 + r_gap) * (1 + r_day) - 1
            else:
                w_old = w
                r = sum(w[t] * (close.loc[d1, t] / close.loc[d0, t] - 1) for t in tickers)
            cost = 0.0
            if prev[name] is not None:
                cost = sum(abs(w[t] - prev[name][t]) for t in tickers) * cost_bps / 1e4
            prev[name] = w
            out[name].append(out[name][-1] * (1 + r - cost))
    return pd.DataFrame(out, index=days)


def summary(paths: pd.DataFrame) -> dict[str, dict[str, float]]:
    res = {}
    for col in paths.columns:
        s = paths[col]
        dd = (s / s.cummax() - 1).min()
        res[col] = {"return": float(s.iloc[-1] / s.iloc[0] - 1), "max_drawdown": float(dd)}
    return res
