"""Module A maths: a risk-controlled sentiment tilt on a 20-stock S&P 100 index.

Borrowed from the S&P 500 RavenPack AI Sentiment Index methodology and active management:
- only events that pass the filters count: relevance >= 60, novelty >= 50, |sentiment| >= 0.3
  ("strict mode" uses the S&P DJI values: 100, 100 and 0.6);
- the entity index is an exponentially decayed mean of qualifying event sentiment (6-hour
  half-life), z-scored across the universe and shrunk towards zero when few events support it:
      z~_i = z_i x n_i / (n_i + k)
- Grinold's alpha_i = IC x sigma_i x z~_i is reported for the active-risk view;
- weights: a multiplicative tilt on capped market-cap weights, with the tilt scaled by inverse
  volatility so a quiet utility moves more than a volatile chipmaker for the same news:
      w_i ~ w_b,i x exp(kappa x z~_i x sigma_bar / sigma_i)
  then a 15% name cap and a 5% active-weight cap per name and per sector;
- risk circuit breaker: a corroborated Credit or Operational event (or a fraud allegation) with
  impact >= 8 (gate TRIGGER) sends the name to its maximum underweight at once, and the other
  names of its business group may not rise above benchmark; a retraction unwinds it;
- no-trade band: rebalance only when total proposed change exceeds 2% or a breaker fires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TiltConfig:
    kappa: float = 0.6
    half_life_hours: float = 6.0
    shrink_k: float = 2.0
    name_cap: float = 0.15
    active_cap: float = 0.05
    sector_active_cap: float = 0.05
    band: float = 0.02
    cost_bps: float = 5.0
    ic: float = 0.05
    min_relevance: int = 60
    min_novelty: int = 50
    min_abs_sentiment: float = 0.3

    @classmethod
    def strict(cls, **kw) -> TiltConfig:
        return cls(min_relevance=100, min_novelty=100, min_abs_sentiment=0.6, **kw)


@dataclass
class Holding:
    ticker: str
    sector: str
    bench: float
    vol: float
    group: str = ""


@dataclass
class TiltState:
    weights: dict[str, float]
    z: dict[str, float] = field(default_factory=dict)
    raw: dict[str, float] = field(default_factory=dict)
    alpha: dict[str, float] = field(default_factory=dict)
    breaker: set[str] = field(default_factory=set)


def capped_cap_weights(caps: dict[str, float], cap: float = 0.15) -> dict[str, float]:
    """Market-cap weights with an iterative cap (excess redistributed pro rata)."""
    w = {k: max(v, 0.0) for k, v in caps.items()}
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}
    for _ in range(50):
        over = {k: v for k, v in w.items() if v > cap + 1e-12}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        free = {k: v for k, v in w.items() if k not in over}
        free_total = sum(free.values()) or 1.0
        for k in over:
            w[k] = cap
        for k, v in free.items():
            w[k] = v + excess * v / free_total
    return w


def zscores(values: dict[str, float], counts: dict[str, int], k: float) -> dict[str, float]:
    tickers = list(values)
    if not tickers:
        return {}
    mean = sum(values.values()) / len(tickers)
    var = sum((v - mean) ** 2 for v in values.values()) / max(1, len(tickers) - 1)
    sd = max(math.sqrt(var), 0.05)
    return {t: ((values[t] - mean) / sd) * counts.get(t, 0) / (counts.get(t, 0) + k) for t in tickers}


def _project(w: dict[str, float], holdings: dict[str, Holding], cfg: TiltConfig, floors: dict[str, float],
             ceilings: set[str] | None = None) -> dict[str, float]:
    """Enforce name, active and sector caps, then restore full investment (alternating passes).

    Sector caps scale the sector's active weights towards the benchmark, so a name keeps the
    direction of its tilt; the residual is spread over names with room, in proportion to room.
    """
    bench = {t: h.bench for t, h in holdings.items()}
    lo = {t: max(0.0, bench[t] - cfg.active_cap) for t in w}
    hi = {t: min(cfg.name_cap, bench[t] + cfg.active_cap) for t in w}
    for t in ceilings or ():
        hi[t] = min(hi[t], bench[t])
        lo[t] = min(lo[t], hi[t])
    for t, f in floors.items():
        lo[t] = hi[t] = f
    sectors: dict[str, list[str]] = {}
    for t, h in holdings.items():
        sectors.setdefault(h.sector, []).append(t)
    for _ in range(50):
        for t in w:
            w[t] = min(hi[t], max(lo[t], w[t]))
        for names in sectors.values():
            active = sum(w[t] - bench[t] for t in names)
            if abs(active) > cfg.sector_active_cap + 1e-12:
                scale = cfg.sector_active_cap / abs(active)
                for t in names:
                    if t not in floors:
                        w[t] = bench[t] + (w[t] - bench[t]) * scale
        resid = 1.0 - sum(w.values())
        if abs(resid) < 1e-10:
            break
        room = {t: (hi[t] - w[t]) if resid > 0 else (w[t] - lo[t]) for t in w if t not in floors}
        total_room = sum(max(0.0, r) for r in room.values())
        if total_room <= 1e-12:
            break
        for t, r in room.items():
            w[t] += resid * max(0.0, r) / total_room
    return w


def target_weights(
    holdings: dict[str, Holding], index_values: dict[str, float], counts: dict[str, int],
    breaker: set[str], cfg: TiltConfig,
) -> TiltState:
    vals = {t: index_values.get(t, 0.0) for t in holdings}
    z = zscores(vals, counts, cfg.shrink_k)
    sbar = sum(h.vol for h in holdings.values()) / len(holdings)
    raw = {t: h.bench * math.exp(cfg.kappa * z[t] * sbar / max(h.vol, 1e-6)) for t, h in holdings.items()}
    total = sum(raw.values())
    raw = {t: v / total for t, v in raw.items()}
    floors = {t: max(0.0, holdings[t].bench - cfg.active_cap) for t in breaker if t in holdings}
    groups = {holdings[t].group for t in breaker if t in holdings and holdings[t].group}
    ceilings = {t for t, h in holdings.items() if h.group in groups and t not in floors}
    w = _project(dict(raw), holdings, cfg, floors, ceilings)
    alpha = {t: cfg.ic * holdings[t].vol * z[t] for t in holdings}
    return TiltState(weights=w, z=z, raw=raw, alpha=alpha, breaker=set(breaker))


def turnover(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
