"""Module A as a bus consumer: sentiment in, index weights out.

Subscribes to event-grain signals for index members (and to gate decisions for the circuit
breaker), keeps a filtered, decayed sentiment index per name, and records a weights snapshot
whenever the proposed change clears the no-trade band. A naive comparator (latest document
sentiment, proportional reweighting, no filters or constraints) runs alongside on the same feed.

Two indices share the code: "US" (20 S&P 100 names, caps from SEC shares x close) and "IN"
(16 Nifty 50 names). The India benchmark is built when the first signal arrives, from shares
outstanding x the NSE close before that moment, so a 2023 replay starts from 2023 weights.
"""

from __future__ import annotations

import json
import math
from datetime import datetime

from seismo.module_a.index import Holding, TiltConfig, capped_cap_weights, target_weights, turnover
from seismo.module_a.market import FALLBACK_VOL, caps_from_shares, ewma_vol, load_caps
from seismo.schemas import EventClass, GateDecision, Signal, stable_id

BREAKER_CLASSES = {EventClass.CREDIT_EVENT, EventClass.OPERATIONAL_ESG}
BREAKER_SUBTYPES = {"FRAUD_ALLEGATION"}   # a corroborated fraud report also trips the breaker
RECORD_KIND = {"US": "weights", "IN": "weights_in"}


class ModuleAConsumer:
    def __init__(self, holdings: dict[str, Holding], store, cfg: TiltConfig | None = None,
                 inputs: str = "market-data", market: str = "US", members: set[str] | None = None,
                 init=None) -> None:
        self.store = store
        self.cfg = cfg or TiltConfig()
        self.market = market
        self.lam = math.log(2) / (self.cfg.half_life_hours * 3600)
        self.members = set(members or holdings)
        self._init = init   # callable(as_of) -> (holdings, inputs), for a benchmark dated to the replay
        self.naive_sent: dict[str, float] = {}
        self.naive_turnover = 0.0
        self.turnover_total = 0.0
        self.breaker: set[str] = set()
        self._qualified: set[str] = set()
        self.started = False
        self.snapshots: list[dict] = []
        self._setup(holdings, inputs)

    def _setup(self, holdings: dict[str, Holding], inputs: str) -> None:
        self.holdings = holdings
        self.inputs = inputs
        self.bench = {t: h.bench for t, h in holdings.items()}
        self.weights = dict(self.bench)
        self.naive = dict(self.bench)
        self.events: dict[str, dict[str, Signal]] = {t: {} for t in holdings}

    def _ensure(self, now: datetime) -> None:
        if self._init is not None and not self.holdings:
            self._setup(*self._init(now))

    @classmethod
    def from_settings(cls, settings, store, strict: bool = False, market: str = "US") -> ModuleAConsumer:
        from seismo.universe import Universe

        universe = Universe.load(settings.path("universe.path"))
        members = universe.index_members(market)
        market_dir = settings.root / "data" / "market"
        cfg = TiltConfig.strict() if strict else TiltConfig(
            kappa=float(settings.get("module_a.kappa", 0.6)),
            half_life_hours=float(settings.get("aggregation.half_life_hours", 6.0)),
        )

        def holdings_for(caps, vols) -> dict[str, Holding]:
            bench = capped_cap_weights(caps or {e.entity_id: 1.0 for e in members}, 0.15)
            return {
                e.entity_id: Holding(e.entity_id, e.sector or "Unknown", bench.get(e.entity_id, 0.0),
                                     (vols or {}).get(e.entity_id, FALLBACK_VOL), e.group)
                for e in members
            }

        if market == "US":
            caps = load_caps(market_dir)
            vols = ewma_vol(market_dir, [e.entity_id for e in members])
            return cls(holdings_for(caps, vols), store, cfg, "market-data" if caps and vols else "fallback")

        def init(as_of: datetime):
            caps = caps_from_shares(market_dir, [e.entity_id for e in members], as_of)
            vols = ewma_vol(market_dir, [e.entity_id for e in members], as_of)
            return holdings_for(caps, vols), "market-data" if caps and vols else "fallback"

        return cls({}, store, cfg, market=market, members={e.entity_id for e in members}, init=init)

    # ------------------------------------------------------------------ inputs
    def qualifies(self, sig: Signal) -> bool:
        """S&P DJI-style filters, with hysteresis: a qualifying story stays in until |s| halves,
        so one neutral repost does not flip a name in and out of the tilt."""
        c = self.cfg
        if sig.relevance < c.min_relevance or sig.status == "retracted":
            return False
        key = sig.cluster_id or sig.signal_id
        threshold = c.min_abs_sentiment / 2 if key in self._qualified else c.min_abs_sentiment
        ok = abs(sig.sentiment_score) >= threshold
        if ok:
            self._qualified.add(key)
        else:
            self._qualified.discard(key)
        return ok

    def index_values(self, now: datetime) -> tuple[dict[str, float], dict[str, int], dict[str, list[Signal]]]:
        vals, counts, used = {}, {}, {}
        for t, by_cluster in self.events.items():
            num = den = 0.0
            live = []
            for sig in by_cluster.values():
                if not self.qualifies(sig):
                    continue
                if sig.novelty < self.cfg.min_novelty:
                    continue
                age = max(0.0, (now - sig.as_of).total_seconds())
                w = (sig.relevance / 100) * max(sig.sentiment_confidence, 0.05) * math.exp(-self.lam * age)
                w *= 1 + math.log2(1 + sig.corroboration.independent_publishers)
                num += w * sig.sentiment_score
                den += w
                live.append(sig)
            vals[t] = num / den if den > 0 else 0.0
            counts[t] = len(live)
            used[t] = live
        return vals, counts, used

    # ------------------------------------------------------------------ bus
    def on_signal(self, sig: Signal) -> None:
        t = sig.entity.id
        if t not in self.members:
            return
        self._ensure(sig.as_of)
        if sig.grain == "document":
            self.naive_sent[t] = sig.sentiment_score
            raw = {k: self.bench[k] * max(0.0, 1 + self.naive_sent.get(k, 0.0)) for k in self.bench}
            total = sum(raw.values()) or 1.0
            new = {k: v / total for k, v in raw.items()}
            self.naive_turnover += turnover(new, self.naive)
            self.naive = new
            return
        if sig.grain != "event" or not sig.cluster_id:
            return
        self.events[t][sig.cluster_id] = sig
        self.rebalance(sig.as_of, f"{t}: {sig.event.primary.value.lower()} signal", trigger=sig)

    def on_decision(self, d: GateDecision) -> None:
        t = d.entity_id
        if t not in self.members:
            return
        self._ensure(d.as_of)
        breaker_event = d.event.primary in BREAKER_CLASSES or (d.event.subtype or "") in BREAKER_SUBTYPES
        if d.decision == "TRIGGER" and breaker_event and d.impact_score >= 8:
            if t not in self.breaker:
                self.breaker.add(t)
                self.rebalance(d.as_of, f"{t}: risk circuit breaker ({d.event.primary.value}, impact {d.impact_score})", force=True)
        elif d.decision == "RETRACT" and t in self.breaker:
            self.breaker.discard(t)
            self.rebalance(d.as_of, f"{t}: retraction unwinds the circuit breaker", force=True)

    # ------------------------------------------------------------------ rebalance
    def rebalance(self, now: datetime, reason: str, force: bool = False, trigger: Signal | None = None) -> dict | None:
        vals, counts, used = self.index_values(now)
        state = target_weights(self.holdings, vals, counts, self.breaker, self.cfg)
        to = turnover(state.weights, self.weights)
        if not self.started:
            self._snapshot(now, "start: benchmark weights", self.weights, self.weights, state, vals, counts, used, 0.0, None)
            self.started = True
        if to < self.cfg.band and not force:
            return None
        prev = self.weights
        self.weights = state.weights
        self.turnover_total += to
        return self._snapshot(now, reason, prev, self.weights, state, vals, counts, used, to, trigger)

    def _snapshot(self, now, reason, prev, weights, state, vals, counts, used, to, trigger) -> dict:
        attribution = {}
        for t in self.holdings:
            attribution[t] = {
                "bench": round(self.bench[t], 6),
                "prev": round(prev[t], 6),
                "signal_target": round(state.raw.get(t, self.bench[t]), 6),
                "final": round(weights.get(t, self.bench[t]), 6),
                "from_signal": round(state.raw.get(t, self.bench[t]) - self.bench[t], 6),
                "from_constraints": round(state.weights.get(t, self.bench[t]) - state.raw.get(t, self.bench[t]), 6),
                "breaker": t in state.breaker,
                "events": [
                    {"signal_id": s.signal_id, "headline": (s.evidence[0].title or s.evidence[0].span) if s.evidence else "",
                     "span": s.evidence[0].span if s.evidence else "", "publisher": s.evidence[0].publisher if s.evidence else "",
                     "sentiment": s.sentiment_score, "impact": s.impact_score, "event": s.event.primary.value,
                     "publishers": s.corroboration.independent_publishers}
                    for s in used.get(t, [])
                ],
            }
        snap = {
            "snapshot_id": stable_id("wts", now.isoformat(), reason),
            "as_of": now.isoformat(),
            "reason": reason,
            "weights": {t: round(v, 6) for t, v in weights.items()},
            "bench": {t: round(v, 6) for t, v in self.bench.items()},
            "naive": {t: round(v, 6) for t, v in self.naive.items()},
            "index": {t: round(v, 4) for t, v in vals.items()},
            "z": {t: round(v, 4) for t, v in state.z.items()},
            "alpha": {t: round(v, 5) for t, v in state.alpha.items()},
            "counts": counts,
            "breaker": sorted(state.breaker),
            "turnover": round(to, 6),
            "cost_bps": round(to * self.cfg.cost_bps, 4),
            "turnover_total": round(self.turnover_total, 6),
            "naive_turnover_total": round(self.naive_turnover, 6),
            "attribution": attribution,
            "trigger_signal": trigger.signal_id if trigger else None,
            "inputs": self.inputs,
            "market": self.market,
            "config": {"kappa": self.cfg.kappa, "half_life_hours": self.cfg.half_life_hours,
                       "name_cap": self.cfg.name_cap, "active_cap": self.cfg.active_cap, "band": self.cfg.band,
                       "strict": self.cfg.min_abs_sentiment >= 0.6},
        }
        self.snapshots.append(snap)
        if self.store is not None:
            self.store.put_record(RECORD_KIND.get(self.market, "weights"), snap["snapshot_id"], now, json.dumps(snap))
        return snap


def load_snapshots(store, market: str = "US") -> list[dict]:
    return [json.loads(r) for r in store.records(RECORD_KIND.get(market, "weights"))]
