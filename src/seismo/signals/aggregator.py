"""Entity-grain signals: an exponentially decayed, weighted sentiment index per entity.

    S_e(t) = sum_i w_i s_i exp(-lambda (t - t_i)) / sum_i w_i exp(-lambda (t - t_i)),
    w_i = relevance x novelty x confidence x source weight,  lambda = ln 2 / half-life.

Impact and event come from the strongest document in the impact window; corroboration counts
the distinct publishers reporting that same event class in the window.
"""

from __future__ import annotations

import math

from seismo.schemas import Corroboration, Signal, SourceType, stable_id

VERSION = "decay-agg-0.1"
SOURCE_WEIGHT = {SourceType.FILING: 1.0, SourceType.NEWS: 0.7, SourceType.SOCIAL: 0.3}


class EntityAggregator:
    def __init__(
        self,
        half_life_hours: float = 6.0,
        impact_window_hours: float = 6.0,
        history_hours: float = 72.0,
    ) -> None:
        self.lam = math.log(2) / (half_life_hours * 3600.0)
        self.impact_window = impact_window_hours * 3600.0
        self.history = history_hours * 3600.0
        self._items: dict[str, list[Signal]] = {}

    @staticmethod
    def _weight(sig: Signal) -> float:
        source = sig.corroboration.source_types[0] if sig.corroboration.source_types else SourceType.NEWS
        return (
            (sig.relevance / 100.0)
            * (sig.novelty / 100.0)
            * max(sig.sentiment_confidence, 0.05)
            * SOURCE_WEIGHT[source]
        )

    def update(self, sig: Signal) -> Signal:
        entity_id = sig.entity.id
        now = sig.as_of
        items = [s for s in self._items.get(entity_id, []) if (now - s.as_of).total_seconds() <= self.history]
        items.append(sig)
        self._items[entity_id] = items

        num = den = 0.0
        weighted: list[tuple[float, Signal]] = []
        for s in items:
            age = max(0.0, (now - s.as_of).total_seconds())
            w = self._weight(s) * math.exp(-self.lam * age)
            num += w * s.sentiment_score
            den += w
            weighted.append((w, s))
        index = num / den if den > 0 else 0.0

        window = [s for s in items if 0 <= (now - s.as_of).total_seconds() <= self.impact_window] or [sig]
        top = max(window, key=lambda s: (s.impact_score, s.as_of))
        same_event = [s for s in window if s.event.primary == top.event.primary]
        publishers = {ev.publisher for s in same_event for ev in s.evidence}
        source_types = sorted({st for s in same_event for st in s.corroboration.source_types})
        evidence = [s.evidence[0] for _, s in sorted(weighted, key=lambda x: -x[0])[:3] if s.evidence]

        return Signal(
            signal_id=stable_id("sig_ent", entity_id, sig.signal_id),
            grain="entity",
            as_of=now,
            entity=sig.entity,
            sentiment_score=round(max(-1.0, min(1.0, index)), 4),
            sentiment_confidence=round(den / (den + 0.5), 4),
            event=top.event,
            impact_score=top.impact_score,
            novelty=top.novelty,
            relevance=max(s.relevance for s in window),
            corroboration=Corroboration(
                independent_publishers=len(publishers),
                source_types=source_types,
                authoritative=any(s.corroboration.authoritative for s in same_event),
            ),
            evidence=evidence,
            model_versions={**sig.model_versions, "aggregator": VERSION},
            latency_ms=sig.latency_ms,
            synthetic=any(s.synthetic for s in items),
        )
