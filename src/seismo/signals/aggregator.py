"""Entity-grain signals: an exponentially decayed, weighted sentiment index per entity, built from
event-grain signals so a syndicated story or a bot swarm counts once.

    S_e(t) = sum_k w_k s_k exp(-lambda (t - t_k)) / sum_k w_k exp(-lambda (t - t_k)),
    w_k = relevance_k x confidence_k x (1 + log2(1 + publishers_k)),  lambda = ln 2 / half-life,

where k runs over the entity's story clusters (latest state of each) and t_k is the cluster's
latest update. Retracted stories get zero weight, which unwinds anything they moved. Impact and
event come from the strongest live story in the impact window.
"""

from __future__ import annotations

import math

from seismo.schemas import Corroboration, Signal, stable_id

VERSION = "decay-agg-0.2"


class EntityAggregator:
    def __init__(
        self,
        half_life_hours: float = 6.0,
        impact_window_hours: float = 6.0,
        history_hours: float = 72.0,
    ) -> None:
        self.half_life_hours = half_life_hours
        self.lam = math.log(2) / (half_life_hours * 3600.0)
        self.impact_window = impact_window_hours * 3600.0
        self.history = history_hours * 3600.0
        self._clusters: dict[str, dict[str, Signal]] = {}

    @staticmethod
    def weight(sig: Signal) -> float:
        if sig.status == "retracted":
            return 0.0
        pubs = sig.corroboration.independent_publishers
        return (sig.relevance / 100.0) * max(sig.sentiment_confidence, 0.05) * (1 + math.log2(1 + pubs))

    def update(self, sig: Signal) -> Signal:
        entity_id = sig.entity.id
        now = sig.as_of
        key = sig.cluster_id or sig.signal_id
        book = self._clusters.setdefault(entity_id, {})
        book[key] = sig
        for k in [k for k, s in book.items() if (now - s.as_of).total_seconds() > self.history]:
            del book[k]
        items = list(book.values())

        num = den = 0.0
        weighted: list[tuple[float, Signal]] = []
        for s in items:
            age = max(0.0, (now - s.as_of).total_seconds())
            w = self.weight(s) * math.exp(-self.lam * age)
            num += w * s.sentiment_score
            den += w
            weighted.append((w, s))
        index = num / den if den > 0 else 0.0

        live = [s for s in items if s.status != "retracted"
                and 0 <= (now - s.as_of).total_seconds() <= self.impact_window]
        top = max(live, key=lambda s: (s.impact_score, s.as_of)) if live else sig
        evidence = [ev for _, s in sorted(weighted, key=lambda x: -x[0])[:3] for ev in s.evidence[:1]]
        c = top.corroboration
        return Signal(
            signal_id=stable_id("sig_ent", entity_id, sig.signal_id),
            grain="entity",
            as_of=now,
            entity=sig.entity,
            sentiment_score=round(max(-1.0, min(1.0, index)), 4),
            sentiment_confidence=round(den / (den + 0.5), 4),
            event=top.event,
            impact_score=top.impact_score if live else 1,
            p_large_move=top.p_large_move if live else None,
            novelty=top.novelty,
            relevance=max((s.relevance for s in items), default=sig.relevance),
            corroboration=Corroboration(
                independent_publishers=c.independent_publishers,
                source_types=c.source_types,
                authoritative=c.authoritative,
                disputed=c.disputed,
                publishers_60m=c.publishers_60m,
                social_authors=c.social_authors,
                coordinated=c.coordinated,
                publishers=c.publishers,
                first_seen=c.first_seen,
            ),
            evidence=evidence,
            model_versions={**sig.model_versions, "aggregator": VERSION},
            latency_ms=sig.latency_ms,
            synthetic=any(s.synthetic for s in items),
            cluster_id=top.cluster_id,
            status="active",
            flags=[f"stories:{len(items)}"] + (["retraction_unwound"] if sig.status == "retracted" else []),
        )
