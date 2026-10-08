"""Event-grain signals: one evolving signal per story cluster.

Every time a document joins a cluster (or an official denial disputes it), the cluster is
re-scored: credibility-weighted sentiment, a weighted vote on the event class, corroboration by
independent owner, social reach with coordinated groups collapsed, and impact from the impact
model. Retracted stories keep their history but their status flips to "retracted".
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from seismo.nlp.clusters import Cluster, Member
from seismo.nlp.impact import ImpactFeatures, PriorImpact
from seismo.schemas import (
    Corroboration,
    EntityRef,
    EventClass,
    EventLabel,
    Evidence,
    Signal,
    SourceType,
    stable_id,
)

VERSION = "event-agg-v1"
CORROBORATION_WINDOW = timedelta(minutes=60)


def _member_weight(m: Member, group_size: int) -> float:
    w = (m.relevance / 100.0) * max(m.credibility, 0.05) * max(m.sent_conf, 0.05)
    return w / max(1, group_size)


def event_label(cluster: Cluster) -> EventLabel:
    votes: dict[EventClass, float] = defaultdict(float)
    best_sub: dict[EventClass, tuple[float, str | None]] = {}
    for m in cluster.members:
        w = max(m.credibility, 0.05) * m.event_conf
        votes[m.event] += w
        if w > best_sub.get(m.event, (-1.0, None))[0]:
            best_sub[m.event] = (w, m.subtype)
    if not votes:
        return EventLabel(primary=EventClass.OTHER, confidence=0.3, labels=[EventClass.OTHER], method=VERSION)
    # OTHER only wins when nothing else was said.
    ranked = sorted(votes.items(), key=lambda kv: (kv[0] == EventClass.OTHER, -kv[1]))
    top, top_w = ranked[0]
    # Commentary (OTHER) neither wins nor dilutes: confidence is the share among real classes.
    total = sum(w for c, w in votes.items() if c != EventClass.OTHER) or sum(votes.values())
    members = [m for m in cluster.members if m.event == top]
    mean_conf = sum(m.event_conf for m in members) / len(members)
    conf = (top_w / total) * mean_conf
    if any(m.source_type == SourceType.FILING for m in members):
        conf = max(conf, 0.95)
    labels = [c for c, w in ranked if w >= 0.3 * top_w]
    return EventLabel(
        primary=top, subtype=best_sub[top][1], confidence=round(min(0.99, conf), 3),
        labels=labels, method=VERSION,
    )


class EventBuilder:
    def __init__(self, impact_model: PriorImpact | None = None) -> None:
        self.impact = impact_model or PriorImpact()

    def build(
        self,
        cluster: Cluster,
        entity: EntityRef,
        as_of: datetime,
        versions: dict[str, str],
        latency_ms: dict[str, float] | None = None,
        synthetic: bool = False,
    ) -> Signal:
        members = [m for m in cluster.members if m.published_at <= as_of] or cluster.members
        group_size: dict[str, int] = {}
        for g in cluster.groups:
            if g.coordinated:
                for d in g.doc_ids:
                    group_size[d] = len(g.doc_ids)

        num = den = 0.0
        weighted: list[tuple[float, Member]] = []
        for m in members:
            w = _member_weight(m, group_size.get(m.doc_id, 1))
            num += w * m.sentiment
            den += w
            weighted.append((w, m))
        sentiment = num / den if den > 0 else 0.0
        sent_conf = den / (den + 0.25)

        label = event_label(cluster)
        news_like = [m for m in members if m.source_type != SourceType.SOCIAL and not m.lookalike]
        keys = sorted({m.key for m in news_like})
        recent_keys = {m.key for m in news_like if as_of - m.published_at <= CORROBORATION_WINDOW}
        social_authors = {m.author or m.doc_id for m in members if m.source_type == SourceType.SOCIAL}
        coordinated = any(g.coordinated for g in cluster.groups)
        authoritative = any(m.authoritative and not m.lookalike for m in members)
        relevance = max(m.relevance for m in members)
        features = ImpactFeatures(
            event=label.primary, sentiment=sentiment, publishers=len(keys),
            social_authors=len(social_authors), authoritative=authoritative,
            novelty=100, relevance=relevance, subtype=label.subtype,
        )
        impact, p = self.impact.score(features)

        flags: list[str] = []
        if coordinated:
            flags.append("coordinated_posts")
        if any(m.lookalike for m in members):
            flags.append("lookalike_source")
        if any(m.syndicated for m in members):
            flags.append("syndicated_copies")
        if authoritative:
            flags.append("authoritative_source")
        status = "active"
        if cluster.disputed:
            status = "retracted"
            flags.append("official_denial")

        top = sorted(weighted, key=lambda x: (-x[0], x[1].published_at))[:3]
        evidence = [
            Evidence(doc_id=m.doc_id, publisher=m.publisher, url=m.url, title=m.title, span=m.span,
                     weight=round(w / den, 4) if den > 0 else 0.0)
            for w, m in top if m.span
        ]
        latest = max(members, key=lambda m: (m.published_at, m.doc_id))
        return Signal(
            signal_id=stable_id("sig_evt", cluster.cluster_id, latest.doc_id, status, len(cluster.disputed_by)),
            grain="event",
            as_of=as_of,
            entity=entity,
            sentiment_score=round(max(-1.0, min(1.0, sentiment)), 4),
            sentiment_confidence=round(sent_conf, 4),
            event=label,
            impact_score=impact,
            p_large_move=round(p, 4) if self.impact.calibrated else None,
            novelty=100,
            relevance=relevance,
            corroboration=Corroboration(
                independent_publishers=len(keys),
                source_types=sorted({m.source_type for m in members}),
                authoritative=authoritative,
                disputed=cluster.disputed,
                publishers_60m=len(recent_keys),
                social_authors=len(social_authors),
                coordinated=coordinated,
                publishers=[k.split(":", 1)[1] for k in keys][:12],
                first_seen=cluster.first_seen,
            ),
            evidence=evidence,
            model_versions={**versions, "event_aggregator": VERSION, "impact": self.impact.name},
            latency_ms=latency_ms or {},
            synthetic=synthetic,
            cluster_id=cluster.cluster_id,
            status=status,  # type: ignore[arg-type]
            flags=flags,
        )
