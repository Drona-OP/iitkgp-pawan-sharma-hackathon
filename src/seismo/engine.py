"""The core engine: one Document in; document-grain and event-grain signals out.

Stages: event classification -> entity linking -> entity-window sentiment -> credibility,
authority and lookalike checks -> story clustering (syndication, paraphrase, coordination,
24-hour novelty chains) -> denial matching -> impact. Every evidence span is an exact substring
of the source text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from seismo.config import Settings, load_settings
from seismo.nlp import clusters as clusters_mod
from seismo.nlp import denial as denial_mod
from seismo.nlp.clusters import Cluster, StoryClusterer
from seismo.nlp.credibility import credibility, is_authoritative, is_lookalike
from seismo.nlp.denial import is_denial
from seismo.nlp.events import classify_event
from seismo.nlp.impact import ImpactFeatures, PriorImpact, load_impact_model
from seismo.nlp.linker import EntityLinker
from seismo.nlp.sentiment import (
    SentimentBackend,
    average,
    confidence,
    entity_windows,
    make_backend,
)
from seismo.schemas import (
    Corroboration,
    Document,
    EntityMention,
    EventClass,
    EventLabel,
    Evidence,
    Signal,
    SourceType,
    stable_id,
)
from seismo.signals.events import EventBuilder
from seismo.universe import Universe

VERSION = "engine-0.2"
MACRO_FALLBACK = "MACRO:MARKET"
MAX_SPAN = 280


@dataclass
class EngineResult:
    doc_signals: list[Signal] = field(default_factory=list)
    event_signals: list[Signal] = field(default_factory=list)

    @property
    def all(self) -> list[Signal]:
        return [*self.doc_signals, *self.event_signals]


class Engine:
    def __init__(
        self,
        universe: Universe,
        backend: SentimentBackend,
        settings: Settings | None = None,
        impact_model: PriorImpact | None = None,
    ) -> None:
        settings = settings or load_settings()
        self.universe = universe
        self.backend = backend
        self.linker = EntityLinker(universe, float(settings.get("linker.min_link_score", 0.6)))
        self.clusters = StoryClusterer(window_hours=float(settings.get("novelty.window_hours", 24.0)))
        self.impact = impact_model or load_impact_model(settings.path("impact.model_path"))
        self.events = EventBuilder(self.impact)

    def _macro_fallback(self, doc: Document, event: EventLabel) -> list[EntityMention]:
        """Macro or geopolitical news that names no tracked entity still moves the market."""
        if event.primary not in (EventClass.MACROECONOMIC, EventClass.GEOPOLITICAL) or event.confidence < 0.5:
            return []
        if self.universe.get(MACRO_FALLBACK) is None:
            return []
        end = len(doc.title) if doc.title else len(doc.text)
        return [
            EntityMention(
                entity_id=MACRO_FALLBACK, entity_type="macro", name="US equity market",
                surface=doc.text[:end][:60], spans=[(0, end)], ner_score=0.5, ned_score=0.5,
                relevance=50, method="macro-fallback",
            )
        ]

    def versions(self, event: EventLabel) -> dict[str, str]:
        return {
            "engine": VERSION,
            "linker": EntityLinker.VERSION,
            "sentiment": self.backend.name,
            "event": event.method,
            "novelty": clusters_mod.VERSION,
            "impact": self.impact.name,
            "denial": denial_mod.VERSION,
        }

    def process(self, doc: Document) -> list[Signal]:
        """Document-grain signals only (the /v1/analyze contract)."""
        return self.process_full(doc).doc_signals

    def process_full(self, doc: Document) -> EngineResult:
        t0 = time.perf_counter()
        text = doc.text
        if not text.strip():
            return EngineResult()
        event = classify_event(doc)
        mentions = self.linker.link(doc) or self._macro_fallback(doc, event)
        if not mentions:
            return EngineResult()

        targeted = getattr(self.backend, "targeted", False)
        other_spans = {m.entity_id: [sp for o in mentions if o.entity_id != m.entity_id for sp in o.spans] for m in mentions}
        windows = [
            # A target-aware model reads the whole sentence with the entities masked; the lexicon
            # needs the sentence cut down to the entity's own clause.
            entity_windows(doc, m, others=other_spans[m.entity_id], split_clauses=not targeted)
            for m in mentions
        ]
        if targeted:
            items = [
                (text[s:e], [(a - s, b - s) for a, b in m.spans if s <= a and b <= e],
                 [(a - s, b - s) for a, b in other_spans[m.entity_id] if s <= a and b <= e])
                for m, ws in zip(mentions, windows, strict=True) for s, e in ws
            ]
            probs = self.backend.predict_targeted(items)  # type: ignore[attr-defined]
        else:
            probs = self.backend.predict([text[s:e] for ws in windows for s, e in ws])
        cred = credibility(doc.publisher, doc.source_type, doc.author)
        authoritative = is_authoritative(doc.publisher, doc.source_type)
        lookalike = is_lookalike(doc.publisher, doc.author)
        social = doc.source_type == SourceType.SOCIAL
        versions = self.versions(event)

        result = EngineResult()
        touched: dict[str, Cluster] = {}
        drafts = []
        cursor = 0
        for mention, ws in zip(mentions, windows, strict=True):
            p = average(probs[cursor:cursor + len(ws)])
            cursor += len(ws)
            score = max(-1.0, min(1.0, p[2] - p[0]))
            s0, e0 = ws[0]
            span = text[s0:e0].strip()[:MAX_SPAN] or text[:MAX_SPAN]
            entity = self.universe.get(mention.entity_id)
            flags: list[str] = []
            if lookalike:
                flags.append("lookalike_source")
            cluster_id = None
            novelty = 100
            if entity is not None and is_denial(doc, entity.domains):
                flags.append("official_denial")
                target = self.clusters.best_match(mention.entity_id, doc)
                if target is not None:
                    if self.clusters.dispute(target, doc) == "contested":
                        flags.append("denial_contested")
                    touched[target.cluster_id] = target
                    cluster_id = target.cluster_id
            else:
                cluster, member, novelty = self.clusters.assign(
                    mention.entity_id, doc, event.primary, event.confidence
                )
                cluster_id = cluster.cluster_id
                touched[cluster.cluster_id] = cluster
                if member.syndicated:
                    flags.append("syndicated_copy")
            impact, _ = self.impact.score(
                ImpactFeatures(
                    event=event.primary, sentiment=score, publishers=0 if social else 1,
                    social_authors=1 if social else 0, authoritative=authoritative,
                    novelty=novelty, relevance=mention.relevance, subtype=event.subtype,
                )
            )
            if cluster_id is not None and "official_denial" not in flags:
                member = self.clusters.get(cluster_id).members[-1]  # type: ignore[union-attr]
                member.sentiment, member.sent_conf = score, confidence(p)
                member.relevance, member.credibility = mention.relevance, cred
                member.authoritative, member.impact, member.span = authoritative, impact, span
                member.subtype, member.lookalike = event.subtype, lookalike
            drafts.append((mention, score, confidence(p), span, novelty, impact, cluster_id, flags))

        engine_ms = round((time.perf_counter() - t0) * 1000, 3)
        for mention, score, conf, span, novelty, impact, cluster_id, flags in drafts:
            result.doc_signals.append(
                Signal(
                    signal_id=stable_id("sig_doc", mention.entity_id, doc.doc_id),
                    grain="document",
                    as_of=doc.published_at,
                    entity=self.universe.ref(mention.entity_id),
                    sentiment_score=round(score, 4),
                    sentiment_confidence=round(conf, 4),
                    event=event,
                    impact_score=impact,
                    novelty=novelty,
                    relevance=mention.relevance,
                    corroboration=Corroboration(
                        independent_publishers=0 if social else 1,
                        source_types=[doc.source_type],
                        authoritative=authoritative,
                        social_authors=1 if social else 0,
                    ),
                    evidence=[
                        Evidence(
                            doc_id=doc.doc_id, publisher=doc.publisher, url=doc.url,
                            title=doc.title or None, span=span, weight=1.0,
                        )
                    ],
                    model_versions=versions,
                    latency_ms={"engine": engine_ms},
                    synthetic=doc.synthetic,
                    cluster_id=cluster_id,
                    flags=flags,
                )
            )
        for cluster in touched.values():
            result.event_signals.append(
                self.events.build(
                    cluster, self.universe.ref(cluster.entity_id), doc.published_at, versions,
                    latency_ms={"engine": engine_ms}, synthetic=doc.synthetic,
                )
            )
        return result


def build_engine(settings: Settings | None = None, backend: SentimentBackend | None = None) -> Engine:
    settings = settings or load_settings()
    universe = Universe.load(settings.path("universe.path"))
    backend = backend or make_backend(
        str(settings.get("sentiment.backend", "auto")),
        str(settings.get("sentiment.finbert_model", "ProsusAI/finbert")),
        settings.path("sentiment.target_model"),
    )
    return Engine(universe, backend, settings)
