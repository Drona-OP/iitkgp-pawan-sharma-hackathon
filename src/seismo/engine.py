"""The core engine: one Document in, one document-grain Signal per linked entity out.

Stages: event classification -> entity linking -> entity-window sentiment -> credibility and
authority -> novelty -> impact. Every evidence span is an exact substring of the source text.
"""

from __future__ import annotations

import time

from seismo.config import Settings, load_settings
from seismo.nlp import impact as impact_mod
from seismo.nlp import novelty as novelty_mod
from seismo.nlp.credibility import credibility, is_authoritative
from seismo.nlp.events import classify_event
from seismo.nlp.impact import heuristic_impact
from seismo.nlp.linker import EntityLinker
from seismo.nlp.novelty import NoveltyTracker
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
    stable_id,
)
from seismo.universe import Universe

VERSION = "engine-0.1"
MACRO_FALLBACK = "MACRO:MARKET"
MAX_SPAN = 280


class Engine:
    def __init__(self, universe: Universe, backend: SentimentBackend, settings: Settings | None = None) -> None:
        settings = settings or load_settings()
        self.universe = universe
        self.backend = backend
        self.linker = EntityLinker(universe, float(settings.get("linker.min_link_score", 0.6)))
        self.novelty = NoveltyTracker(window_hours=float(settings.get("novelty.window_hours", 24.0)))

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

    def process(self, doc: Document) -> list[Signal]:
        t0 = time.perf_counter()
        text = doc.text
        if not text.strip():
            return []
        event = classify_event(doc)
        mentions = self.linker.link(doc) or self._macro_fallback(doc, event)
        if not mentions:
            return []

        windows = [entity_windows(doc, m) for m in mentions]
        flat = [text[s:e] for ws in windows for s, e in ws]
        probs = self.backend.predict(flat)
        cred = credibility(doc.publisher, doc.source_type)
        authoritative = is_authoritative(doc.publisher, doc.source_type)

        drafts = []
        cursor = 0
        for mention, ws in zip(mentions, windows, strict=True):
            p = average(probs[cursor:cursor + len(ws)])
            cursor += len(ws)
            score = max(-1.0, min(1.0, p[2] - p[0]))
            s0, e0 = ws[0]
            span = text[s0:e0].strip()[:MAX_SPAN] or text[:MAX_SPAN]
            novelty = self.novelty.observe(mention.entity_id, doc)
            impact, _ = heuristic_impact(event.primary, score, cred, mention.relevance, novelty)
            drafts.append((mention, score, confidence(p), span, novelty, impact))

        engine_ms = round((time.perf_counter() - t0) * 1000, 3)
        versions = {
            "engine": VERSION,
            "linker": EntityLinker.VERSION,
            "sentiment": self.backend.name,
            "event": event.method,
            "novelty": novelty_mod.VERSION,
            "impact": impact_mod.VERSION,
        }
        signals = []
        for mention, score, conf, span, novelty, impact in drafts:
            signals.append(
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
                        independent_publishers=1,
                        source_types=[doc.source_type],
                        authoritative=authoritative,
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
                )
            )
        return signals


def build_engine(settings: Settings | None = None, backend: SentimentBackend | None = None) -> Engine:
    settings = settings or load_settings()
    universe = Universe.load(settings.path("universe.path"))
    backend = backend or make_backend(
        str(settings.get("sentiment.backend", "auto")),
        str(settings.get("sentiment.finbert_model", "ProsusAI/finbert")),
    )
    return Engine(universe, backend, settings)
