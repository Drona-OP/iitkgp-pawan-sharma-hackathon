"""Entity recognition and linking in the style of Kensho NERD.

Each linked entity carries the surface text, its spans, a recognition score (is this span an
entity?) and a linking score (is it *this* entity?). Ambiguous names such as "Apple" or
"Amazon" only link when the surrounding text looks financial or other entities are present.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from seismo.schemas import Document, EntityMention, SourceType
from seismo.universe import Entity, Universe

CASHTAG_RE = re.compile(r"(?<![\w$])\$([A-Za-z]{1,5})\b")
EXCHANGE_TICKER_RE = re.compile(
    r"\((?:NASDAQ|Nasdaq|NYSE|NYSE American|NSE|BSE)\s*:\s*([A-Z][A-Z.&]{0,11})\)"
)
WORD_RE = re.compile(r"[a-z][a-z'\-]*")

FINANCE_CONTEXT = frozenset(
    {
        "shares", "share", "stock", "stocks", "earnings", "revenue", "revenues", "profit",
        "profits", "quarter", "quarterly", "guidance", "outlook", "dividend", "buyback",
        "investors", "investor", "analyst", "analysts", "market", "markets", "valuation",
        "ceo", "cfo", "inc", "corp", "company", "nasdaq", "nyse", "sec", "filing", "ipo",
        "acquire", "acquisition", "merger", "deal", "billion", "million", "trillion",
        "rating", "downgrade", "upgrade", "target", "targets", "results", "sales", "demand",
        "regulators", "regulator", "antitrust", "lawsuit", "fined", "probe", "unveils",
        "iphone", "cloud", "chips", "chip", "subscribers", "layoffs", "factory", "supplier",
        "suppliers", "premarket", "trading", "traders", "earnings-call",
    }
)

STRONG_KINDS = {"cashtag", "exchange", "filer"}


@dataclass
class _Hit:
    entity: Entity
    start: int
    end: int
    kind: str  # cashtag | exchange | filer | alias | ambiguous


def _alias_pattern(alias: str) -> re.Pattern[str]:
    flags = 0 if any(ch.isupper() for ch in alias) else re.IGNORECASE
    return re.compile(rf"(?<![\w$]){re.escape(alias)}(?!\w)", flags)


def _dedupe_overlaps(hits: list[_Hit]) -> list[_Hit]:
    """Keep the longest hit wherever two hits of one entity overlap ("Exxon" in "Exxon Mobil")."""
    kept: list[_Hit] = []
    for hit in sorted(hits, key=lambda h: (h.start, -(h.end - h.start))):
        if kept and hit.start < kept[-1].end and hit.end > kept[-1].start:
            if hit.kind in STRONG_KINDS and kept[-1].kind not in STRONG_KINDS:
                kept[-1] = hit
            continue
        kept.append(hit)
    return kept


class EntityLinker:
    VERSION = "alias-context-v0"

    def __init__(self, universe: Universe, min_link_score: float = 0.6) -> None:
        self.universe = universe
        self.min_link_score = min_link_score
        self._patterns: list[tuple[Entity, re.Pattern[str], bool]] = []
        for entity in universe.entities:
            for alias in entity.aliases:
                self._patterns.append((entity, _alias_pattern(alias), False))
            for alias in entity.ambiguous_aliases:
                self._patterns.append((entity, _alias_pattern(alias), True))

    def _find_hits(self, doc: Document, text: str) -> list[_Hit]:
        hits: list[_Hit] = []
        cik = doc.meta.get("cik")
        if doc.source_type == SourceType.FILING and cik:
            filer = self.universe.by_cik(cik)
            if filer is not None:
                hits.append(_Hit(filer, 0, len(doc.title), "filer"))
        for m in CASHTAG_RE.finditer(text):
            entity = self.universe.by_cashtag(m.group(1))
            if entity is not None:
                hits.append(_Hit(entity, m.start(), m.end(), "cashtag"))
        for m in EXCHANGE_TICKER_RE.finditer(text):
            entity = self.universe.by_cashtag(m.group(1))
            if entity is not None:
                hits.append(_Hit(entity, m.start(), m.end(), "exchange"))
        for entity, pattern, ambiguous in self._patterns:
            for m in pattern.finditer(text):
                hits.append(_Hit(entity, m.start(), m.end(), "ambiguous" if ambiguous else "alias"))
        return hits

    @staticmethod
    def _ambiguity_score(text: str, hit: _Hit, has_other_entities: bool) -> float:
        window = text[max(0, hit.start - 160): hit.end + 160].lower()
        context = len(set(WORD_RE.findall(window)) & FINANCE_CONTEXT)
        score = 0.35
        if context >= 1:
            score = 0.75
        if context >= 2:
            score = 0.9
        if has_other_entities:
            score = max(score, 0.8)
        return score

    @staticmethod
    def _relevance(doc: Document, text: str, hits: list[_Hit]) -> int:
        if any(h.kind == "filer" for h in hits):
            return 100
        title_len = len(doc.title)
        in_title = bool(doc.title) and any(h.start < title_len for h in hits)
        first_pos = min(h.start for h in hits) / max(1, len(text))
        count = len(hits)
        if doc.source_type == SourceType.SOCIAL:
            rel = 70 + 10 * min(count - 1, 2) + (10 if any(h.kind == "cashtag" for h in hits) else 0)
        else:
            rel = 20 + (45 if in_title else 0) + 20 * (1 - first_pos) + 5 * min(count, 3)
        return int(max(0, min(100, round(rel))))

    @staticmethod
    def _drop_nested(hits: list[_Hit]) -> list[_Hit]:
        """Across entities, the longest name wins: "Adani" inside "Adani Ports" is Adani Ports."""
        names = [h for h in hits if h.kind in ("alias", "ambiguous")]
        keep = []
        for h in hits:
            if h.kind in ("alias", "ambiguous") and any(
                o.entity is not h.entity and o.start <= h.start and h.end <= o.end and (o.end - o.start) > (h.end - h.start)
                for o in names
            ):
                continue
            keep.append(h)
        return keep

    def link(self, doc: Document) -> list[EntityMention]:
        text = doc.text
        hits = self._drop_nested(self._find_hits(doc, text))
        if not hits:
            return []
        grouped: dict[str, list[_Hit]] = defaultdict(list)
        for hit in hits:
            grouped[hit.entity.entity_id].append(hit)
        confident = {eid for eid, hs in grouped.items() if any(h.kind != "ambiguous" for h in hs)}
        mentions: list[EntityMention] = []
        for entity_id, entity_hits in grouped.items():
            entity_hits = _dedupe_overlaps(entity_hits)
            kinds = {h.kind for h in entity_hits}
            if kinds == {"ambiguous"}:
                others = bool(confident - {entity_id})
                ner = 0.8
                ned = max(self._ambiguity_score(text, h, others) for h in entity_hits)
            elif kinds & STRONG_KINDS:
                ner, ned = 0.99, 1.0
            else:
                ner, ned = 0.95, 0.95
            if ned < self.min_link_score:
                continue
            first = min(entity_hits, key=lambda h: h.start)
            surface = text[first.start:first.end] if first.end > first.start else first.entity.name
            mentions.append(
                EntityMention(
                    entity_id=entity_id,
                    entity_type=first.entity.entity_type,  # type: ignore[arg-type]
                    name=first.entity.name,
                    surface=surface,
                    spans=[(h.start, h.end) for h in sorted(entity_hits, key=lambda h: h.start)],
                    ner_score=ner,
                    ned_score=round(ned, 3),
                    relevance=self._relevance(doc, text, entity_hits),
                    method=self.VERSION,
                )
            )
        mentions.sort(key=lambda m: (-m.relevance, m.entity_id))
        return mentions
