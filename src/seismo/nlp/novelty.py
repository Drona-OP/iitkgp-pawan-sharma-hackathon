"""Novelty v0, after RavenPack's 24-hour chains.

The first story about an entity scores 100; repeats inside 24 hours decay as 100 / (1 + n);
a gap longer than 24 hours starts a new chain. v0 matches stories by a title fingerprint,
which catches syndicated copies. Day 2 adds MinHash and embedding clusters for paraphrases.
"""

from __future__ import annotations

import re
from datetime import timedelta

from seismo.schemas import Document

VERSION = "title-fingerprint-v0"
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "as", "at", "by", "with",
     "its", "is", "are", "after", "over", "from", "this", "that", "it", "be"}
)


def fingerprint(doc: Document) -> str:
    source = doc.title or doc.body
    tokens = [t for t in TOKEN_RE.findall(source.lower()) if t not in STOPWORDS]
    return " ".join(tokens[:10])


class NoveltyTracker:
    def __init__(self, window_hours: float = 24.0) -> None:
        self.window = timedelta(hours=window_hours)
        self._seen: dict[tuple[str, str], list] = {}

    def observe(self, entity_id: str, doc: Document) -> int:
        key = (entity_id, fingerprint(doc))
        now = doc.published_at
        recent = [t for t in self._seen.get(key, []) if timedelta(0) <= now - t <= self.window]
        score = round(100 / (1 + len(recent)))
        recent.append(now)
        self._seen[key] = recent
        return int(score)
