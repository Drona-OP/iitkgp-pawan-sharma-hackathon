"""Bluesky Jetstream: the public, keyless, real-time post stream (JSON over WebSocket).

Posts pass cheap filters before entering the pipeline: English, a minimum length, a cashtag
cap against pump spam, a per-author rate cap, and a keyword prefilter built from the universe.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from seismo.ingest.base import Adapter
from seismo.schemas import Document, SourceType, stable_id
from seismo.universe import Universe

log = logging.getLogger(__name__)

CASHTAG_RE = re.compile(r"\$[A-Za-z]{1,5}\b")


def build_prefilter(universe: Universe) -> re.Pattern[str]:
    terms = {t.lower() for e in universe.entities for t in (*e.aliases, *e.ambiguous_aliases)}
    alternation = "|".join(sorted((re.escape(t) for t in terms), key=len, reverse=True))
    tags = "|".join(re.escape(tag) for e in universe.companies() for tag in e.cashtags)
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)|\$(?:{tags})\b", re.IGNORECASE)


def _created_at(record: dict[str, Any], time_us: int | None) -> datetime:
    raw = record.get("createdAt")
    if raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    if time_us:
        return datetime.fromtimestamp(time_us / 1_000_000, tz=UTC)
    return datetime.now(UTC)


def parse_jetstream(
    msg: dict[str, Any], prefilter: re.Pattern[str], max_cashtags: int = 5, min_chars: int = 20
) -> Document | None:
    if msg.get("kind") != "commit":
        return None
    commit = msg.get("commit") or {}
    if commit.get("operation") != "create" or commit.get("collection") != "app.bsky.feed.post":
        return None
    record = commit.get("record") or {}
    text = " ".join((record.get("text") or "").split())
    langs = record.get("langs") or []
    if langs and not any(lang.split("-")[0].lower() == "en" for lang in langs):
        return None
    if len(text) < min_chars or len(CASHTAG_RE.findall(text)) > max_cashtags:
        return None
    if not prefilter.search(text):
        return None
    did, rkey = msg.get("did", ""), commit.get("rkey", "")
    return Document(
        doc_id=stable_id("d", "bluesky", did, rkey),
        source="bluesky",
        source_type=SourceType.SOCIAL,
        publisher="bsky.app",
        author=did,
        url=f"https://bsky.app/profile/{did}/post/{rkey}",
        published_at=_created_at(record, msg.get("time_us")),
        body=text,
        meta={"cid": commit.get("cid")},
    )


class BlueskyAdapter(Adapter):
    name = "bluesky"

    def __init__(
        self,
        universe: Universe,
        endpoint: str,
        max_cashtags: int = 5,
        min_chars: int = 20,
        max_posts_per_author_hour: int = 5,
    ) -> None:
        self.endpoint = endpoint
        self.prefilter = build_prefilter(universe)
        self.max_cashtags = max_cashtags
        self.min_chars = min_chars
        self.max_per_author = max_posts_per_author_hour
        self._author_times: dict[str, deque] = defaultdict(deque)
        self._cursor: int | None = None

    def _rate_ok(self, doc: Document) -> bool:
        times = self._author_times[doc.author or ""]
        now = doc.published_at.timestamp()
        while times and now - times[0] > 3600:
            times.popleft()
        if len(times) >= self.max_per_author:
            return False
        times.append(now)
        return True

    async def stream(self) -> AsyncIterator[Document]:
        import websockets  # local import keeps replay-only installs light

        backoff = 1.0
        while True:
            url = f"{self.endpoint}?wantedCollections=app.bsky.feed.post"
            if self._cursor:
                url += f"&cursor={self._cursor}"
            try:
                async with websockets.connect(url, ping_interval=20, max_size=2**20) as ws:
                    backoff = 1.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        self._cursor = msg.get("time_us") or self._cursor
                        doc = parse_jetstream(msg, self.prefilter, self.max_cashtags, self.min_chars)
                        if doc is not None and self._rate_ok(doc):
                            yield doc
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any network failure
                log.warning("Jetstream disconnected (%s); reconnecting in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
