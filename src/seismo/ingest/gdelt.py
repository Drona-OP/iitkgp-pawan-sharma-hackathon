"""GDELT DOC 2.0 API: keyless global news, updated every 15 minutes, rolling 3-month search window."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx

from seismo.ingest.base import Adapter
from seismo.schemas import Document, SourceType, stable_id
from seismo.universe import Universe

log = logging.getLogger(__name__)

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MACRO_TERMS = ("Federal Reserve", "tariffs", "sanctions", "bank run", "OPEC", "Reserve Bank of India")


def build_queries(universe: Universe, terms_per_query: int = 6) -> list[str]:
    """One query per chunk of terms: company names (first unambiguous alias) plus macro themes."""
    terms: list[str] = []
    for company in universe.companies():
        alias = company.aliases[0] if company.aliases else company.name
        terms.append(alias)
    terms.extend(MACRO_TERMS)
    queries = []
    for i in range(0, len(terms), terms_per_query):
        chunk = " OR ".join(f'"{t}"' for t in terms[i:i + terms_per_query])
        queries.append(f"({chunk}) sourcelang:english")
    return queries


def parse_artlist(payload: dict[str, Any]) -> list[Document]:
    docs: list[Document] = []
    for art in payload.get("articles", []) or []:
        url = art.get("url")
        title = " ".join((art.get("title") or "").split())
        if not url or not title:
            continue
        if (art.get("language") or "English").lower() != "english":
            continue
        try:
            published = datetime.strptime(art["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except (KeyError, ValueError):
            continue
        publisher = art.get("domain") or urlparse(url).netloc
        docs.append(
            Document(
                doc_id=stable_id("d", "gdelt", url),
                source="gdelt",
                source_type=SourceType.NEWS,
                publisher=publisher,
                url=url,
                published_at=published,
                title=title,
                meta={"sourcecountry": art.get("sourcecountry")},
            )
        )
    return docs


class GdeltAdapter(Adapter):
    name = "gdelt"

    def __init__(
        self,
        universe: Universe,
        poll_seconds: float = 900,
        timespan: str = "1h",
        max_records: int = 250,
        terms_per_query: int = 6,
        pause_seconds: float = 6,
    ) -> None:
        self.queries = build_queries(universe, terms_per_query)
        self.poll_seconds = poll_seconds
        self.timespan = timespan
        self.max_records = max_records
        self.pause_seconds = pause_seconds
        self._seen: set[str] = set()

    async def _fetch(self, client: httpx.AsyncClient, query: str) -> list[Document]:
        params = {
            "query": query, "mode": "ArtList", "format": "json", "sort": "DateDesc",
            "maxrecords": str(self.max_records), "timespan": self.timespan,
        }
        resp = await client.get(GDELT_DOC_URL, params=params)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError:  # GDELT answers some bad queries with plain text
            log.warning("GDELT returned non-JSON for %r: %s", query, resp.text[:120])
            return []
        return parse_artlist(payload)

    async def stream(self) -> AsyncIterator[Document]:
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "seismo/0.1"}) as client:
            while True:
                batch: list[Document] = []
                for query in self.queries:
                    try:
                        batch.extend(await self._fetch(client, query))
                    except httpx.HTTPError as exc:
                        log.warning("GDELT request failed: %s", exc)
                    await asyncio.sleep(self.pause_seconds)
                for doc in sorted(batch, key=lambda d: d.published_at):
                    if doc.url in self._seen:
                        continue
                    self._seen.add(doc.url or doc.doc_id)
                    yield doc
                await asyncio.sleep(self.poll_seconds)
