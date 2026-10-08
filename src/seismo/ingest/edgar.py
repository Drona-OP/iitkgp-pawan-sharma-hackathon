"""SEC EDGAR current 8-K feed. Free; SEC requires a descriptive User-Agent and at most 10 req/s.

Item codes become authoritative event labels: 1.03 bankruptcy, 1.05 cyber incident,
2.02 results, 5.02 officer change, and so on (see nlp/events.py).
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator
from datetime import datetime

import httpx

from seismo.ingest.base import Adapter
from seismo.schemas import Document, SourceType, stable_id
from seismo.universe import Universe

log = logging.getLogger(__name__)

EDGAR_CURRENT_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
TITLE_RE = re.compile(r"^(?P<form>\S+) - (?P<company>.+?) \((?P<cik>\d{10})\)")
ITEM_RE = re.compile(r"Item\s+(\d\.\d{2})\s*:\s*([^<\n]+)")
ACCESSION_RE = re.compile(r"accession-number=([\d-]+)")
XML_DECL_RE = re.compile(r"^\s*<\?xml[^>]*\?>")
PLACEHOLDER_UA = "contact@example.com"


def parse_current_atom(
    xml_text: str, universe: Universe | None = None, universe_only: bool = True
) -> list[Document]:
    root = ET.fromstring(XML_DECL_RE.sub("", xml_text))
    docs: list[Document] = []
    for entry in root.findall("a:entry", ATOM):
        m = TITLE_RE.match(entry.findtext("a:title", default="", namespaces=ATOM).strip())
        if not m or not m["form"].upper().startswith("8-K"):
            continue
        cik = m["cik"]
        if universe_only and universe is not None and universe.by_cik(cik) is None:
            continue
        summary = html.unescape(entry.findtext("a:summary", default="", namespaces=ATOM))
        items = [(code, desc.strip()) for code, desc in ITEM_RE.findall(summary)]
        updated = entry.findtext("a:updated", default="", namespaces=ATOM)
        try:
            published = datetime.fromisoformat(updated)
        except ValueError:
            continue
        link_el = entry.find("a:link", ATOM)
        link = link_el.get("href") if link_el is not None else None
        acc_match = ACCESSION_RE.search(entry.findtext("a:id", default="", namespaces=ATOM))
        accession = acc_match.group(1) if acc_match else (link or "")
        company = m["company"].strip()
        item_text = "; ".join(f"Item {code} {desc}" for code, desc in items)
        title = f"{company} files {m['form']}: {item_text}" if item_text else f"{company} files {m['form']}"
        docs.append(
            Document(
                doc_id=stable_id("d", "edgar", accession),
                source="edgar",
                source_type=SourceType.FILING,
                publisher="sec.gov",
                url=link,
                published_at=published,
                title=title,
                meta={"cik": cik, "form": m["form"], "items": [c for c, _ in items], "accession": accession},
            )
        )
    return docs


class EdgarAdapter(Adapter):
    name = "edgar"

    def __init__(self, universe: Universe, user_agent: str, poll_seconds: float = 60, universe_only: bool = True) -> None:
        self.universe = universe
        self.user_agent = user_agent
        self.poll_seconds = poll_seconds
        self.universe_only = universe_only
        self._seen: set[str] = set()
        if PLACEHOLDER_UA in user_agent:
            log.warning("Set SEISMO_EDGAR_USER_AGENT to 'Your Name your@email' - SEC blocks anonymous clients.")

    async def stream(self) -> AsyncIterator[Document]:
        params = {
            "action": "getcurrent", "type": "8-K", "company": "", "dateb": "", "owner": "include",
            "start": "0", "count": "100", "output": "atom",
        }
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            while True:
                try:
                    resp = await client.get(EDGAR_CURRENT_URL, params=params)
                    resp.raise_for_status()
                    docs = parse_current_atom(resp.text, self.universe, self.universe_only)
                except (httpx.HTTPError, ET.ParseError) as exc:
                    log.warning("EDGAR poll failed: %s", exc)
                    docs = []
                for doc in sorted(docs, key=lambda d: d.published_at):
                    if doc.doc_id in self._seen:
                        continue
                    self._seen.add(doc.doc_id)
                    yield doc
                await asyncio.sleep(self.poll_seconds)
