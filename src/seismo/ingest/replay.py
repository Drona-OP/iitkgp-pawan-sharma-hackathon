"""Replay packs: append-only JSONL of Documents, re-emitted on an accelerated clock.

Live runs record what they ingest; replays make demos, tests and backtests deterministic.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from seismo.ingest.base import Adapter
from seismo.schemas import Document


def load_pack(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", "//")):
                continue
            docs.append(Document.model_validate_json(line))
    return sorted(docs, key=lambda d: (d.published_at, d.doc_id))


class Recorder:
    """Appends every ingested Document to a JSONL replay pack."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, doc: Document) -> None:
        self._fh.write(doc.model_dump_json() + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class ReplayAdapter(Adapter):
    name = "replay"

    def __init__(self, path: str | Path, speed: float = 600.0, max_gap_seconds: float = 2.0) -> None:
        self.path = Path(path)
        self.speed = speed  # <= 0 means as fast as possible
        self.max_gap_seconds = max_gap_seconds

    async def stream(self) -> AsyncIterator[Document]:
        previous = None
        for doc in load_pack(self.path):
            if previous is not None and self.speed > 0:
                gap = (doc.published_at - previous).total_seconds() / self.speed
                await asyncio.sleep(min(max(gap, 0.0), self.max_gap_seconds))
            else:
                await asyncio.sleep(0)
            previous = doc.published_at
            # In replay, ingestion time is event time, so outputs are identical run after run.
            yield doc.model_copy(update={"ingested_at": doc.published_at})
