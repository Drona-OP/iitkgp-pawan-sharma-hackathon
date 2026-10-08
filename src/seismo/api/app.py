"""Signal API: REST for queries, POST /v1/analyze for ad-hoc text, a WebSocket for the live stream.

OpenAPI docs are served at /docs.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from seismo import __version__
from seismo.config import load_settings
from seismo.engine import Engine, build_engine
from seismo.schemas import Document, Signal, SourceType, stable_id
from seismo.store.sqlite import SQLiteStore


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    title: str = ""
    source_type: SourceType = SourceType.NEWS
    publisher: str = "user-input"


def create_app(store: SQLiteStore | None = None, engine: Engine | None = None) -> FastAPI:
    app = FastAPI(
        title="Seismo signal API",
        version=__version__,
        description="Structured, explainable risk signals from news, filings and social posts.",
    )
    state: dict[str, object] = {"store": store, "engine": engine}

    def get_store() -> SQLiteStore:
        if state["store"] is None:
            state["store"] = SQLiteStore(load_settings().path("store.path"))
        return state["store"]  # type: ignore[return-value]

    def get_engine() -> Engine:
        if state["engine"] is None:
            state["engine"] = build_engine(load_settings())
        return state["engine"]  # type: ignore[return-value]

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "version": __version__, **get_store().stats()}

    @app.get("/v1/signals", response_model=list[Signal])
    def list_signals(
        grain: str | None = Query(None, pattern="^(document|event|entity)$"),
        entity: str | None = None,
        min_impact: int | None = Query(None, ge=1, le=10),
        limit: int = Query(100, ge=1, le=1000),
    ) -> list[Signal]:
        return get_store().signals(grain=grain, entity=entity, min_impact=min_impact, limit=limit)

    @app.get("/v1/entities", response_model=list[Signal])
    def latest_entities() -> list[Signal]:
        return get_store().latest_entities()

    @app.get("/v1/documents/{doc_id}", response_model=Document)
    def get_document(doc_id: str) -> Document:
        doc = get_store().document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"No document {doc_id}")
        return doc

    @app.post("/v1/analyze", response_model=list[Signal])
    def analyze(req: AnalyzeRequest) -> list[Signal]:
        now = datetime.now(UTC)
        doc = Document(
            doc_id=stable_id("d", "api", req.title, req.text, now.isoformat()),
            source="api", source_type=req.source_type, publisher=req.publisher,
            published_at=now, title=req.title, body=req.text,
        )
        return get_engine().process(doc)

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket, since: int | None = None) -> None:
        """Pushes every new signal as JSON. ?since=0 replays everything already stored."""
        await ws.accept()
        store_ = get_store()
        cursor = store_.max_seq() if since is None else since
        try:
            while True:
                for seq, sig in store_.signals_after(cursor):
                    await ws.send_text(sig.model_dump_json())
                    cursor = seq
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            return

    return app


app = create_app()
