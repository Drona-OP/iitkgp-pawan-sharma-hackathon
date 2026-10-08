import asyncio

from fastapi.testclient import TestClient

from seismo.api.app import create_app
from seismo.bus import InMemoryBus
from seismo.engine import Engine
from seismo.ingest.replay import ReplayAdapter, load_pack
from seismo.nlp.sentiment import LexiconBackend
from seismo.runner import run_pipeline
from seismo.signals.aggregator import EntityAggregator
from seismo.store.sqlite import SQLiteStore
from tests.conftest import DEMO_PACK


def _replay(tmp_path, universe, settings, name="a.db"):
    store = SQLiteStore(tmp_path / name)
    engine = Engine(universe, LexiconBackend(), settings)
    counts = asyncio.run(
        run_pipeline([ReplayAdapter(DEMO_PACK, speed=0)], engine, EntityAggregator(), store)
    )
    return store, counts


def test_bus_preserves_order_and_ends():
    async def scenario():
        bus = InMemoryBus()
        sub = bus.subscribe("t")
        for i in range(3):
            await bus.publish("t", None, i)
        await bus.end("t")
        return [v async for _, v in sub]

    assert asyncio.run(scenario()) == [0, 1, 2]


def test_replay_smoke(tmp_path, universe, settings):
    store, counts = _replay(tmp_path, universe, settings)
    docs = load_pack(DEMO_PACK)
    assert counts["documents"] == len(docs) == 35
    assert counts["signals"] >= 30
    doc_signals = store.signals(grain="document", limit=1000)
    assert store.latest_entities(), "entity-grain signals should exist"
    texts = {d.doc_id: d.text for d in docs}
    for sig in doc_signals:
        assert sig.synthetic
        for ev in sig.evidence:
            assert ev.span in texts[ev.doc_id]
    stats = store.stats()
    assert stats["documents"] == 35 and stats["signals"] == len(doc_signals)


def test_replay_is_deterministic(tmp_path, universe, settings):
    a, _ = _replay(tmp_path, universe, settings, "a.db")
    b, _ = _replay(tmp_path, universe, settings, "b.db")
    key = lambda s: (s.signal_id, s.impact_score, round(s.sentiment_score, 4))  # noqa: E731
    assert sorted(map(key, a.signals(limit=5000))) == sorted(map(key, b.signals(limit=5000)))


def test_syndicated_nvidia_story_is_one_novel_and_two_repeats(tmp_path, universe, settings):
    store, _ = _replay(tmp_path, universe, settings)
    novelty = sorted(
        s.novelty for s in store.signals(grain="document", entity="NVDA", limit=50)
        if s.evidence and s.evidence[0].doc_id in {"syn_001", "syn_002", "syn_003"}
    )
    assert novelty == [33, 50, 100]


def test_api_endpoints(tmp_path, universe, settings):
    store, _ = _replay(tmp_path, universe, settings)
    client = TestClient(create_app(store=store, engine=Engine(universe, LexiconBackend(), settings)))

    health = client.get("/health").json()
    assert health["status"] == "ok" and health["documents"] == 35

    severe_or_more = client.get("/v1/signals", params={"grain": "document", "min_impact": 1, "limit": 5}).json()
    assert len(severe_or_more) == 5

    assert client.get("/v1/entities").json()
    assert client.get("/v1/documents/syn_001").json()["doc_id"] == "syn_001"
    assert client.get("/v1/documents/nope").status_code == 404

    out = client.post("/v1/analyze", json={"text": "Tesla recalls 200,000 vehicles over a software issue"}).json()
    assert out[0]["entity"]["id"] == "TSLA" and out[0]["sentiment_score"] < 0

    with client.websocket_connect("/v1/stream?since=0") as ws:
        first = ws.receive_json()
        assert "signal_id" in first
