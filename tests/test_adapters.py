import json
from datetime import UTC, datetime

from seismo.ingest.bluesky import build_prefilter, parse_jetstream
from seismo.ingest.edgar import parse_current_atom
from seismo.ingest.gdelt import build_queries, parse_artlist
from seismo.schemas import SourceType
from tests.conftest import FIXTURES


def test_gdelt_artlist_parsing_keeps_english_only():
    payload = json.loads((FIXTURES / "gdelt_artlist.json").read_text())
    docs = parse_artlist(payload)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.source_type == SourceType.NEWS
    assert doc.publisher == "example-news.com"
    assert doc.published_at == datetime(2025, 1, 27, 13, 15, tzinfo=UTC)


def test_gdelt_queries_cover_the_universe(universe):
    queries = build_queries(universe, terms_per_query=6)
    joined = " ".join(queries)
    assert all(q.endswith("sourcelang:english") for q in queries)
    assert '"NVIDIA"' in joined and '"Federal Reserve"' in joined


def test_edgar_atom_parsing_extracts_items_and_cik(universe):
    xml = (FIXTURES / "edgar_current.atom").read_text()
    docs = parse_current_atom(xml, universe, universe_only=True)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.meta["cik"] == "0001045810"
    assert doc.meta["items"] == ["2.02", "9.01"]
    assert doc.source_type == SourceType.FILING and doc.publisher == "sec.gov"
    assert doc.published_at == datetime(2025, 1, 27, 21, 5, 12, tzinfo=UTC)
    assert "Item 2.02" in doc.title


def test_edgar_parsing_skips_non_8k_forms(universe):
    xml = (FIXTURES / "edgar_current.atom").read_text()
    docs = parse_current_atom(xml, universe, universe_only=False)
    assert sorted(d.meta["form"] for d in docs) == ["8-K", "8-K"]
    assert any(d.meta["items"] == ["1.03"] for d in docs)


def test_jetstream_filters(universe):
    prefilter = build_prefilter(universe)
    messages = [json.loads(line) for line in (FIXTURES / "jetstream.jsonl").read_text().splitlines() if line.strip()]
    results = [parse_jetstream(m, prefilter) for m in messages]
    kept = [r for r in results if r is not None]
    assert len(kept) == 1
    assert "$NVDA" in kept[0].body
    assert kept[0].source_type == SourceType.SOCIAL
    assert kept[0].url.startswith("https://bsky.app/profile/")


def test_gdelt_headlines_are_rebuilt_from_url_slugs():
    from seismo.ingest.gdelt_replay import headline_from_url

    acr = {"SVB", "FDIC"}
    assert headline_from_url("https://www.cnbc.com/2023/03/10/silicon-valley-bank-collapse-fdic-takes-over.html", acr) == \
        "Silicon Valley Bank Collapse FDIC Takes over"
    assert headline_from_url("https://economictimes.indiatimes.com/markets/stocks/news/adani-group-stocks-fall-after-"
                             "hindenburg-report/articleshow/97279870.cms") == "Adani Group Stocks Fall after Hindenburg Report"
    assert headline_from_url("https://example.com/?p=12345") is None
