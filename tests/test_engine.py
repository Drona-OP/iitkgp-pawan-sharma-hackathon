from datetime import UTC, datetime, timedelta

from seismo.schemas import EventClass, SourceType


def test_competitive_shock_is_negative_for_nvidia(engine, make_doc):
    doc = make_doc(title="Nvidia shares plunge 17% as a rival unveils a cheaper AI model")
    nvda = next(s for s in engine.process(doc) if s.entity.id == "NVDA")
    assert nvda.sentiment_score < 0
    assert nvda.event.primary == EventClass.PRODUCT_STRATEGY
    assert 1 <= nvda.impact_score <= 10
    assert nvda.grain == "document"


def test_every_evidence_span_is_an_exact_substring(engine, make_doc):
    doc = make_doc(
        title="Boeing faces a fresh FAA probe after a 737 MAX grounding",
        body="Regulators opened an investigation. Airbus shares rose.",
    )
    for sig in engine.process(doc):
        for ev in sig.evidence:
            assert ev.span and ev.span in doc.text


def test_ambiguous_social_post_produces_no_company_signal(engine, make_doc):
    doc = make_doc(body="Apple pie recipe for the weekend, any tips?", source_type=SourceType.SOCIAL)
    assert all(s.entity.id != "AAPL" for s in engine.process(doc))


def test_filing_signal_is_authoritative(engine, make_doc):
    doc = make_doc(
        title="JPMorgan Chase & Co files 8-K: Item 2.02 Results of Operations and Financial Condition",
        source_type=SourceType.FILING, publisher="sec.gov", meta={"cik": "0000019617", "items": ["2.02"]},
    )
    jpm = next(s for s in engine.process(doc) if s.entity.id == "JPM")
    assert jpm.corroboration.authoritative
    assert jpm.event.primary == EventClass.EARNINGS_GUIDANCE
    assert jpm.relevance == 100


def test_unnamed_geopolitical_news_falls_back_to_the_market(engine, make_doc):
    sigs = engine.process(make_doc(title="Missile strikes escalate the conflict overnight"))
    assert [s.entity.id for s in sigs] == ["MACRO:MARKET"]


def test_syndicated_copies_lose_novelty(engine, make_doc):
    t0 = datetime(2026, 9, 14, 6, 0, tzinfo=UTC)
    title = "Nvidia shares slide after a rival releases a cheaper model"
    first = engine.process(make_doc(title=title, published_at=t0))[0]
    second = engine.process(make_doc(title=title, publisher="other.example", published_at=t0 + timedelta(minutes=3)))[0]
    later = engine.process(make_doc(title=title, published_at=t0 + timedelta(hours=30)))[0]
    assert first.novelty == 100 and second.novelty == 50 and later.novelty == 100


def test_social_rumour_alone_stays_below_the_stress_trigger(engine, make_doc):
    doc = make_doc(body="BofA halting withdrawals is how a bank run starts, sell everything",
                   source_type=SourceType.SOCIAL, publisher="bsky.app")
    bac = next(s for s in engine.process(doc) if s.entity.id == "BAC")
    assert bac.event.primary == EventClass.CREDIT_EVENT
    assert bac.impact_score < 8
    assert not bac.corroboration.authoritative
