from seismo.nlp.linker import EntityLinker
from seismo.schemas import SourceType


def _ids(mentions):
    return {m.entity_id for m in mentions}


def test_cashtags_link_including_extra_tags(universe, make_doc):
    doc = make_doc(body="$NVDA and $GOOG both moving premarket", source_type=SourceType.SOCIAL)
    assert {"NVDA", "GOOGL"} <= _ids(EntityLinker(universe).link(doc))


def test_exchange_ticker_in_parentheses(universe, make_doc):
    doc = make_doc(title="NVIDIA Corp. (NASDAQ: NVDA) reports results")
    mention = next(m for m in EntityLinker(universe).link(doc) if m.entity_id == "NVDA")
    assert mention.ner_score == 0.99 and mention.ned_score == 1.0


def test_ambiguous_name_links_with_financial_context(universe, make_doc):
    doc = make_doc(title="Apple shares rose after quarterly earnings beat estimates")
    mention = next(m for m in EntityLinker(universe).link(doc) if m.entity_id == "AAPL")
    assert mention.ned_score >= 0.75


def test_ambiguous_name_without_context_is_dropped(universe, make_doc):
    doc = make_doc(body="Apple pie recipe for the weekend, any tips?", source_type=SourceType.SOCIAL)
    assert "AAPL" not in _ids(EntityLinker(universe).link(doc))


def test_title_mention_outranks_body_mention(universe, make_doc):
    linker = EntityLinker(universe)
    in_title = linker.link(make_doc(title="Boeing faces a new probe", body="Analysts weigh the impact."))
    in_body = linker.link(make_doc(title="Aerospace roundup", body="Elsewhere, Boeing faces a new probe."))
    assert in_title[0].relevance > in_body[0].relevance


def test_macro_entities(universe, make_doc):
    doc = make_doc(title="Federal Reserve holds rates; OPEC+ extends cuts")
    assert {"MACRO:FED", "MACRO:OPEC"} <= _ids(EntityLinker(universe).link(doc))


def test_filer_cik_links_with_full_relevance(universe, make_doc):
    doc = make_doc(
        title="JPMorgan Chase & Co files 8-K: Item 2.02 Results of Operations",
        source_type=SourceType.FILING, publisher="sec.gov", meta={"cik": "0000019617", "items": ["2.02"]},
    )
    jpm = next(m for m in EntityLinker(universe).link(doc) if m.entity_id == "JPM")
    assert jpm.relevance == 100 and jpm.ned_score == 1.0


def test_overlapping_aliases_count_once(universe, make_doc):
    doc = make_doc(title="Exxon Mobil raises output")
    xom = next(m for m in EntityLinker(universe).link(doc) if m.entity_id == "XOM")
    assert len(xom.spans) == 1 and xom.surface == "Exxon Mobil"
