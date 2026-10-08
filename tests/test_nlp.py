from seismo.nlp.credibility import credibility, is_authoritative
from seismo.nlp.events import classify_event
from seismo.nlp.impact import heuristic_impact
from seismo.nlp.sentiment import LexiconBackend, confidence, entity_windows
from seismo.schemas import EntityMention, EventClass, SourceType


def test_lexicon_polarity_and_negation():
    lex = LexiconBackend()
    pos, neg, negated, neutral = lex.predict(
        ["Shares surge after record results", "Shares plunge after weak guidance",
         "The company denies any default", "The meeting is on Tuesday"]
    )
    assert pos[2] > pos[0]
    assert neg[0] > neg[2]
    assert negated[2] > negated[0]
    assert neutral[1] > 0.5


def test_confidence_bounds():
    assert confidence((0.0, 0.0, 1.0)) == 1.0
    assert abs(confidence((1 / 3, 1 / 3, 1 / 3))) < 1e-9


def test_entity_window_is_the_mentioning_sentence(make_doc):
    doc = make_doc(title="Markets wrap", body="Oil rose. Tesla recalls vehicles over software. Gold fell.")
    start = doc.text.index("Tesla")
    mention = EntityMention(entity_id="TSLA", entity_type="company", name="Tesla", surface="Tesla",
                            spans=[(start, start + 5)], ner_score=0.95, ned_score=0.95, relevance=50, method="t")
    (s, e), = entity_windows(doc, mention)
    assert doc.text[s:e].strip() == "Tesla recalls vehicles over software."


def test_8k_items_drive_the_event_label(make_doc):
    doc = make_doc(title="Example Corp files 8-K", source_type=SourceType.FILING,
                   publisher="sec.gov", meta={"items": ["1.03", "9.01"]})
    label = classify_event(doc)
    assert label.primary == EventClass.CREDIT_EVENT and label.subtype == "BANKRUPTCY"
    assert label.confidence >= 0.9


def test_keyword_rules(make_doc):
    cases = {
        "Depositors rush out as a bank run hits the lender": (EventClass.CREDIT_EVENT, "BANK_RUN"),
        "New tariffs spark fears of a trade war": (EventClass.GEOPOLITICAL, "TRADE_WAR"),
        "Company beats estimates and raises guidance": (EventClass.EARNINGS_GUIDANCE, None),
        "Tesla recalls 200,000 vehicles": (EventClass.PRODUCT_STRATEGY, "RECALL"),
        "Missile strikes escalate the conflict overnight": (EventClass.GEOPOLITICAL, "ARMED_CONFLICT"),
    }
    for title, (cls, subtype) in cases.items():
        label = classify_event(make_doc(title=title))
        assert label.primary == cls, title
        if subtype:
            assert label.subtype == subtype, title


def test_trade_war_is_not_armed_conflict(make_doc):
    assert classify_event(make_doc(title="Trade war escalates")).subtype == "TRADE_WAR"


def test_unclassified_text_is_other(make_doc):
    label = classify_event(make_doc(title="Weather is pleasant this weekend"))
    assert label.primary == EventClass.OTHER and label.confidence <= 0.3


def test_credibility_and_authority():
    assert credibility("www.reuters.com", SourceType.NEWS) == 0.95
    assert credibility("unknown-blog.example", SourceType.NEWS) == 0.6
    assert credibility("reuters.com", SourceType.SOCIAL) == 0.3  # a social post never borrows a wire's reputation
    assert is_authoritative("sec.gov", SourceType.FILING)
    assert not is_authoritative("bsky.app", SourceType.SOCIAL)


def test_impact_is_monotone_and_bounded():
    weak, _ = heuristic_impact(EventClass.CREDIT_EVENT, -0.1, 0.6, 80, 100)
    strong, _ = heuristic_impact(EventClass.CREDIT_EVENT, -0.9, 0.6, 80, 100)
    social, _ = heuristic_impact(EventClass.CREDIT_EVENT, -0.9, 0.3, 80, 100)
    wire, _ = heuristic_impact(EventClass.CREDIT_EVENT, -0.9, 0.95, 80, 100)
    assert weak < strong and social < wire
    top, _ = heuristic_impact(EventClass.CREDIT_EVENT, -1.0, 1.0, 100, 100)
    low, _ = heuristic_impact(EventClass.OTHER, 0.0, 0.3, 0, 0)
    assert top == 10 and low == 1
