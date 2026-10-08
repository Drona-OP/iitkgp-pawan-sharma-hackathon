from seismo.nlp.credibility import credibility, is_authoritative
from seismo.nlp.events import classify_event
from seismo.nlp.impact import ImpactFeatures, PriorImpact, heuristic_impact
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
    def imp(**kw):
        base = dict(event=EventClass.CREDIT_EVENT, sentiment=-0.9, publishers=1, social_authors=0,
                    authoritative=False, novelty=100, relevance=80)
        base.update(kw)
        return PriorImpact().score(ImpactFeatures(**base))[0]

    assert imp(sentiment=-0.1) < imp(sentiment=-0.9)
    assert imp(publishers=1) <= imp(publishers=4) <= imp(publishers=12)
    assert imp(social_authors=0) <= imp(social_authors=40)
    assert imp(event=EventClass.OTHER) < imp(event=EventClass.CREDIT_EVENT)
    assert imp(sentiment=-1.0, publishers=10, social_authors=40, authoritative=True, relevance=100) == 10
    assert imp(event=EventClass.OTHER, sentiment=0.0, publishers=0, novelty=0, relevance=0) == 1


def test_credibility_is_not_part_of_impact():
    """Impact is "how big if true"; the gate decides whether it is true."""
    a, _ = heuristic_impact(EventClass.CREDIT_EVENT, -0.9, 0.3, 80, 100)
    b, _ = heuristic_impact(EventClass.CREDIT_EVENT, -0.9, 0.95, 80, 100)
    assert a == b

def test_target_masked_model_scores_each_company_separately(tmp_path):
    import json
    import re

    from seismo.eval.sentiment_train import _fit, to_spec
    from seismo.nlp.target_model import TargetModelBackend, mask

    names = ["Infosys", "Wipro", "SpiceJet", "IndiGo", "Maruti", "Cipla"]
    up, down = ["gains", "jumps", "rallies", "surges"], ["falls", "slides", "slumps", "drops"]
    rows = []
    for i in range(240):
        a, b = names[i % 6], names[(i + 1 + i // 6) % 6]
        if a == b:
            continue
        title = f"{a} {up[i % 4]} while {b} {down[(i // 4) % 4]}"
        sa = [(0, len(a))]
        sb = [(m.start(), m.end()) for m in re.finditer(b, title)]
        rows.append({"hid": i, "title": title, "target": sa, "others": sb, "label": "positive"})
        rows.append({"hid": i, "title": title, "target": sb, "others": sa, "label": "negative"})
    vec, clf = _fit(rows, True, 1.0)
    path = tmp_path / "m.json"
    path.write_text(json.dumps(to_spec(vec, clf, "unit test")), encoding="utf-8")
    model = TargetModelBackend(path)
    title = "Cipla gains while Maruti slides"
    p_cipla, p_maruti = model.predict_targeted([(title, [(0, 5)], [(18, 24)]), (title, [(18, 24)], [(0, 5)])])
    assert p_cipla[2] > p_cipla[0] and p_maruti[0] > p_maruti[2]
    assert mask(title, [(0, 5)], [(18, 24)]) == "TGT gains while OTH slides"
