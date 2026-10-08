from datetime import UTC, datetime, timedelta

import pytest

from seismo.engine import Engine
from seismo.ingest.replay import load_pack
from seismo.nlp.sentiment import LexiconBackend
from seismo.scenarios import build_gate, load_library
from seismo.schemas import EventClass, SourceType
from seismo.signals.aggregator import EntityAggregator
from seismo.signals.gate import GateConsumer
from tests.conftest import ROOT

PACKS = ROOT / "data" / "replay"
T0 = datetime(2026, 9, 15, 14, 0, tzinfo=UTC)


def _run(pack, universe, settings):
    engine = Engine(universe, LexiconBackend(), settings)
    gate = GateConsumer(build_gate(settings))
    events = []
    for doc in load_pack(PACKS / f"{pack}.jsonl"):
        result = engine.process_full(doc)
        for sig in result.doc_signals:
            gate.on_signal(sig)
        for sig in result.event_signals:
            events.append(sig)
            gate.on_signal(sig)
    return engine, gate.log, events


def _last_event(engine, doc, entity):
    return next(s for s in engine.process_full(doc).event_signals if s.entity.id == entity)


def test_sister_outlets_count_as_one_publisher(engine, make_doc):
    a = make_doc(title="Boeing faces new FAA probe over 737 MAX", publisher="street-journal.example", published_at=T0)
    b = make_doc(title="FAA opens fresh Boeing 737 MAX investigation", publisher="street-journal-markets.example",
                 published_at=T0 + timedelta(minutes=5))
    engine.process_full(a)
    sig = _last_event(engine, b, "BA")
    assert sig.corroboration.independent_publishers == 1


def test_syndicated_copy_inherits_the_original_voice(engine, make_doc):
    title = "Boeing faces new FAA probe over 737 MAX door plug"
    engine.process_full(make_doc(title=title, publisher="wire-one.example", published_at=T0))
    copy = make_doc(title=title, publisher="fin-daily.example", published_at=T0 + timedelta(minutes=2))
    result = engine.process_full(copy)
    assert "syndicated_copy" in result.doc_signals[0].flags
    assert result.event_signals[0].corroboration.independent_publishers == 1


def test_independent_publishers_accumulate(engine, make_doc):
    for i, pub in enumerate(["wire-one.example", "biz-times.example", "markets-tv.example"]):
        sig = _last_event(engine, make_doc(title=f"Boeing faces FAA probe, report {i} says new details emerged",
                                           publisher=pub, published_at=T0 + timedelta(minutes=10 * i)), "BA")
    assert sig.corroboration.independent_publishers == 3
    assert sig.corroboration.publishers_60m == 3


def test_coordinated_reposts_are_flagged(engine, make_doc):
    for i in range(6):
        sig = _last_event(engine, make_doc(body="Harbor National Bank just halted withdrawals!! get your money out now",
                                           source_type=SourceType.SOCIAL, publisher="bsky.app",
                                           published_at=T0 + timedelta(seconds=30 * i)), "HNB")
        assert sig is not None
    assert sig.corroboration.coordinated
    assert "coordinated_posts" in sig.flags


def test_lookalike_domain_is_flagged_and_not_counted(engine, make_doc):
    doc = make_doc(title="BREAKING: Harbor National Bank halts customer withdrawals",
                   publisher="wire-one-alerts.example", published_at=T0)
    result = engine.process_full(doc)
    assert "lookalike_source" in result.doc_signals[0].flags
    assert result.event_signals[0].corroboration.independent_publishers == 0
    assert not result.event_signals[0].corroboration.authoritative


def test_red_team_naive_fires_seismo_holds_then_retracts(universe, settings):
    _, log, _ = _run("red_team", universe, settings)
    hnb = [d for d in log if d.entity_id == "HNB"]
    assert any(d.naive_decision == "TRIGGER" for d in hnb), "the naive pipeline should fire on the fake"
    assert not any(d.decision == "TRIGGER" for d in hnb), "Seismo must never auto-trigger on the fake"
    assert any(d.decision == "REVIEW" and d.impact_score >= 8 for d in hnb)
    assert hnb[-1].decision == "RETRACT"
    review = next(d for d in hnb if d.decision == "REVIEW")
    corroboration = next(c for c in review.checks if c.name == "corroboration")
    assert not corroboration.passed


def test_svb_triggers_before_the_bank_is_closed(universe, settings):
    _, log, _ = _run("svb_2023", universe, settings)
    triggers = [d for d in log if d.decision == "TRIGGER" and d.entity_id == "SIVB"]
    assert triggers and triggers[0].scenario == "svb_2023"
    closure = datetime(2023, 3, 10, 16, 15, tzinfo=UTC)
    assert triggers[0].as_of < closure


def test_tariff_shock_maps_to_the_trade_war_analog(universe, settings):
    _, log, _ = _run("tariff_2025", universe, settings)
    assert any(d.decision == "TRIGGER" and d.scenario == "tariff_2025" for d in log)


@pytest.mark.parametrize("pack", ["quiet_day"])
def test_quiet_day_never_triggers(pack, universe, settings):
    _, log, _ = _run(pack, universe, settings)
    assert not any(d.decision == "TRIGGER" for d in log)
    assert any(d.naive_decision == "TRIGGER" for d in log), "the naive comparator over-fires on calm days"


def test_retraction_unwinds_the_entity_index(universe, settings):
    _, _, events = _run("red_team", universe, settings)
    agg = EntityAggregator()
    before = None
    for sig in events:
        out = agg.update(sig)
        if sig.status == "active" and sig.entity.id == "HNB":
            before = out.sentiment_score
    after = out.sentiment_score
    assert before is not None and before < -0.2
    assert after == 0.0


def test_scenario_library_loads(settings):
    lib = load_library(str(settings.path("scenarios.path")))
    assert lib.multiplier(9) == 1.0 and lib.z(10) == -2.33
    assert lib.scenario_map().lookup(EventClass.CREDIT_EVENT, "BANK_RUN") == "svb_2023"


def test_adani_pack_triggers_before_the_nse_open_and_the_denial_contests(universe, settings):
    _, log, events = _run("adani_2023", universe, settings)
    triggers = [d for d in log if d.decision == "TRIGGER" and d.entity_id == "ADANIENT.NS"]
    assert triggers and triggers[0].as_of < datetime(2023, 1, 25, 3, 45, tzinfo=UTC)
    assert triggers[0].scenario == "adani_2023"
    assert triggers[0].event.subtype == "FRAUD_ALLEGATION"
    contested = [s for s in events if "contested" in s.flags]
    assert contested and all(s.status == "active" for s in contested)
    assert not any(d.decision == "RETRACT" for d in log)


def test_a_denial_retracts_an_unconfirmed_rumour_but_only_contests_a_confirmed_story(engine, make_doc):
    rumour = make_doc(title="Harbor National Bank halts customer withdrawals, source says",
                      publisher="fin-daily.example", published_at=T0)
    engine.process_full(rumour)
    deny = make_doc(title="Harbor National Bank denies halting withdrawals; reports are false",
                    publisher="wire-one.example", published_at=T0 + timedelta(minutes=20))
    sig = _last_event(engine, deny, "HNB")
    assert sig.status == "retracted"

    t1 = T0 + timedelta(days=3)
    reports = [("fin-daily.example", "Short seller accuses Adani Group of accounting fraud"),
               ("biz-times.example", "Adani Group shares under pressure after a report alleging stock manipulation"),
               ("market-daily.example", "Hindenburg report flags offshore shell companies linked to the Adani Group")]
    for i, (pub, title) in enumerate(reports):
        engine.process_full(make_doc(title=title, publisher=pub, published_at=t1 + timedelta(minutes=5 * i)))
    deny = make_doc(title="Adani Group calls the short seller's fraud report baseless",
                    publisher="wire-two.example", published_at=t1 + timedelta(hours=2))
    sig = _last_event(engine, deny, "ADANIENT.NS")
    assert sig.status == "active" and "contested" in sig.flags
