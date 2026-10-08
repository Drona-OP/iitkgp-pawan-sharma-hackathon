import math

import pytest

from seismo.ingest.replay import load_pack
from seismo.module_a.consumer import ModuleAConsumer
from seismo.module_a.index import Holding, TiltConfig, capped_cap_weights, target_weights, zscores
from seismo.module_b import credit
from seismo.module_b.blotter import aggregate, generate
from seismo.module_b.book import build_book
from seismo.module_b.factors import compute_shocks, load_market, rate_at, write_shocks
from seismo.module_b.memo import write_memo
from seismo.module_b.pricing import bond_risk, bs_greeks, swap_dv01
from seismo.module_b.stress import RBI_FLOOR, StressEngine, StressRequest
from seismo.scenarios import load_library
from tests.conftest import ROOT
from tests.synth_market import write_synthetic_market

# ---------------------------------------------------------------- finance maths


def test_basel_correlation_bounds():
    assert math.isclose(credit.basel_rho(0.0001), 0.24, abs_tol=0.01)
    assert math.isclose(credit.basel_rho(0.5), 0.12, abs_tol=0.001)


def test_vasicek_is_monotone_in_the_systematic_factor():
    pd = credit.PD_1Y["BBB"]
    assert credit.vasicek_pd(pd, -2.0) > credit.vasicek_pd(pd, -1.0) > credit.vasicek_pd(pd, 0.0)
    assert credit.vasicek_pd(1.0, -2.0) == 1.0


def test_staging_rules():
    assert credit.stage("A", "A") == 1
    assert credit.stage("A", "BBB") == 2          # three notches
    assert credit.stage("BBB", "BB-") == 2        # BB- or worse
    assert credit.stage("BBB", "D") == 3


def test_ecl_by_stage():
    assert credit.ecl(1, 0.01, 0.4, 100.0, 5) == pytest.approx(0.4)
    assert credit.ecl(2, 0.01, 0.4, 100.0, 5) > credit.ecl(1, 0.01, 0.4, 100.0, 5)
    assert credit.ecl(3, 0.01, 0.4, 100.0, 5) == pytest.approx(40.0)


def test_risk_weights():
    assert credit.risk_weight("AA") == 0.20 and credit.risk_weight("A-") == 0.50
    assert credit.risk_weight("BBB") == 0.75 and credit.risk_weight("BB-") == 1.0
    assert credit.risk_weight("B") == 1.5 and credit.risk_weight(None) == 1.0
    assert credit.risk_weight("AA+", sovereign=True) == 0.0


def test_par_bond_prices_at_100_and_duration_is_sane():
    price, mod, conv = bond_risk(0.05, 10, 0.05)
    assert price == pytest.approx(100, abs=1e-6)
    assert 7.0 < mod < 8.0 and conv > 0


def test_swap_dv01_sign():
    assert swap_dv01(1e8, 5, 0.04, receive_fixed=True) < 0 < swap_dv01(1e8, 5, 0.04, receive_fixed=False)


def test_put_call_delta_parity():
    c = bs_greeks(100, 100, 1.0, 0.2, 0.0, call=True)
    p = bs_greeks(100, 100, 1.0, 0.2, 0.0, call=False)
    assert c["delta"] - p["delta"] == pytest.approx(1.0)
    assert c["gamma"] == pytest.approx(p["gamma"])


# ---------------------------------------------------------------- Module B


@pytest.fixture(scope="module")
def stress_engine(tmp_path_factory, settings):
    market = tmp_path_factory.mktemp("market")
    write_synthetic_market(market)
    lib = load_library(str(settings.path("scenarios.path")))
    write_shocks(compute_shocks(load_market(market), lib.scenarios), market / "analog_shocks.csv")
    return StressEngine(lib, market / "analog_shocks.csv", build_book(ROOT))


def test_blotter_is_deterministic_and_aggregates():
    _, a = generate(ROOT)
    _, b = generate(ROOT)
    assert [t.trade_id for t in a] == [t.trade_id for t in b]
    positions = aggregate(a)
    assert len(a) > len(positions) > 150


def test_analog_shocks_come_from_the_data(stress_engine):
    svb = stress_engine.shocks["svb_2023"]
    assert svb["EQ:BANKS"] < svb["EQ:MKT"] < 0      # regional banks fell more than the market
    assert svb["IR:2Y"] < 0                         # front-end rally
    assert rate_at(svb, 2.0) == pytest.approx(svb["IR:2Y"])


def test_svb_stress_defaults_the_epicenter_and_hits_capital(stress_engine):
    r = stress_engine.run(StressRequest("svb_2023", 10, "SIVB", "CREDIT_EVENT", "BANK_RUN"))
    assert "SIVB" in r.defaulted
    assert r.ecl_after > r.ecl_before
    assert r.cet1_ratio_after < r.cet1_ratio_before
    assert any(f["to"] == "Stage 3" for f in r.stage_flows)
    assert r.top_contributors[0].loss > 0
    memo = write_memo(r)
    assert "CET1" in memo and "SIVB" in memo


def test_more_severity_means_less_capital(stress_engine):
    req = StressRequest("lehman_2008", 9, None, "CREDIT_EVENT", "BANKRUPTCY")
    ratios = [stress_engine.ratio(req, m) for m in (0.0, 0.5, 1.0, 2.0)]
    assert ratios == sorted(ratios, reverse=True)


def test_reverse_stress_finds_the_floor(stress_engine):
    req = StressRequest("lehman_2008", 9, None, "CREDIT_EVENT", "BANKRUPTCY")
    m = stress_engine.reverse(req, RBI_FLOOR)
    assert m is not None
    assert stress_engine.ratio(req, m) <= RBI_FLOOR + 1e-3
    assert stress_engine.ratio(req, max(0.0, m - 0.05)) > RBI_FLOOR - 1e-3


# ---------------------------------------------------------------- Module A


def _holdings():
    sectors = ["IT", "IT", "Fin", "Fin", "Energy"]
    return {f"T{i}": Holding(f"T{i}", sectors[i], 0.2, 0.2 + 0.05 * i) for i in range(5)}


def test_capped_cap_weights():
    w = capped_cap_weights({"A": 90, "B": 5, "C": 5}, cap=0.5)
    assert max(w.values()) <= 0.5 + 1e-9 and sum(w.values()) == pytest.approx(1.0)


def test_shrinkage_means_no_news_no_tilt():
    z = zscores({"A": -0.6, "B": 0.0, "C": 0.0}, {"A": 1}, k=2)
    assert z["B"] == 0.0 and z["C"] == 0.0 and z["A"] < 0


def test_negative_sentiment_cuts_weight_within_caps():
    h = _holdings()
    st = target_weights(h, {"T0": -0.8}, {"T0": 3}, set(), TiltConfig(name_cap=0.3))
    assert st.weights["T0"] < 0.2
    assert sum(st.weights.values()) == pytest.approx(1.0)
    assert all(abs(st.weights[t] - h[t].bench) <= 0.05 + 1e-6 for t in h)


def test_circuit_breaker_forces_max_underweight():
    h = _holdings()
    st = target_weights(h, {}, {}, {"T2"}, TiltConfig(name_cap=0.3))
    assert st.weights["T2"] == pytest.approx(0.15)


def test_deepseek_replay_cuts_nvidia_before_monday_open(universe, settings):
    from seismo.engine import Engine
    from seismo.nlp.sentiment import LexiconBackend

    engine = Engine(universe, LexiconBackend(), settings)
    module = ModuleAConsumer.from_settings(settings, store=None)
    for doc in load_pack(ROOT / "data" / "replay" / "deepseek_2025.jsonl"):
        result = engine.process_full(doc)
        for sig in result.all:
            module.on_signal(sig)
    monday_open = "2025-01-27T14:30"
    before_open = [s for s in module.snapshots if s["as_of"] < monday_open]
    assert before_open[-1]["weights"]["NVDA"] < before_open[-1]["bench"]["NVDA"]
    assert module.naive_turnover > module.turnover_total, "the naive tilt churns more"


def test_india_analog_is_local_and_spreads_through_the_group(stress_engine):
    analog = stress_engine.analog("adani_2023")
    assert "EQ:MKT" not in analog and "CR:HY" not in analog          # US noise dropped
    assert analog["EQN:ADANIENT.NS"] < analog["EQ:IN"] <= 0           # the name fell far more than the market
    r = stress_engine.run(StressRequest("adani_2023", 10, "ADANIENT.NS", "MANAGEMENT_GOVERNANCE", "FRAUD_ALLEGATION"))
    assert "ADANIPORTS.NS" in r.downgraded                            # group contagion
    assert "CAT" not in r.downgraded and "LT.NS" not in r.downgraded  # not sector contagion
    assert not r.defaulted
    assert r.cet1_ratio_after < r.cet1_ratio_before


def test_sector_contagion_stays_in_the_country(stress_engine):
    r = stress_engine.run(StressRequest("svb_2023", 10, "SIVB", "CREDIT_EVENT", "BANK_RUN"))
    assert "FRC" in r.downgraded
    assert not any(k.endswith(".NS") for k in r.downgraded)


def test_india_caps_use_shares_times_the_close_before_the_replay(tmp_path):
    import pandas as pd

    from seismo.module_a import market as mk

    pd.DataFrame({"date": ["2023-01-23", "2023-01-24", "2023-01-25"], "ADANIENT.NS": [3400.0, 3442.0, 3389.0]}).to_csv(
        tmp_path / "prices_daily.csv", index=False)
    (tmp_path / "shares_in.csv").write_text("ticker,shares\nADANIENT.NS,1000\n", encoding="utf-8")
    mk._prices.cache_clear()
    caps = mk.caps_from_shares(tmp_path, ["ADANIENT.NS"], pd.Timestamp("2023-01-25T03:00", tz="UTC"))
    assert caps == {"ADANIENT.NS": 3442.0 * 1000}
    mk._prices.cache_clear()


def test_breaker_freezes_the_rest_of_the_group_at_benchmark():
    holdings = {"A": Holding("A", "Industrials", 0.10, 0.3, "G"), "B": Holding("B", "Industrials", 0.05, 0.3, "G"),
                **{f"X{i}": Holding(f"X{i}", f"S{i}", 0.85 / 8, 0.2) for i in range(8)}}
    vals = {"B": 0.8}  # good news on the sister company would normally lift it
    state = target_weights(holdings, vals, {"B": 3}, {"A"}, TiltConfig())
    assert state.weights["A"] == pytest.approx(0.05)
    assert state.weights["B"] <= 0.05 + 1e-9
    assert sum(state.weights.values()) == pytest.approx(1.0)
