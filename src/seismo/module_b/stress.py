"""Module B: a corroborated high-impact event becomes a bank-grade stress result in seconds.

Pipeline for one run:
  1. Shock vector = the matching historical analog x severity(impact) x m (m = 1 normally; the
     reverse stress test and the dashboard slider move it).
  2. Idiosyncratic overlay: the obligor at the centre is downgraded (and defaults for bank runs
     and bankruptcies at high impact); same-sector peers take contagion notches, or, for a
     governance shock, the other companies of the same business group do (group contagion, as in
     rating agencies' group notching).
  3. Revaluation: bonds by duration-convexity on rate + spread moves, swaps by DV01, CDS by CS01
     (with jump-to-default), FX forwards by notional x FX return, options by delta-gamma-vega,
     equities by beta x sector return plus the idiosyncratic shock; an NSE stake by its own
     observed move in the analog window when the data has it; INR assets also carry USD/INR.
  4. Credit: Vasicek stressed PDs at systematic factor Z = Z(impact) x s x m, where the systemic
     intensity s = clip(max(|S&P 500 move| / 20%, |HY spread move| / 400bp), 0.1, 1) is read
     off the analog itself (a sector shock like DeepSeek barely moves the credit cycle; Lehman
     and Covid are fully systemic), then damped rating migration,
     IFRS 9 / RBI-style staging and ECL; also the probability-weighted (base/adverse/severe) ECL.
  5. Capital: CET1 after tax-effected losses; RWA re-computed on migrated ratings; Basel 7.0%
     and RBI 8.0% floors; reverse stress by bisection on m.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from seismo.module_b.book import (
    Book,
    Position,
    _value,
    base_rate,
    baseline_ecl,
    load_book,
    rwa_position,
)
from seismo.module_b.credit import PD_1Y, ecl, notch_down, stage, systematic_notches, vasicek_pd
from seismo.module_b.factors import load_shocks, rate_at, spread_shock
from seismo.scenarios import ScenarioLibrary

TAX_RATE = 0.25
CET1_START_RATIO = 0.13
BASEL_FLOOR = 0.07
RBI_FLOOR = 0.08
ASSET_CLASSES = ["Loans", "Bonds", "Derivatives", "Equities"]
INDIA_FACTORS = {"EQ:IN", "EQ:IN_BANKS", "FX:USDINR", "IR:IN10Y"}


@dataclass(frozen=True)
class StressRequest:
    scenario: str
    impact: int = 9
    epicenter: str | None = None          # obligor id at the centre (e.g. "SIVB")
    event_class: str | None = None        # e.g. "CREDIT_EVENT"
    subtype: str | None = None            # e.g. "BANK_RUN"
    m: float = 1.0                        # extra severity multiplier (reverse stress, slider)


class ContributorRow(BaseModel):
    pos_id: str
    name: str
    asset_class: str
    sector: str
    loss: float
    detail: str


class StressResult(BaseModel):
    run_id: str
    as_of: datetime | None = None
    scenario: str
    scenario_label: str
    impact: int
    multiplier: float
    m: float
    epicenter: str | None
    epicenter_name: str | None = None
    event: str | None = None
    defaulted: list[str] = Field(default_factory=list)
    downgraded: dict[str, str] = Field(default_factory=dict)
    shocks: dict[str, float]
    value_before: dict[str, float]
    value_after: dict[str, float]
    pnl_by_class: dict[str, float]
    ecl_before: float
    ecl_after: float
    ecl_weighted: float
    ecl_by_stage_before: dict[str, float]
    ecl_by_stage_after: dict[str, float]
    stage_flows: list[dict] = Field(default_factory=list)   # {from, to, count, ead}
    heatmap: list[dict] = Field(default_factory=list)       # {sector, asset_class, loss}
    top_contributors: list[ContributorRow] = Field(default_factory=list)
    rwa_before: float
    rwa_after: float
    cet1_before: float
    cet1_after: float
    cet1_ratio_before: float
    cet1_ratio_after: float
    pre_tax_loss: float
    reverse_m_rbi: float | None = None
    reverse_m_basel: float | None = None
    curve: list[tuple[float, float]] = Field(default_factory=list)
    trigger: dict = Field(default_factory=dict)
    memo: str = ""
    synthetic: bool = True


class StressEngine:
    def __init__(self, library: ScenarioLibrary, shocks_path: str | Path, book: Book) -> None:
        self.lib = library
        self.shocks = load_shocks(shocks_path)
        self.book = book
        self._rwa0 = sum(rwa_position(p) for p in book.positions)
        self.cet1_start = CET1_START_RATIO * self._rwa0

    @classmethod
    def from_settings(cls, settings) -> StressEngine:
        from seismo.scenarios import load_library

        root = settings.root
        shocks = Path(settings.get("module_b.shocks_path", "data/market/analog_shocks.csv"))
        shocks = shocks if shocks.is_absolute() else root / shocks
        return cls(load_library(str(settings.path("scenarios.path"))), shocks, load_book(str(root)))

    # ------------------------------------------------------------------ overlays
    def _idio(self, req: StressRequest, m: float) -> tuple[dict[str, int], set[str], float]:
        """Notches per obligor, defaulted obligors, and the epicenter equity shock."""
        if not req.epicenter or req.epicenter not in self.book.obligors or m <= 0:
            return {}, set(), 0.0
        cfg = self.lib.idio(req.event_class or "default")
        notches = int(round(cfg.get("notches", 1) * m))
        epi = self.book.obligors[req.epicenter]
        out = {epi.obligor_id: notches}
        if cfg.get("contagion", "sector") == "group":
            group_notches = int(round(notches * self.lib.group_share))
            if epi.group and group_notches:
                for o in self.book.obligors.values():
                    if o.obligor_id != epi.obligor_id and o.group == epi.group:
                        out[o.obligor_id] = group_notches
        else:
            contagion = int(round(notches * self.lib.contagion_share))
            if contagion:
                for o in self.book.obligors.values():
                    if (o.obligor_id != epi.obligor_id and o.sector == epi.sector and o.bank == epi.bank
                            and o.country == epi.country):
                        out[o.obligor_id] = contagion
        defaulted = set()
        if (
            req.subtype in self.lib.default_subtypes
            and req.impact >= int(cfg.get("default_min_impact", 11))
            and m >= 0.5
        ):
            defaulted.add(epi.obligor_id)
        observed = self.lib.scenarios.get(req.scenario)
        if observed is not None and epi.obligor_id in observed.epicenters:
            return out, defaulted, 0.0   # the analog's own move of this name already holds the shock
        return out, defaulted, float(cfg.get("equity_shock", 0.0)) * m

    # ------------------------------------------------------------------ market revaluation
    def _mtm(self, p: Position, shock: dict[str, float], notches: dict[str, int], defaulted: set[str],
             equity_idio: float, epicenter: str | None, spread_per_notch: float) -> tuple[float, str]:
        fx_inr = shock.get("FX:USDINR", 0.0)
        if p.asset_class == "Bonds":
            if p.obligor_id in defaulted:
                loss = _value(p) - p.d["face"] * (1 - p.d["lgd"])
                return -loss, "default: marked to recovery"
            dy_bp = shock.get("IR:IN10Y", 0.0) if p.currency == "INR" else rate_at(shock, p.years)
            if not p.d["sovereign"]:
                dy_bp += spread_shock(shock, p.rating) + spread_per_notch * notches.get(p.obligor_id, 0)
            dy = dy_bp / 10_000
            dp = -p.d["mod_dur"] * dy + 0.5 * p.d["convexity"] * dy * dy
            v = _value(p)
            pnl = v * dp
            if p.currency == "INR":
                pnl += (v + pnl) * (1 / (1 + fx_inr) - 1)
            return pnl, f"dy {dy_bp:+.0f}bp, D {p.d['mod_dur']:.1f}"
        if p.asset_class == "Derivatives":
            if p.instrument == "IRS":
                r = rate_at(shock, p.years)
                return p.d["dv01"] * r, f"{r:+.0f}bp x DV01"
            if p.instrument == "FX_FWD":
                ret = shock.get("FX:USDINR" if p.d["pair"] == "USDINR" else "FX:EURUSD", 0.0)
                return p.d["notional"] * ret * p.d["sign"], f"{p.d['pair']} {ret:+.1%}"
            if p.instrument == "CDS":
                if p.obligor_id in defaulted:
                    payout = p.d["notional"] * p.d["lgd"]
                    return (payout if p.d["bought"] else -payout), "credit event: protection pays out"
                ds = spread_shock(shock, p.rating) + spread_per_notch * notches.get(p.obligor_id, 0)
                return p.d["cs01"] * ds, f"spread {ds:+.0f}bp"
            if p.instrument == "EQ_OPTION":
                ret = shock.get(p.d["underlying"], shock.get("EQ:MKT", 0.0))
                ds = 100.0 * ret
                dvol = shock.get("VOL:VIX", 0.0) / 100.0
                u = p.d["units"]
                pnl = u * (p.d["delta"] * ds + 0.5 * p.d["gamma"] * ds * ds + p.d["vega"] * dvol)
                return pnl, f"underlying {ret:+.1%}, vol {dvol * 100:+.0f}pt"
            return 0.0, ""
        if p.asset_class == "Equities":
            own = shock.get(f"EQN:{p.obligor_id}")
            ret = own if own is not None else shock.get(p.factor, shock.get("EQ:MKT", 0.0)) * p.d["beta"]
            if p.obligor_id in defaulted:
                ret = -0.95
            elif p.obligor_id == epicenter:
                ret = (1 + ret) * (1 + equity_idio) - 1
            ret = max(-1.0, ret)
            pnl = p.d["value"] * ret
            if p.currency == "INR":
                pnl = p.d["value"] * ((1 + ret) / (1 + fx_inr) - 1)
            return pnl, f"{ret:+.1%}" + (" (own move)" if own is not None else "")
        return 0.0, ""

    # ------------------------------------------------------------------ credit
    def _credit(self, p: Position, z: float, notches: dict[str, int], defaulted: set[str]) -> tuple[str, int, float, float]:
        """Stressed rating, stage, stressed one-year PD and ECL for a loan."""
        orig = p.orig_rating or p.rating
        if p.obligor_id in defaulted:
            return "D", 3, 1.0, ecl(3, 1.0, p.d["lgd"], p.d["ead"], p.years)
        rating = notch_down(p.rating, notches.get(p.obligor_id, 0))
        rating = notch_down(rating, systematic_notches(PD_1Y[rating], z))
        # PD(Z) is a conditional (median-state) PD, below the unconditional PD near Z = 0;
        # a stress test never improves a PD, so take the worse of the two.
        pd = max(PD_1Y[rating], vasicek_pd(PD_1Y[rating], z)) if z < 0 else PD_1Y[rating]
        st = stage(orig, rating)
        return rating, st, pd, ecl(st, pd, p.d["lgd"], p.d["ead"], p.years, PD_1Y[rating])

    # ------------------------------------------------------------------ one run
    def systemic(self, scenario: str) -> float:
        base = self.analog(scenario)
        s = max(abs(base.get("EQ:MKT", 0.0)) / 0.20, abs(base.get("CR:HY", 0.0)) / 400.0,
                abs(base.get("EQ:IN", 0.0)) / 0.20)
        return round(min(1.0, max(0.1, s)), 3)

    @staticmethod
    def _cap(factor: str, value: float) -> float:
        """Keep scaled shocks physical: prices cannot fall below -95%, yields not below zero."""
        if factor.startswith(("EQ:", "EQN:", "FX:", "CM:")):
            return max(-0.95, value)
        if factor.startswith("IR:"):
            tenor = {"IR:3M": 0.25, "IR:2Y": 2, "IR:5Y": 5, "IR:10Y": 10, "IR:30Y": 30}.get(factor, 10)
            return max(-base_rate(tenor) * 10_000, value)
        if factor.startswith("CR:"):
            return max(-50.0, value)
        if factor == "VOL:VIX":
            return max(-10.0, value)
        return value

    def analog(self, scenario: str) -> dict[str, float]:
        """The analog's shock vector; a local (India-scope) analog keeps only Indian factors."""
        base = self.shocks.get(scenario, {})
        sc = self.lib.scenarios.get(scenario)
        if sc is not None and sc.scope == "india":
            base = {k: v for k, v in base.items() if k in INDIA_FACTORS or k.startswith("EQN:")}
        return base

    def _core(self, req: StressRequest, m: float) -> dict:
        base = self.analog(req.scenario)
        mult = self.lib.multiplier(req.impact) * m
        shock = {k: self._cap(k, v * mult) for k, v in base.items()}
        z = self.lib.z(req.impact) * self.systemic(req.scenario) * m
        notches, defaulted, eq_idio = self._idio(req, m)
        spread_per_notch = float(self.lib.idio(req.event_class or "default").get("spread_bp_per_notch", 50))

        before = defaultdict(float)
        after = defaultdict(float)
        pnl = defaultdict(float)
        contributors = []
        heat = defaultdict(float)
        ecl0 = ecl1 = 0.0
        ecl_stage0 = defaultdict(float)
        ecl_stage1 = defaultdict(float)
        flows = defaultdict(lambda: [0, 0.0])
        rwa1 = 0.0
        downgraded: dict[str, str] = {}
        for p in self.book.positions:
            v0 = _value(p)
            before[p.asset_class] += v0
            if p.asset_class == "Loans":
                st0, e0 = baseline_ecl(p)
                rating1, st1, _, e1 = self._credit(p, z, notches, defaulted)
                ecl0 += e0
                ecl1 += e1
                ecl_stage0[f"Stage {st0}"] += e0
                ecl_stage1[f"Stage {st1}"] += e1
                flows[(st0, st1)][0] += 1
                flows[(st0, st1)][1] += p.d["ead"]
                loss = e1 - e0
                after["Loans"] += v0 - e1
                pnl["Loans"] -= loss
                heat[(p.sector, "Loans")] += loss
                rwa1 += rwa_position(p, rating1, defaulted=st1 == 3)
                if rating1 != p.rating:
                    downgraded[p.obligor_id] = rating1
                contributors.append((loss, p, f"{p.rating}->{rating1}, stage {st0}->{st1}"))
            else:
                d, detail = self._mtm(p, shock, notches, defaulted, eq_idio, req.epicenter, spread_per_notch)
                after[p.asset_class] += v0 + d
                pnl[p.asset_class] += d
                heat[(p.sector, p.asset_class)] += -d
                rating1 = p.rating
                if p.asset_class == "Bonds" and p.rating and not p.d["sovereign"]:
                    rating1 = "D" if p.obligor_id in defaulted else notch_down(p.rating, notches.get(p.obligor_id, 0))
                rwa1 += rwa_position(p, rating1, defaulted=p.obligor_id in defaulted, value=v0 + d)
                contributors.append((-d, p, detail))
        # Loans are reported net of the ECL allowance on both sides.
        before["Loans"] -= ecl0
        mtm = sum(v for k, v in pnl.items() if k != "Loans")
        delta_ecl = ecl1 - ecl0
        pre_tax_loss = delta_ecl - mtm
        cet1_after = self.cet1_start - (1 - TAX_RATE) * pre_tax_loss
        return {
            "shock": shock, "mult": mult, "z": z, "notches": notches, "defaulted": defaulted,
            "before": dict(before), "after": dict(after), "pnl": dict(pnl), "contributors": contributors,
            "heat": heat, "ecl0": ecl0, "ecl1": ecl1, "ecl_stage0": dict(ecl_stage0),
            "ecl_stage1": dict(ecl_stage1), "flows": flows, "rwa1": rwa1, "cet1_after": cet1_after,
            "pre_tax_loss": pre_tax_loss, "downgraded": downgraded,
        }

    def ratio(self, req: StressRequest, m: float) -> float:
        c = self._core(req, m)
        return c["cet1_after"] / c["rwa1"]

    def reverse(self, req: StressRequest, floor: float, hi: float = 10.0) -> float | None:
        """Smallest m at which CET1 falls to the floor: grid scan, then bisection inside the bracket."""
        grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, hi]
        lo = None
        for a, b in zip(grid, grid[1:], strict=False):
            if self.ratio(req, b) <= floor:
                lo, hi = a, b
                break
        if lo is None:
            return None
        for _ in range(30):
            mid = (lo + hi) / 2
            if self.ratio(req, mid) > floor:
                lo = mid
            else:
                hi = mid
        return round(hi, 2)

    def run(self, req: StressRequest, run_id: str = "adhoc", with_reverse: bool = True) -> StressResult:
        c = self._core(req, req.m)
        sc = self.lib.scenarios.get(req.scenario)
        weights = self.lib.ecl_weights
        if req.m > 0:
            adverse = self._core(req, req.m * 0.5)["ecl1"]
            ecl_weighted = weights.get("base", 0.5) * c["ecl0"] + weights.get("adverse", 0.3) * adverse + weights.get("severe", 0.2) * c["ecl1"]
        else:
            ecl_weighted = c["ecl0"]
        contributors = sorted(c["contributors"], key=lambda x: -x[0])[:10]
        curve = []
        if with_reverse:
            for m in (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
                curve.append((m, round(self.ratio(req, m), 5)))
        epi = self.book.obligors.get(req.epicenter or "")
        result = StressResult(
            run_id=run_id,
            scenario=req.scenario,
            scenario_label=sc.label if sc else req.scenario,
            impact=req.impact,
            multiplier=round(c["mult"], 4),
            m=req.m,
            epicenter=req.epicenter,
            epicenter_name=epi.name if epi else None,
            event=f"{req.event_class}/{req.subtype}" if req.event_class else None,
            defaulted=sorted(c["defaulted"]),
            downgraded=c["downgraded"],
            shocks={k: round(v, 6) for k, v in c["shock"].items()},
            value_before={k: round(v, 2) for k, v in c["before"].items()},
            value_after={k: round(v, 2) for k, v in c["after"].items()},
            pnl_by_class={k: round(v, 2) for k, v in c["pnl"].items()},
            ecl_before=round(c["ecl0"], 2),
            ecl_after=round(c["ecl1"], 2),
            ecl_weighted=round(ecl_weighted, 2),
            ecl_by_stage_before={k: round(v, 2) for k, v in c["ecl_stage0"].items()},
            ecl_by_stage_after={k: round(v, 2) for k, v in c["ecl_stage1"].items()},
            stage_flows=[{"from": f"Stage {a}", "to": f"Stage {b}", "count": n, "ead": round(e, 2)}
                         for (a, b), (n, e) in sorted(c["flows"].items())],
            heatmap=[{"sector": s, "asset_class": a, "loss": round(v, 2)} for (s, a), v in sorted(c["heat"].items())],
            top_contributors=[ContributorRow(pos_id=p.pos_id, name=p.name, asset_class=p.asset_class,
                                             sector=p.sector, loss=round(loss, 2), detail=detail)
                              for loss, p, detail in contributors if loss > 0],
            rwa_before=round(self._rwa0, 2),
            rwa_after=round(c["rwa1"], 2),
            cet1_before=round(self.cet1_start, 2),
            cet1_after=round(c["cet1_after"], 2),
            cet1_ratio_before=round(CET1_START_RATIO, 5),
            cet1_ratio_after=round(c["cet1_after"] / c["rwa1"], 5),
            pre_tax_loss=round(c["pre_tax_loss"], 2),
            reverse_m_rbi=self.reverse(req, RBI_FLOOR) if with_reverse else None,
            reverse_m_basel=self.reverse(req, BASEL_FLOOR) if with_reverse else None,
            curve=curve,
        )
        return result
