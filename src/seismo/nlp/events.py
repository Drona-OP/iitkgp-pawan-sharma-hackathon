"""Event classification v0: SEC 8-K item codes plus weighted keyword rules.

Day 2 adds a learned classifier; these rules then serve as weak labels and a fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from seismo.schemas import Document, EventClass, EventLabel, SourceType

E = EventClass
METHOD = "rules-v0"

# 8-K item -> (class, subtype, severity rank). Rank picks the headline item in multi-item filings.
ITEM_MAP: dict[str, tuple[EventClass, str, int]] = {
    "1.03": (E.CREDIT_EVENT, "BANKRUPTCY", 10),
    "2.04": (E.CREDIT_EVENT, "ACCELERATION_TRIGGER", 9),
    "1.05": (E.OPERATIONAL_ESG, "CYBER_INCIDENT", 8),
    "4.02": (E.LEGAL_REGULATORY, "RESTATEMENT", 8),
    "3.01": (E.LEGAL_REGULATORY, "DELISTING_NOTICE", 7),
    "5.01": (E.MA_CORPORATE_ACTION, "CHANGE_IN_CONTROL", 7),
    "2.06": (E.EARNINGS_GUIDANCE, "IMPAIRMENT", 6),
    "2.01": (E.MA_CORPORATE_ACTION, "ACQUISITION_COMPLETED", 6),
    "2.02": (E.EARNINGS_GUIDANCE, "RESULTS", 6),
    "1.01": (E.MA_CORPORATE_ACTION, "MATERIAL_AGREEMENT", 5),
    "1.02": (E.MA_CORPORATE_ACTION, "AGREEMENT_TERMINATED", 5),
    "2.05": (E.OPERATIONAL_ESG, "RESTRUCTURING", 5),
    "5.02": (E.MANAGEMENT_GOVERNANCE, "OFFICER_CHANGE", 5),
    "4.01": (E.MANAGEMENT_GOVERNANCE, "AUDITOR_CHANGE", 5),
    "2.03": (E.MA_CORPORATE_ACTION, "DEBT_ISSUANCE", 3),
}

# Tie-break order when two classes score the same: more severe classes first.
PRIORITY = [
    E.CREDIT_EVENT, E.GEOPOLITICAL, E.MACROECONOMIC, E.LEGAL_REGULATORY, E.OPERATIONAL_ESG,
    E.EARNINGS_GUIDANCE, E.MA_CORPORATE_ACTION, E.PRODUCT_STRATEGY, E.MANAGEMENT_GOVERNANCE,
    E.OTHER,
]


@dataclass(frozen=True)
class Rule:
    event: EventClass
    subtype: str
    weight: float
    pattern: re.Pattern[str]


def _r(event: EventClass, subtype: str, weight: float, pattern: str) -> Rule:
    return Rule(event, subtype, weight, re.compile(pattern, re.IGNORECASE))


RULES: tuple[Rule, ...] = (
    _r(E.CREDIT_EVENT, "BANKRUPTCY", 3.0, r"\bbankrupt\w*|\bchapter 11\b|\breceivership\b|\binsolven\w*"),
    _r(E.CREDIT_EVENT, "DEFAULT", 3.0, r"\bdefault(?:ed|s)? on\b|\bmissed (?:a |an |its )?(?:coupon|interest|debt) payment"),
    _r(E.CREDIT_EVENT, "DOWNGRADE", 2.5, r"\b(?:credit )?rating (?:cut|downgrade)\b|\bdowngrade[sd]? (?:its |the )?(?:credit|debt|rating)\b|\bcut to junk\b|\bjunk status\b"),
    _r(E.CREDIT_EVENT, "BANK_RUN", 3.0, r"\bbank run\b|\bdeposit (?:flight|outflows?|withdrawals?)\b|\b(?:pull|pulls|pulled|pulling|withdraw|withdrawing|withdrew|move|moving) (?:their |our |its )?(?:deposits|cash|money|funds)\b|\blimit(?:s|ing|ed)? withdrawals\b|\bhalt(?:s|ing|ed)? (?:customer )?withdrawals\b|\bwithdrawal limits?\b|\bbank (?:collapse|failure)s?\b|\bcollapses?\b.{0,40}\bbank\b|\bclosed by (?:\w+ )?regulators?\b|\bFDIC\b"),
    _r(E.CREDIT_EVENT, "LIQUIDITY_STRESS", 2.0, r"\bliquidity (?:crisis|crunch|squeeze|fears?|concerns?)\b|\bcovenant breach\b|\bdebt restructuring\b|\bshore up (?:its )?(?:balance sheet|capital|finances)\b|\bemergency (?:capital|funding|lending|backstop)\b|\bcapital raise\b|\bbank failures?\b|\bcontagion\b"),
    _r(E.GEOPOLITICAL, "TRADE_WAR", 2.5, r"\btariffs?\b|\btrade war\b|\bimport dut(?:y|ies)\b|\bretaliatory\b"),
    _r(E.GEOPOLITICAL, "SANCTIONS", 2.5, r"\bsanctions?\b|\bembargo\b|\bexport controls?\b"),
    _r(E.GEOPOLITICAL, "ARMED_CONFLICT", 3.0, r"(?<!trade )\bwar\b|\binvasion\b|\binvade[sd]?\b|\bmissiles?\b|\bairstrikes?\b|\bmilitary strikes?\b|\bceasefire\b"),
    _r(E.GEOPOLITICAL, "POLITICAL_INSTABILITY", 2.0, r"\bcoup\b|\bterror(?:ist|ism)?\b|\bunrest\b|\bsnap election\b"),
    _r(E.MACROECONOMIC, "MONETARY_POLICY", 2.5, r"\binterest rates?\b|\brate (?:hike|cut|decision)s?\b|\brepo rate\b|\bbasis points?\b|\bFOMC\b|\bFederal Reserve\b|\bcentral bank\b|\bmonetary policy\b"),
    _r(E.MACROECONOMIC, "INFLATION", 2.0, r"\binflation\b|\bCPI\b|\bconsumer prices\b"),
    _r(E.MACROECONOMIC, "GROWTH_JOBS", 2.0, r"\bGDP\b|\brecession\b|\bjobs report\b|\bpayrolls\b|\bunemployment\b"),
    _r(E.MACROECONOMIC, "RATES_MARKET", 1.5, r"\btreasury yields?\b|\bbond yields?\b|\b10-year yield\b"),
    _r(E.MACROECONOMIC, "PANDEMIC", 2.5, r"\bpandemic\b|\boutbreak\b|\blockdowns?\b|\bcoronavirus\b|\bCOVID(?:-19)?\b"),
    _r(E.MACROECONOMIC, "COMMODITIES", 2.0, r"\boutput cuts?\b|\boil prices?\b|\bcrude\b|\bOPEC\b"),
    _r(E.MA_CORPORATE_ACTION, "MERGER_ACQUISITION", 2.5, r"\bacquir(?:e|es|ed|ing)\b|\bacquisition\b|\bmerger\b|\btakeover\b|\bbuyout\b|\bdeal to buy\b"),
    _r(E.MA_CORPORATE_ACTION, "DIVESTITURE", 2.0, r"\bdivest\w*|\bspin[- ]?off\b|\bsell (?:a |its |the )?stake\b|\bsell (?:its|the) \w+ (?:unit|division|business)\b"),
    _r(E.MA_CORPORATE_ACTION, "CAPITAL_RETURN", 2.0, r"\bbuybacks?\b|\bshare repurchase\b|\bdividend\b"),
    _r(E.MA_CORPORATE_ACTION, "IPO", 2.0, r"\bIPO\b|\binitial public offering\b"),
    _r(E.EARNINGS_GUIDANCE, "RESULTS", 2.0, r"\bearnings\b|\bquarterly (?:results|profit|revenue)\b|\bprofit estimates\b|\bEPS\b|\bbeats? (?:estimates|expectations)\b|\bmiss(?:es|ed)? (?:estimates|expectations)\b"),
    _r(E.EARNINGS_GUIDANCE, "GUIDANCE", 2.5, r"\bguidance\b|\boutlook\b|\bforecast\b"),
    _r(E.EARNINGS_GUIDANCE, "ANALYST_ACTION", 1.5, r"\bprice targets?\b|\banalysts? (?:upgrade|downgrade)s?\b|\b(?:upgrade|downgrade)[sd]? to (?:buy|sell|hold|neutral|overweight|underweight)\b"),
    _r(E.PRODUCT_STRATEGY, "COMPETITIVE_THREAT", 2.0, r"\brivals?\b|\bcompetitors?\b|\bcheaper\b|\bdisrupt\w*"),
    _r(E.PRODUCT_STRATEGY, "LAUNCH", 2.0, r"\blaunch(?:es|ed)?\b|\bunveil(?:s|ed)?\b|\breleases? (?:a |its )?(?:new|cheaper)\b|\bnew (?:model|product|lineup|chip)\b"),
    _r(E.PRODUCT_STRATEGY, "RECALL", 2.5, r"\brecalls?\b|\brecalled\b"),
    _r(E.PRODUCT_STRATEGY, "PARTNERSHIP", 1.5, r"\bpartnership\b|\bpartners with\b|\bteams up\b"),
    _r(E.LEGAL_REGULATORY, "LITIGATION", 2.0, r"\blawsuit\b|\bsued\b|\bsues\b|\bcourt\b|\bjudge\b|\bsettlement\b"),
    _r(E.LEGAL_REGULATORY, "ENFORCEMENT", 2.5, r"\bantitrust\b|\bprobe\b|\binvestigation\b|\bfined\b|\bfines? of\b|\bpenalt(?:y|ies)\b|\bregulators?\b"),
    _r(E.LEGAL_REGULATORY, "APPROVAL", 2.0, r"\bFDA\b|\bapproval\b|\bapproved\b|\bclearance\b"),
    _r(E.OPERATIONAL_ESG, "CYBER_INCIDENT", 3.0, r"\bcyber ?attack\b|\bhack(?:ed|ers)?\b|\bdata breach\b|\bransomware\b"),
    _r(E.OPERATIONAL_ESG, "OUTAGE_ACCIDENT", 2.5, r"\boutage\b|\bexplosion\b|\bfire at\b|\baccident\b|\bplane crash\b|\bgrounded\b|\bgrounding\b"),
    _r(E.OPERATIONAL_ESG, "LABOR", 2.0, r"\bstrike\b|\bwalkout\b|\blayoffs?\b|\bjob cuts\b"),
    _r(E.OPERATIONAL_ESG, "ENVIRONMENTAL", 2.0, r"\boil spill\b|\bemissions\b|\bpollution\b"),
    _r(E.MANAGEMENT_GOVERNANCE, "EXECUTIVE_CHANGE", 2.5, r"\b(?:CEO|CFO|chief executive|chairman)\b.{0,40}\b(?:steps? down|resign\w*|ousted|replaced|retire\w*|appointed|named)\b|\b(?:steps? down|resign\w*|appoint\w*|names?|named)\b.{0,40}\b(?:CEO|CFO|chief executive)\b"),
    _r(E.MANAGEMENT_GOVERNANCE, "ACCOUNTING", 2.5, r"\baccounting (?:irregularit\w*|probe|issues?)\b|\bauditor\b|\brestate\w*"),
    _r(E.MANAGEMENT_GOVERNANCE, "FRAUD_ALLEGATION", 3.0, r"\b(?:accounting |corporate )?fraud\b|\bfraudulent\b|\bstock manipulation\b|\bmanipulat(?:e|es|ed|ing|ion)\b.{0,30}\b(?:shares?|stocks?|prices?)\b|\bshort[- ]sell(?:er|ers|ing)?\b.{0,60}\b(?:report|allegations?|claims?|attack)\b|\b(?:largest|biggest) con\b|\bshell compan(?:y|ies)\b|\boffshore (?:shell|entities|funds)\b"),
)


def _from_scores(scores: dict[EventClass, float], subtypes: dict[EventClass, tuple[float, str]]) -> EventLabel:
    if not scores:
        return EventLabel(primary=E.OTHER, subtype=None, confidence=0.3, labels=[E.OTHER], method=METHOD)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], PRIORITY.index(kv[0])))
    top_cls, top = ranked[0]
    share = top / sum(scores.values())
    strength = min(1.0, top / 3.0)
    conf = round(0.3 + 0.65 * share * strength, 3)
    labels = [cls for cls, s in ranked if s >= 0.5 * top]
    return EventLabel(primary=top_cls, subtype=subtypes[top_cls][1], confidence=conf, labels=labels, method=METHOD)


def classify_event(doc: Document) -> EventLabel:
    text, title = doc.text, doc.title
    scores: dict[EventClass, float] = {}
    subtypes: dict[EventClass, tuple[float, str]] = {}
    for rule in RULES:
        n = len(rule.pattern.findall(text))
        if not n:
            continue
        in_title = bool(title) and bool(rule.pattern.search(title))
        s = rule.weight * (1 + 0.5 * min(n - 1, 2)) * (1.5 if in_title else 1.0)
        scores[rule.event] = scores.get(rule.event, 0.0) + s
        if s > subtypes.get(rule.event, (0.0, ""))[0]:
            subtypes[rule.event] = (s, rule.subtype)
    keyword_label = _from_scores(scores, subtypes)

    items = [str(i) for i in doc.meta.get("items", [])] if doc.source_type == SourceType.FILING else []
    mapped = [ITEM_MAP[i] for i in items if i in ITEM_MAP]
    if mapped:
        cls, subtype, _ = max(mapped, key=lambda m: m[2])
        labels = list(dict.fromkeys([m[0] for m in mapped] + keyword_label.labels))
        labels = [lbl for lbl in labels if lbl != E.OTHER] or [cls]
        return EventLabel(primary=cls, subtype=subtype, confidence=0.95, labels=labels, method=f"8k-items+{METHOD}")
    return keyword_label
