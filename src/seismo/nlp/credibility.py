"""Source credibility and authority. Reputation comes from the publisher, never from a badge."""

from __future__ import annotations

from seismo.schemas import SourceType

DOMAIN_CREDIBILITY: dict[str, float] = {
    "sec.gov": 1.0, "federalreserve.gov": 1.0, "rbi.org.in": 1.0, "ecb.europa.eu": 1.0,
    "treasury.gov": 1.0, "reuters.com": 0.95, "apnews.com": 0.95, "bloomberg.com": 0.95,
    "wsj.com": 0.9, "ft.com": 0.9, "cnbc.com": 0.85, "barrons.com": 0.85, "nytimes.com": 0.85,
    "marketwatch.com": 0.8, "finance.yahoo.com": 0.75, "economictimes.indiatimes.com": 0.8,
    "livemint.com": 0.8, "business-standard.com": 0.8, "thehindubusinessline.com": 0.8,
    "moneycontrol.com": 0.75,
}

# Regulators, filings and the major wires can corroborate an event on their own.
AUTHORITATIVE = frozenset(
    {"sec.gov", "federalreserve.gov", "rbi.org.in", "ecb.europa.eu", "treasury.gov",
     "reuters.com", "apnews.com", "bloomberg.com"}
)

DEFAULT_BY_TYPE = {SourceType.NEWS: 0.6, SourceType.SOCIAL: 0.3, SourceType.FILING: 1.0}


def base_domain(publisher: str) -> str:
    host = (publisher or "").lower().strip()
    host = host.split("://")[-1].split("/")[0]
    return host[4:] if host.startswith("www.") else host


def _lookup(domain: str, table: dict[str, float] | frozenset[str]) -> str | None:
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in table:
            return candidate
    return None


def credibility(publisher: str, source_type: SourceType) -> float:
    key = _lookup(base_domain(publisher), DOMAIN_CREDIBILITY)
    if key is not None and source_type != SourceType.SOCIAL:
        return DOMAIN_CREDIBILITY[key]
    return DEFAULT_BY_TYPE[source_type]


def is_authoritative(publisher: str, source_type: SourceType) -> bool:
    if source_type == SourceType.FILING:
        return True
    if source_type == SourceType.SOCIAL:
        return False
    return _lookup(base_domain(publisher), AUTHORITATIVE) is not None
