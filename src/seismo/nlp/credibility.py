"""Source credibility, authority and independence. Reputation comes from the publisher, never a badge.

Three ideas from the 2023 fake-Pentagon-explosion episode, where a paid-verified account
impersonating Bloomberg briefly moved the S&P 500:
- credibility is looked up by domain, so a checkmark or display name earns nothing;
- a domain that borrows a known brand without being that brand ("bloomberg-news.co") is a
  lookalike and is scored below an anonymous source;
- independence is counted by owner, so WSJ, MarketWatch and Barron's are one voice, not three.
"""

from __future__ import annotations

from seismo.schemas import SourceType

DOMAIN_CREDIBILITY: dict[str, float] = {
    "sec.gov": 1.0, "federalreserve.gov": 1.0, "rbi.org.in": 1.0, "ecb.europa.eu": 1.0,
    "treasury.gov": 1.0, "fdic.gov": 1.0, "reuters.com": 0.95, "apnews.com": 0.95,
    "bloomberg.com": 0.95, "wsj.com": 0.9, "ft.com": 0.9, "cnbc.com": 0.85, "barrons.com": 0.85,
    "nytimes.com": 0.85, "marketwatch.com": 0.8, "finance.yahoo.com": 0.75,
    "economictimes.indiatimes.com": 0.8, "livemint.com": 0.8, "business-standard.com": 0.8,
    "thehindubusinessline.com": 0.8, "moneycontrol.com": 0.75, "axios.com": 0.8,
    "washingtonpost.com": 0.85, "bbc.co.uk": 0.85, "bbc.com": 0.85, "theguardian.com": 0.8,
}

# Regulators, filings and the major wires can corroborate an event on their own.
AUTHORITATIVE = frozenset(
    {"sec.gov", "federalreserve.gov", "rbi.org.in", "ecb.europa.eu", "treasury.gov", "fdic.gov",
     "reuters.com", "apnews.com", "bloomberg.com"}
)

# Common ownership: outlets in one group count as one independent publisher.
OWNER_GROUP: dict[str, str] = {
    "wsj.com": "dow-jones", "marketwatch.com": "dow-jones", "barrons.com": "dow-jones",
    "dowjones.com": "dow-jones", "cnbc.com": "nbcuniversal", "nbcnews.com": "nbcuniversal",
    "finance.yahoo.com": "yahoo", "yahoo.com": "yahoo", "ft.com": "nikkei", "nikkei.com": "nikkei",
    "economictimes.indiatimes.com": "times-group", "timesofindia.indiatimes.com": "times-group",
    "livemint.com": "ht-media", "hindustantimes.com": "ht-media",
    "moneycontrol.com": "network18", "cnbctv18.com": "network18",
    "bbc.co.uk": "bbc", "bbc.com": "bbc",
}

# Brand tokens a lookalike domain or handle tends to borrow.
BRANDS = ("reuters", "bloomberg", "apnews", "associatedpress", "wsj", "wallstreetjournal",
          "cnbc", "marketwatch", "barrons", "financialtimes", "nytimes", "federalreserve", "secgov")

DEFAULT_BY_TYPE = {SourceType.NEWS: 0.6, SourceType.SOCIAL: 0.3, SourceType.FILING: 1.0}
LOOKALIKE_CREDIBILITY = 0.15


def base_domain(publisher: str) -> str:
    host = (publisher or "").lower().strip()
    host = host.split("://")[-1].split("/")[0]
    return host[4:] if host.startswith("www.") else host


def _lookup(domain: str, table: dict[str, float] | dict[str, str] | frozenset[str]) -> str | None:
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in table:
            return candidate
    return None


def is_lookalike(publisher: str, author: str | None = None) -> bool:
    """True when a domain or handle carries a known brand but is not that brand's domain."""
    for raw in (publisher, author or ""):
        domain = base_domain(raw)
        if not domain or _lookup(domain, DOMAIN_CREDIBILITY) is not None:
            continue
        squashed = domain.replace("-", "").replace("_", "").replace(".", "")
        if any(brand in squashed for brand in BRANDS):
            return True
    return False


def credibility(publisher: str, source_type: SourceType, author: str | None = None) -> float:
    if is_lookalike(publisher, author):
        return LOOKALIKE_CREDIBILITY
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


def is_official(publisher: str, entity_domains: tuple[str, ...]) -> bool:
    """The issuer speaking for itself, e.g. a statement on newsroom.bankofamerica.com."""
    domain = base_domain(publisher)
    return any(domain == d or domain.endswith("." + d) for d in entity_domains)


def independence_key(publisher: str, source_type: SourceType, author: str | None = None) -> str:
    """Who is really speaking: owner group for news, the filer for filings, the account for social."""
    if source_type == SourceType.FILING:
        return "filing:" + (base_domain(publisher) or "sec.gov")
    if source_type == SourceType.SOCIAL:
        return "social:" + (author or base_domain(publisher) or "unknown")
    domain = base_domain(publisher)
    group = _lookup(domain, OWNER_GROUP)
    return "news:" + (OWNER_GROUP[group] if group else domain)
