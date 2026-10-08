"""Real-news replay packs from GDELT 2.0: real article URLs, real publishers, real timestamps.

    python -m seismo.ingest.gdelt_replay     # data/gdelt/*.csv -> data/replay/*_gdelt.jsonl

scripts/fetch_data.py (`--only gdelt`) keeps every article in GDELT's 15-minute event exports
whose URL matches a window's terms (SVB week, Adani-Hindenburg week, a quiet control week).
GDELT stores the URL but not the headline, so the headline is rebuilt from the URL slug
("/adani-group-stocks-fall-after-hindenburg-report/" -> "Adani Group Stocks Fall After Hindenburg
Report"). Slugs that carry fewer than three words are dropped. The timestamp is GDELT's
DATEADDED, the 15-minute slot in which GDELT first saw the article, so it can trail the true
publication time by up to about 15-30 minutes. Nothing here is synthetic, and every record says so.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from seismo.config import load_settings

VERSION = "gdelt-slug-v1"
MIN_WORDS = 3
MAX_PER_DOMAIN_SLOT = 3   # syndication floods: at most 3 articles per domain per 15-minute slot

WINDOWS = {
    # window: (pack name, header)
    "svb_2023": ("svb_2023_gdelt", "REAL NEWS: article URLs and GDELT timestamps, 8-10 March 2023 (SVB). Headlines rebuilt from URL slugs."),
    "adani_2023": ("adani_2023_gdelt", "REAL NEWS: article URLs and GDELT timestamps, 24-27 January 2023 (Adani-Hindenburg). Headlines rebuilt from URL slugs."),
    "control_2024": ("control_2024_gdelt", "REAL NEWS: article URLs and GDELT timestamps, 7-8 May 2024, a quiet control window. Headlines rebuilt from URL slugs."),
}

_ID_RE = re.compile(r"^(?:[0-9]+|[0-9a-f]{8,}|[a-z]?[0-9]{5,}[a-z0-9]*|index|amp|story|news|article|articleshow)$", re.IGNORECASE)
_EXT_RE = re.compile(r"\.(?:s?html?|php|aspx?|cms|jsp|ece|story)$", re.IGNORECASE)
_DATE_RE = re.compile(r"^(?:19|20)\d{2}(?:[01]\d(?:[0-3]\d)?)?$")
_SMALL = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "vs", "with", "from", "after", "over"}


def _acronyms() -> set[str]:
    """All-caps aliases in the universes (SVB, SBI, FDIC, RBI ...) keep their capitals in a slug."""
    root = load_settings().root
    out = {"US", "UK", "EU", "IPO", "FPO", "CEO", "CFO", "AI", "FDIC", "SEC", "RBI", "SEBI", "LIC", "NSE", "BSE",
           "SVB", "SBI", "UBS", "ETF", "GDP", "CPI", "FOMC", "ECB", "IMF", "NYSE", "LSE", "JPMORGAN"}
    for name in ("universe.csv", "universe_in.csv", "watchlist.csv"):
        path = root / "data" / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                for alias in (row.get("aliases") or "").split("|") + (row.get("extra_cashtags") or "").split("|"):
                    for word in alias.split():
                        if word.isupper() and word.isalpha() and len(word) >= 2:
                            out.add(word)
    out.discard("JPMORGAN")
    return out


def headline_from_url(url: str, acronyms: set[str] | None = None) -> str | None:
    """The most word-like path segment, title-cased; None when the URL carries no headline."""
    parts = [p for p in urlparse(url).path.split("/") if p]
    best: list[str] = []
    for part in parts:
        part = _EXT_RE.sub("", part)
        words = [w for w in re.split(r"[-_+]+", part) if w]
        words = [w for w in words if not _ID_RE.match(w) and not _DATE_RE.match(w)]
        words = [w for w in words if re.fullmatch(r"[A-Za-z][A-Za-z0-9'&.]*|\d+(?:\.\d+)?%?", w)]
        if len(words) > len(best):
            best = words
    if len(best) < MIN_WORDS:
        return None
    acronyms = acronyms or set()
    out = []
    for i, w in enumerate(best[:24]):
        low = w.lower()
        if w.upper() in acronyms:
            out.append(w.upper())
        elif i and low in _SMALL:
            out.append(low)
        else:
            out.append(low[:1].upper() + low[1:])
    return " ".join(out)


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def build_pack(csv_path: Path, out_path: Path, pack: str, header: str) -> int:
    acronyms = _acronyms()
    rows = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            title = headline_from_url(r["url"], acronyms)
            if not title:
                continue
            when = datetime.strptime(r["date_added"], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
            rows.append((when, r["url"], title, r.get("gdelt_tone", "")))
    rows.sort()
    seen_titles: set[tuple[str, str]] = set()
    per_slot: dict[tuple[str, datetime], int] = {}
    kept = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {header}\n")
        fh.write(f"# Built by python -m seismo.ingest.gdelt_replay ({VERSION}) from {csv_path.name}. Real articles; \"synthetic\": false.\n")
        for when, url, title, tone in rows:
            domain = _domain(url)
            key = (domain, title.lower())
            if key in seen_titles:
                continue
            slot = (domain, when)
            if per_slot.get(slot, 0) >= MAX_PER_DOMAIN_SLOT:
                continue
            seen_titles.add(key)
            per_slot[slot] = per_slot.get(slot, 0) + 1
            kept += 1
            rec = {
                "doc_id": f"{pack}_{kept:05d}",
                "source": "gdelt",
                "source_type": "news",
                "publisher": domain,
                "published_at": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "title": title,
                "body": "",
                "url": url,
                "meta": {"pack": pack, "gdelt_tone": tone, "headline_from": "url_slug"},
                "synthetic": False,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return kept


def main() -> int:
    root = load_settings().root
    built = 0
    for window, (pack, header) in WINDOWS.items():
        src = root / "data" / "gdelt" / f"{window}.csv"
        if not src.exists():
            print(f"{src.relative_to(root)} missing: run `python scripts/fetch_data.py --only gdelt` first")
            continue
        n = build_pack(src, root / "data" / "replay" / f"{pack}.jsonl", pack, header)
        built += 1
        print(f"data/replay/{pack}.jsonl: {n} real articles")
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(main())
