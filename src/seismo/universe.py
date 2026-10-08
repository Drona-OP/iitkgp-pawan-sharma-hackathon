"""Entity master: two index universes, a watchlist of extra obligors, and macro entities.

Index members (data/universe.csv, 20 S&P 100 names) are what Module A trades by default; the
India universe (data/universe_in.csv, 16 Nifty 50 names) is a second index Module A can run.
Watchlist names (data/watchlist.csv), such as the regional banks at the centre of March 2023,
are linked by the engine and can be obligors in Module B, but are never index constituents.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from seismo.schemas import EntityRef


@dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: str  # "company" | "macro"
    name: str
    aliases: tuple[str, ...] = ()
    ambiguous_aliases: tuple[str, ...] = ()
    cik: str | None = None
    sector: str | None = None
    cashtags: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    index_member: bool = False
    index: str | None = None   # "US" or "IN" for index members, None otherwise
    group: str = ""            # business group ("Adani"), for group-level risk rules


# Macro entities let policy, commodity and market-wide news produce signals
# even when no company is named. Aliases with a capital letter match case-sensitively.
MACRO_ENTITIES: tuple[Entity, ...] = (
    Entity("MACRO:FED", "macro", "US Federal Reserve",
           aliases=("Federal Reserve", "FOMC", "the Fed", "The Fed", "Fed officials", "Fed chair")),
    Entity("MACRO:ECB", "macro", "European Central Bank", aliases=("European Central Bank", "ECB")),
    Entity("MACRO:RBI", "macro", "Reserve Bank of India", aliases=("Reserve Bank of India", "RBI")),
    Entity("MACRO:OPEC", "macro", "OPEC+", aliases=("OPEC+", "OPEC")),
    Entity("MACRO:OIL", "macro", "Crude oil",
           aliases=("crude oil", "oil prices", "Brent crude", "WTI crude")),
    Entity("MACRO:RATES", "macro", "US Treasury yields",
           aliases=("Treasury yields", "bond yields", "10-year yield")),
    Entity("MACRO:MARKET", "macro", "US equity market",
           aliases=("Wall Street", "S&P 500", "Dow Jones", "stock market")),
    Entity("MACRO:MARKET_IN", "macro", "Indian equity market",
           aliases=("Nifty 50", "Nifty", "Sensex", "Dalal Street", "Indian stocks", "Indian markets")),
    Entity("MACRO:SEBI", "macro", "Securities and Exchange Board of India",
           aliases=("Securities and Exchange Board of India", "SEBI", "Sebi")),
)


def _split(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split("|") if part.strip())


class Universe:
    def __init__(self, entities: list[Entity]) -> None:
        self._by_id = {e.entity_id: e for e in entities}
        self._by_cik = {e.cik.lstrip("0"): e for e in entities if e.cik}
        self._by_cashtag = {tag.upper(): e for e in entities for tag in e.cashtags}

    @staticmethod
    def _read(csv_path: Path, index_member: bool, index: str | None = None) -> list[Entity]:
        companies: list[Entity] = []
        with csv_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                ticker = row["ticker"].strip().upper()
                cik = (row.get("cik") or "").strip()
                companies.append(
                    Entity(
                        entity_id=ticker,
                        entity_type="company",
                        name=row["name"].strip(),
                        aliases=_split(row.get("aliases")),
                        ambiguous_aliases=_split(row.get("ambiguous_aliases")),
                        cik=cik.zfill(10) if cik else None,
                        sector=(row.get("sector") or "").strip() or None,
                        cashtags=(ticker, *_split(row.get("extra_cashtags"))),
                        domains=_split(row.get("domains")),
                        index_member=index_member,
                        index=index if index_member else None,
                        group=(row.get("group") or "").strip(),
                    )
                )
        return companies

    @classmethod
    def load(cls, csv_path: str | Path, watchlist_path: str | Path | None = None) -> Universe:
        csv_path = Path(csv_path)
        companies = cls._read(csv_path, index_member=True, index="US")
        india = csv_path.with_name("universe_in.csv")
        if india.exists():
            known = {c.entity_id for c in companies}
            companies += [e for e in cls._read(india, index_member=True, index="IN") if e.entity_id not in known]
        watch = Path(watchlist_path) if watchlist_path else csv_path.with_name("watchlist.csv")
        if watch.exists():
            known = {c.entity_id for c in companies}
            companies += [e for e in cls._read(watch, index_member=False) if e.entity_id not in known]
        return cls([*companies, *MACRO_ENTITIES])

    @property
    def entities(self) -> list[Entity]:
        return list(self._by_id.values())

    def companies(self) -> list[Entity]:
        return [e for e in self._by_id.values() if e.entity_type == "company"]

    def index_members(self, index: str = "US") -> list[Entity]:
        return [e for e in self._by_id.values() if e.index_member and e.index == index]

    def macros(self) -> list[Entity]:
        return [e for e in self._by_id.values() if e.entity_type == "macro"]

    def get(self, entity_id: str) -> Entity | None:
        return self._by_id.get(entity_id)

    def by_cik(self, cik: str | int) -> Entity | None:
        return self._by_cik.get(str(cik).lstrip("0"))

    def by_cashtag(self, tag: str) -> Entity | None:
        return self._by_cashtag.get(tag.upper())

    def ref(self, entity_id: str) -> EntityRef:
        e = self._by_id[entity_id]
        return EntityRef(type=e.entity_type, id=e.entity_id, name=e.name, cik=e.cik, sector=e.sector)
