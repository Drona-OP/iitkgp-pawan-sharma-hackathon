"""Entity master: the 20-company universe (CSV) plus macro entities."""

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

    @classmethod
    def load(cls, csv_path: str | Path) -> Universe:
        companies: list[Entity] = []
        with Path(csv_path).open(encoding="utf-8", newline="") as fh:
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
                    )
                )
        return cls([*companies, *MACRO_ENTITIES])

    @property
    def entities(self) -> list[Entity]:
        return list(self._by_id.values())

    def companies(self) -> list[Entity]:
        return [e for e in self._by_id.values() if e.entity_type == "company"]

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
