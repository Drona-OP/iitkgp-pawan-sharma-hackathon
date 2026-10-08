"""Event clustering, novelty chains, syndication and coordination detection.

Seismo scores events, not posts. Every (entity, document) pair joins an event cluster:
- an event is one entity and one event class within a rolling 24-hour chain, so "SVB shares
  plunge" and "SVB customers pull deposits" are two reports of one credit event;
- commentary without its own event class (class OTHER) attaches to the entity's live event
  from the last six hours, so "who's next?" posts count as reach for the run, not as new events;
- novelty follows RavenPack's 24-hour chains on the text: the first story scores 100, a story
  similar to n earlier ones (MinHash or content-word overlap) scores 100 / (1 + n), and a gap of
  more than 24 hours starts a new chain;
- near-duplicates (MinHash similarity >= 0.8) are syndicated copies and inherit the original
  publisher, so a wire story re-run by ten sites is still one voice;
- five or more accounts posting near-identical text within an hour form a coordinated group,
  which counts as a single voice and is flagged;
- a denial retracts a story only while the story is unconfirmed. Once three independent
  newsrooms (or one authoritative source) carry it, a denial makes it *contested*: kept live and
  flagged, because a rumour dies when it is denied but a confirmed story does not (the Adani
  group called the Hindenburg report baseless within hours; its shares then halved).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from seismo.nlp.credibility import independence_key
from seismo.nlp.minhash import MinHasher, content_words, jaccard, shingles
from seismo.schemas import Document, EventClass, SourceType, stable_id

VERSION = "event-chain-minhash-v1"

DUP_SIM = 0.8            # MinHash similarity treated as the same text
JOIN_SIM = 0.5           # MinHash similarity that joins a cluster
JOIN_JACCARD = 0.3       # content-word Jaccard that joins a cluster (paraphrase)
SOCIAL_ATTACH_HOURS = 6.0
COORD_SIM = 0.7
COORD_MIN_AUTHORS = 5
COORD_WINDOW = timedelta(minutes=60)
CONTEST_MIN_PUBLISHERS = 3   # independent newsrooms after which a denial contests, not retracts


@dataclass
class Member:
    doc_id: str
    published_at: datetime
    source_type: SourceType
    publisher: str
    author: str | None
    sig: tuple[int, ...]
    words: frozenset[str]
    key: str                       # independence key after syndication collapse
    event: EventClass
    event_conf: float
    title: str | None = None
    url: str | None = None
    # Filled in by the engine once the document is scored:
    sentiment: float = 0.0
    sent_conf: float = 0.0
    relevance: int = 0
    credibility: float = 0.5
    authoritative: bool = False
    impact: int = 1
    span: str = ""
    subtype: str | None = None
    lookalike: bool = False
    syndicated: bool = False


@dataclass
class SocialGroup:
    root: tuple[int, ...]
    start: datetime
    authors: set[str] = field(default_factory=set)
    doc_ids: list[str] = field(default_factory=list)

    @property
    def coordinated(self) -> bool:
        return len(self.authors) >= COORD_MIN_AUTHORS


@dataclass
class Cluster:
    cluster_id: str
    entity_id: str
    first_seen: datetime
    last_seen: datetime
    members: list[Member] = field(default_factory=list)
    groups: list[SocialGroup] = field(default_factory=list)
    disputed: bool = False
    disputed_by: list[str] = field(default_factory=list)
    disputed_at: datetime | None = None
    contested: bool = False
    contested_by: list[str] = field(default_factory=list)

    def corroborated(self) -> bool:
        """Confirmed by >= 3 independent newsrooms or one authoritative source (no social, no lookalikes)."""
        keys = {m.key for m in self.members if m.source_type != SourceType.SOCIAL and not m.lookalike}
        authoritative = any(m.authoritative and not m.lookalike for m in self.members)
        return len(keys) >= CONTEST_MIN_PUBLISHERS or authoritative

    def classes(self) -> set[EventClass]:
        return {m.event for m in self.members}

    def coordinated_doc_ids(self) -> set[str]:
        return {d for g in self.groups if g.coordinated for d in g.doc_ids}

    def coordinated_key(self, doc_id: str) -> str | None:
        for i, g in enumerate(self.groups):
            if g.coordinated and doc_id in g.doc_ids:
                return f"coordinated:{self.cluster_id}:{i}"
        return None


def _compatible(a: EventClass, classes: set[EventClass]) -> bool:
    return a == EventClass.OTHER or EventClass.OTHER in classes or a in classes


class StoryClusterer:
    def __init__(self, window_hours: float = 24.0, num_perm: int = 64) -> None:
        self.window = timedelta(hours=window_hours)
        self.hasher = MinHasher(num_perm=num_perm)
        self._clusters: dict[str, Cluster] = {}
        self._by_entity: dict[str, list[str]] = {}
        self._history: dict[str, list[Member]] = {}

    @staticmethod
    def key_text(doc: Document) -> str:
        """Titles carry the story; posts and title-less items use the first 300 characters."""
        return doc.title if doc.title else doc.body[:300]

    def get(self, cluster_id: str) -> Cluster | None:
        return self._clusters.get(cluster_id)

    def active(self, entity_id: str, now: datetime, horizon: timedelta | None = None) -> list[Cluster]:
        horizon = horizon or self.window
        out = []
        for cid in self._by_entity.get(entity_id, []):
            c = self._clusters[cid]
            if timedelta(0) <= now - c.last_seen <= horizon or (c.last_seen > now and c.first_seen <= now):
                out.append(c)
        return out

    def assign(
        self, entity_id: str, doc: Document, event: EventClass, event_conf: float
    ) -> tuple[Cluster, Member, int]:
        """Place a document in an event cluster for this entity. Returns cluster, member, novelty."""
        text = self.key_text(doc)
        sig = self.hasher.signature(shingles(text))
        words = content_words(text)
        now = doc.published_at

        # Novelty and syndication look at every recent story about the entity, across events.
        history = [h for h in self._history.get(entity_id, []) if timedelta(0) <= now - h.published_at <= self.window]
        self._history[entity_id] = history
        similar = 0
        near: Member | None = None
        near_sim = 0.0
        for h in history:
            sim = MinHasher.similarity(sig, h.sig)
            if sim >= JOIN_SIM or jaccard(words, h.words) >= JOIN_JACCARD:
                similar += 1
            if sim > near_sim:
                near, near_sim = h, sim
        novelty = int(round(100 / (1 + similar)))

        key = independence_key(doc.publisher, doc.source_type, doc.author)
        syndicated = False
        if (
            near is not None and near_sim >= DUP_SIM and near.key != key
            and doc.source_type != SourceType.SOCIAL and near.source_type != SourceType.SOCIAL
        ):
            key, syndicated = near.key, True  # a syndicated copy speaks with the original's voice

        cluster = self._pick(entity_id, now, event)
        if cluster is None:
            cluster = Cluster(stable_id("clu", entity_id, doc.doc_id), entity_id, now, now)
            self._clusters[cluster.cluster_id] = cluster
            self._by_entity.setdefault(entity_id, []).append(cluster.cluster_id)

        member = Member(
            doc_id=doc.doc_id, published_at=now, source_type=doc.source_type, publisher=doc.publisher,
            author=doc.author, sig=sig, words=words, key=key, event=event, event_conf=event_conf,
            title=doc.title or None, url=doc.url, syndicated=syndicated,
        )
        cluster.members.append(member)
        cluster.last_seen = max(cluster.last_seen, now)
        history.append(member)
        if doc.source_type == SourceType.SOCIAL:
            self._group_social(cluster, member)
        return cluster, member, novelty

    def _pick(self, entity_id: str, now: datetime, event: EventClass) -> Cluster | None:
        live = [c for c in self.active(entity_id, now) if not c.disputed]
        live.sort(key=lambda c: c.last_seen, reverse=True)
        if event == EventClass.OTHER:
            attach = timedelta(hours=SOCIAL_ATTACH_HOURS)
            return next((c for c in live if now - c.last_seen <= attach), None)
        same = [c for c in live if event in c.classes()]
        if same:
            return same[0]
        attach = timedelta(hours=SOCIAL_ATTACH_HOURS)
        return next((c for c in live if not c.classes() - {EventClass.OTHER} and now - c.last_seen <= attach), None)

    @staticmethod
    def _group_social(cluster: Cluster, member: Member) -> None:
        author = member.author or member.doc_id
        for g in cluster.groups:
            if member.published_at - g.start <= COORD_WINDOW and MinHasher.similarity(g.root, member.sig) >= COORD_SIM:
                g.authors.add(author)
                g.doc_ids.append(member.doc_id)
                return
        cluster.groups.append(SocialGroup(member.sig, member.published_at, {author}, [member.doc_id]))

    def dispute(self, cluster: Cluster, doc: Document) -> str:
        """Apply a denial: "retracted" for an unconfirmed story, "contested" for a confirmed one."""
        if cluster.corroborated() and not cluster.disputed:
            cluster.contested = True
            if doc.doc_id not in cluster.contested_by:
                cluster.contested_by.append(doc.doc_id)
            return "contested"
        cluster.disputed = True
        cluster.disputed_at = doc.published_at
        if doc.doc_id not in cluster.disputed_by:
            cluster.disputed_by.append(doc.doc_id)
        return "retracted"

    def best_match(
        self, entity_id: str, doc: Document, horizon_hours: float = 48.0
    ) -> Cluster | None:
        """The live story a denial most plausibly refers to: overlapping words, most recent first."""
        text_words = content_words(doc.text)
        now = doc.published_at
        scored = []
        for c in self.active(entity_id, now, timedelta(hours=horizon_hours)):
            if doc.doc_id in {m.doc_id for m in c.members}:
                continue
            overlap = max((jaccard(text_words, m.words) for m in c.members), default=0.0)
            weight = sum(abs(m.sentiment) + m.impact / 10 for m in c.members)
            scored.append((overlap, weight, c.last_seen, c))
        scored = [s for s in scored if s[0] >= 0.08]
        if not scored:
            return None
        scored.sort(key=lambda s: (s[0] * (1 + min(s[1], 10) / 10), s[2]), reverse=True)
        return scored[0][3]
