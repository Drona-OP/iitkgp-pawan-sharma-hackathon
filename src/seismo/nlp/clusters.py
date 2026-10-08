"""Story clustering, novelty chains, syndication and coordination detection.

Seismo scores events, not posts. Every (entity, document) pair joins a story cluster:
- near-duplicates (MinHash similarity >= 0.8) are syndicated copies and inherit the original
  publisher, so a wire story re-run by ten sites is still one voice;
- paraphrases join when their content words overlap enough and their event classes agree;
- a social post can attach to a news story about the same entity and event;
- novelty follows RavenPack's 24-hour chains: the first story scores 100, the n-th similar
  story within 24 hours scores 100 / n, and a gap of more than 24 hours starts a new chain;
- five or more accounts posting near-identical text within an hour form a coordinated group,
  which counts as a single voice and is flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from seismo.nlp.credibility import independence_key
from seismo.nlp.minhash import LSHIndex, MinHasher, content_words, jaccard, shingles
from seismo.schemas import Document, EventClass, SourceType, stable_id

VERSION = "minhash-lsh-v1"

DUP_SIM = 0.8            # MinHash similarity treated as the same text
JOIN_SIM = 0.5           # MinHash similarity that joins a cluster
JOIN_JACCARD = 0.3       # content-word Jaccard that joins a cluster (paraphrase)
SOCIAL_ATTACH_JACCARD = 0.1
SOCIAL_ATTACH_HOURS = 6.0
COORD_SIM = 0.7
COORD_MIN_AUTHORS = 5
COORD_WINDOW = timedelta(minutes=60)


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
        self._lsh: dict[str, LSHIndex] = {}

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

    def _score(self, cluster: Cluster, sig: tuple[int, ...], words: frozenset[str]) -> tuple[float, float, Member | None]:
        best_sim, best_jac, best_member = 0.0, 0.0, None
        for m in cluster.members[-60:]:
            sim = MinHasher.similarity(sig, m.sig)
            jac = jaccard(words, m.words)
            if sim > best_sim:
                best_sim, best_member = sim, m
            best_jac = max(best_jac, jac)
        return best_sim, best_jac, best_member

    def assign(
        self, entity_id: str, doc: Document, event: EventClass, event_conf: float
    ) -> tuple[Cluster, Member, int]:
        """Place a document in a cluster for this entity. Returns the cluster, the member and novelty."""
        text = self.key_text(doc)
        sig = self.hasher.signature(shingles(text))
        words = content_words(text)
        now = doc.published_at
        lsh = self._lsh.setdefault(entity_id, LSHIndex(self.hasher.num_perm))

        best: tuple[float, Cluster, Member | None, float] | None = None
        candidate_ids = lsh.candidates(sig) | set(self._by_entity.get(entity_id, [])[-40:])
        for cid in candidate_ids:
            c = self._clusters[cid]
            if not timedelta(0) <= now - c.last_seen <= self.window:
                continue
            sim, jac, near = self._score(c, sig, words)
            compatible = _compatible(event, c.classes())
            joins = sim >= DUP_SIM or (compatible and (sim >= JOIN_SIM or jac >= JOIN_JACCARD))
            if (
                not joins
                and doc.source_type == SourceType.SOCIAL
                and compatible
                and any(m.source_type != SourceType.SOCIAL for m in c.members)
                and (now - c.last_seen) <= timedelta(hours=SOCIAL_ATTACH_HOURS)
                and jac >= SOCIAL_ATTACH_JACCARD
            ):
                joins = True
            if joins:
                score = max(sim, jac) + 0.01 * len(c.members)
                if best is None or score > best[0]:
                    best = (score, c, near, sim)

        key = independence_key(doc.publisher, doc.source_type, doc.author)
        syndicated = False
        if best is None:
            cluster = Cluster(stable_id("clu", entity_id, doc.doc_id), entity_id, now, now)
            self._clusters[cluster.cluster_id] = cluster
            self._by_entity.setdefault(entity_id, []).append(cluster.cluster_id)
        else:
            _, cluster, near, sim = best
            if (
                near is not None and sim >= DUP_SIM and near.key != key
                and doc.source_type != SourceType.SOCIAL and near.source_type != SourceType.SOCIAL
            ):
                key, syndicated = near.key, True  # a syndicated copy speaks with the original's voice

        recent = [m for m in cluster.members if timedelta(0) <= now - m.published_at <= self.window]
        novelty = int(round(100 / (1 + len(recent))))

        member = Member(
            doc_id=doc.doc_id, published_at=now, source_type=doc.source_type, publisher=doc.publisher,
            author=doc.author, sig=sig, words=words, key=key, event=event, event_conf=event_conf,
            title=doc.title or None, url=doc.url, syndicated=syndicated,
        )
        cluster.members.append(member)
        cluster.last_seen = max(cluster.last_seen, now)
        lsh.add(cluster.cluster_id, sig)
        if doc.source_type == SourceType.SOCIAL:
            self._group_social(cluster, member)
        return cluster, member, novelty

    @staticmethod
    def _group_social(cluster: Cluster, member: Member) -> None:
        author = member.author or member.doc_id
        for g in cluster.groups:
            if member.published_at - g.start <= COORD_WINDOW and MinHasher.similarity(g.root, member.sig) >= COORD_SIM:
                g.authors.add(author)
                g.doc_ids.append(member.doc_id)
                return
        cluster.groups.append(SocialGroup(member.sig, member.published_at, {author}, [member.doc_id]))

    def dispute(self, cluster: Cluster, doc: Document) -> None:
        cluster.disputed = True
        cluster.disputed_at = doc.published_at
        if doc.doc_id not in cluster.disputed_by:
            cluster.disputed_by.append(doc.doc_id)

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
