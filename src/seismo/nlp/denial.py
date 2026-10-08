"""Denials and official contradictions: the trigger for a retraction.

A document counts as a denial when it uses denial language and comes from someone entitled to
deny: an SEC filing, an authoritative wire or regulator, or the issuer's own domain. A random
account saying "fake" does not retract anything.
"""

from __future__ import annotations

import re

from seismo.nlp.credibility import is_authoritative, is_lookalike, is_official
from seismo.schemas import Document

VERSION = "denial-rules-v1"

DENIAL_RE = re.compile(
    r"\b(?:den(?:y|ies|ied|ying)|refute[sd]?|debunk(?:s|ed)?|false|fake|fabricated|hoax|baseless|"
    r"unfounded|untrue|not true|no truth|misinformation|disinformation|did not happen|never happened|"
    r"(?:is|are|remain|remains) (?:fully )?(?:open|operational|operating normally)|"
    r"operating normally|no (?:such|explosion|incident|halt|suspension)|"
    r"(?:has|have) not (?:halted|suspended|frozen|limited|paused)|"
    r"(?:not|never) (?:halted|suspended|frozen|limited|paused))\b",
    re.IGNORECASE,
)


def has_denial_language(text: str) -> bool:
    return bool(DENIAL_RE.search(text))


def is_denial(doc: Document, entity_domains: tuple[str, ...] = ()) -> bool:
    if not has_denial_language(doc.text):
        return False
    if is_lookalike(doc.publisher, doc.author):
        return False
    if bool(doc.meta.get("official")) and is_official(doc.publisher, entity_domains):
        return True
    return is_authoritative(doc.publisher, doc.source_type) or is_official(doc.publisher, entity_domains)
