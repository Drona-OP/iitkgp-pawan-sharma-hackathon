"""Target-aware sentiment: a trained model that scores one company inside a headline.

Training data is SEntFiN 1.0 (Sinha et al., 2023): 10,753 Economic Times headlines with 14,404
entity-level labels, 2,847 headlines naming several entities and 1,233 with conflicting
sentiment ("Sensex gains; Infosys slides"). The model is deliberately small and CPU-only:

- target masking: the entity being scored becomes the token TGT and every other linked entity
  OTH, so the model learns *relative position* ("TGT beats OTH") rather than company names;
- features: word 1-2-grams of the masked text, the same words again tagged "near" when they sit
  within three tokens of TGT, and the finance lexicon's three probabilities for the target window;
- classifier: multinomial logistic regression (scikit-learn), weights stored as JSON so the
  artefact is readable, diffable and independent of the scikit-learn version.

    python -m seismo.eval.sentiment_train data/sentfin/SEntFiN-v1.1.csv   # writes models/sentiment_target.json
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from seismo.nlp.sentiment import LexiconBackend, Probs

VERSION = "sentfin-target-lr-v1"
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b|[%$]")
NEAR = 3
LABELS = ("negative", "neutral", "positive")
_LEX = LexiconBackend()


def mask(text: str, target: list[tuple[int, int]], others: list[tuple[int, int]] | None = None) -> str:
    """Replace target spans with TGT and other entity spans with OTH (non-overlapping, left to right)."""
    spans = sorted([(s, e, " TGT ") for s, e in target] + [(s, e, " OTH ") for s, e in (others or [])])
    out, cursor = [], 0
    for s, e, tok in spans:
        if s < cursor:
            continue
        out.append(text[cursor:s])
        out.append(tok)
        cursor = e
    out.append(text[cursor:])
    return re.sub(r"\s+", " ", "".join(out)).strip()


def augmented(masked: str) -> str:
    """Masked text plus 'near' copies of the words within three tokens of TGT."""
    toks = [t.lower() for t in TOKEN_RE.findall(masked)]
    near = []
    for i, t in enumerate(toks):
        if t == "tgt":
            near += [f"near_{w}" for w in toks[max(0, i - NEAR):i] + toks[i + 1:i + 1 + NEAR] if w not in ("tgt", "oth")]
    return " ".join(toks + near)


def lexicon_features(masked: str) -> list[float]:
    """The lexicon's view of the target's own words (the clause around TGT when there is one)."""
    p = _LEX.predict([masked.replace("TGT", " ").replace("OTH", " ")])[0]
    return [p[0], p[2], p[2] - p[0]]


class TargetModelBackend:
    """Loads models/sentiment_target.json. Exposes predict() (no target) and predict_targeted()."""

    targeted = True

    def __init__(self, path: str | Path) -> None:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        self.name = spec.get("version", VERSION)
        self.vocab: dict[str, int] = spec["vocabulary"]
        self.idf: list[float] = spec["idf"]
        self.coef: list[list[float]] = spec["coef"]            # classes x (n_vocab + n_dense)
        self.intercept: list[float] = spec["intercept"]
        self.classes: list[str] = spec["classes"]
        self.dense_scale: float = float(spec.get("dense_scale", 1.0))
        self.ngram = tuple(spec.get("ngram_range", (1, 2)))
        self.metrics = spec.get("metrics", {})

    # -- the same transform as sklearn's TfidfVectorizer(sublinear_tf=True, norm="l2") on the training side
    def _tfidf(self, text: str) -> dict[int, float]:
        toks = text.split()
        counts: dict[int, int] = {}
        lo, hi = self.ngram
        for n in range(lo, hi + 1):
            for i in range(len(toks) - n + 1):
                j = self.vocab.get(" ".join(toks[i:i + n]))
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
        vec = {j: (1 + math.log(c)) * self.idf[j] for j, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {j: v / norm for j, v in vec.items()}

    def _probs(self, masked: str) -> Probs:
        x = self._tfidf(augmented(masked))
        dense = [v * self.dense_scale for v in lexicon_features(masked)]
        nv = len(self.idf)
        logits = []
        for k, row in enumerate(self.coef):
            z = self.intercept[k] + sum(row[j] * v for j, v in x.items())
            z += sum(row[nv + d] * dense[d] for d in range(len(dense)))
            logits.append(z)
        if len(logits) == 1 and len(self.classes) == 2:   # binary logistic regression stores one row
            logits = [0.0, logits[0]]
        m = max(logits)
        ex = [math.exp(z - m) for z in logits]
        tot = sum(ex)
        p = {c: e / tot for c, e in zip(self.classes, ex, strict=True)}
        return (p.get("negative", 0.0), p.get("neutral", 0.0), p.get("positive", 0.0))

    def predict(self, texts: list[str]) -> list[Probs]:
        return [self._probs(t) for t in texts]

    def predict_targeted(self, items: list[tuple[str, list[tuple[int, int]], list[tuple[int, int]]]]) -> list[Probs]:
        return [self._probs(mask(text, tgt, oth)) for text, tgt, oth in items]
