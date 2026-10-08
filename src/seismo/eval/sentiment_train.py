"""Train and evaluate the target-aware sentiment model on SEntFiN 1.0 (real, labelled headlines).

    python -m seismo.eval.sentiment_train data/sentfin/SEntFiN-v1.1.csv

Split: 80/20 by headline (every entity of a headline lands on the same side), seed 2026.
C is chosen by 5-fold grouped cross-validation on the training part only; the test part is
scored once. Compared on the same test pairs:

- lexicon, whole headline      one score per headline, copied to every entity (the naive way)
- lexicon, entity window       Seismo's previous default (clause around the entity)
- trained, no masking          same features without TGT/OTH (ablation: what masking adds)
- trained, target-masked       the shipped model

Writes models/sentiment_target.json (weights as JSON) and docs/results/sentiment_report.json.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from seismo.config import load_settings
from seismo.nlp.sentiment import LexiconBackend
from seismo.nlp.target_model import LABELS, VERSION, augmented, lexicon_features, mask

SEED = 2026
DENSE_SCALE = 1.0


def _parse_decisions(raw: str) -> dict[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        d = ast.literal_eval(raw)
    return {str(k): str(v).strip().lower() for k, v in d.items()}


def _find(title: str, entity: str) -> list[tuple[int, int]]:
    spans = [(m.start(), m.end()) for m in re.finditer(rf"(?<!\w){re.escape(entity)}(?!\w)", title, re.IGNORECASE)]
    if not spans and " " in entity:
        head = entity.split()[0]
        if len(head) >= 4:
            spans = [(m.start(), m.end()) for m in re.finditer(rf"(?<!\w){re.escape(head)}(?!\w)", title, re.IGNORECASE)]
    return spans


def load_sentfin(path: Path) -> list[dict]:
    """One row per (headline, entity): title, target spans, other-entity spans, label, headline id."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    cols = {c.lower().strip(): c for c in rows[0]}
    title_col = next(cols[c] for c in cols if c in ("title", "headline", "news headline", "text"))
    dec_col = next(cols[c] for c in cols if c in ("decisions", "entity-sentiment", "entity_sentiment", "labels"))
    out, skipped = [], 0
    for hid, r in enumerate(rows):
        title = (r[title_col] or "").strip()
        decisions = _parse_decisions(r[dec_col])
        spans = {e: _find(title, e) for e in decisions}
        for ent, lab in decisions.items():
            if lab not in LABELS or not spans[ent]:
                skipped += 1
                continue
            others = [sp for o, sps in spans.items() if o != ent for sp in sps]
            out.append({"hid": hid, "title": title, "entity": ent, "target": spans[ent], "others": others,
                        "label": lab, "n_entities": len(decisions),
                        "conflict": len(set(decisions.values())) > 1})
    print(f"SEntFiN: {len(rows)} headlines -> {len(out)} entity pairs ({skipped} skipped: entity not found in text)")
    return out


def _features(rows: list[dict], masked: bool):
    import numpy as np

    texts, dense = [], []
    for r in rows:
        m = mask(r["title"], r["target"], r["others"]) if masked else r["title"]
        texts.append(augmented(m))
        dense.append([v * DENSE_SCALE for v in lexicon_features(m)])
    return texts, np.asarray(dense)


def _fit(train: list[dict], masked: bool, C: float):
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts, dense = _features(train, masked)
    vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=(1, 2),
                          min_df=2, sublinear_tf=True, max_features=40000)
    X = hstack([vec.fit_transform(texts), csr_matrix(dense)]).tocsr()
    y = [r["label"] for r in train]
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit(X, y)
    return vec, clf


def _predict(vec, clf, rows: list[dict], masked: bool) -> list[str]:
    from scipy.sparse import csr_matrix, hstack

    texts, dense = _features(rows, masked)
    return list(clf.predict(hstack([vec.transform(texts), csr_matrix(dense)]).tocsr()))


def _scores(y: list[str], p: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    present = sorted(set(y) | set(p))
    return {"accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, average="macro", labels=present, zero_division=0)), 4),
            "n": len(y)}


def to_spec(vec, clf, trained_on: str, metrics: dict | None = None) -> dict:
    """The fitted vectorizer and classifier as plain JSON (what TargetModelBackend loads)."""
    return {
        "version": VERSION,
        "trained_on": trained_on,
        "classes": [str(c) for c in clf.classes_],
        "vocabulary": {k: int(v) for k, v in vec.vocabulary_.items()},
        "idf": [round(float(v), 6) for v in vec.idf_],
        "coef": [[round(float(v), 6) for v in row] for row in clf.coef_],
        "intercept": [round(float(v), 6) for v in clf.intercept_],
        "ngram_range": [1, 2],
        "dense_scale": DENSE_SCALE,
        "dense_features": ["lexicon_p_neg", "lexicon_p_pos", "lexicon_balance"],
        "metrics": metrics or {},
    }


def _lex_label(p, eps: float = 0.1) -> str:
    s = p[2] - p[0]
    return "neutral" if abs(s) < eps else ("positive" if s > 0 else "negative")


def _window_probs(lex, r: dict):
    """Seismo v1: the lexicon on the entity's own sentence or clause (seismo.nlp.sentiment.entity_windows)."""
    from datetime import UTC, datetime

    from seismo.nlp.sentiment import average, entity_windows
    from seismo.schemas import Document, EntityMention, SourceType

    doc = Document(doc_id="s", source="sentfin", source_type=SourceType.NEWS, publisher="sentfin",
                   published_at=datetime(2015, 1, 1, tzinfo=UTC), title=r["title"])
    mention = EntityMention(entity_id="E", entity_type="company", name=r["entity"], surface=r["entity"],
                            spans=r["target"], ner_score=1.0, ned_score=1.0, relevance=100, method="gold")
    ws = entity_windows(doc, mention, others=r["others"])
    return average(lex.predict([doc.text[a:b] for a, b in ws]))


def main(argv: list[str]) -> int:
    import numpy as np
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    settings = load_settings()
    root = settings.root
    path = Path(argv[0]) if argv else next(iter(sorted((root / "data" / "sentfin").glob("*.csv"))), None)
    if path is None or not path.exists():
        print("Usage: python -m seismo.eval.sentiment_train path/to/SEntFiN-v1.1.csv")
        return 1
    rows = load_sentfin(path)
    groups = np.array([r["hid"] for r in rows])
    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(rows, groups=groups))
    train = [rows[i] for i in tr_idx]
    test = [rows[i] for i in te_idx]

    # C by grouped 5-fold CV on the training part only.
    best_c, best = 1.0, -1.0
    gtrain = np.array([r["hid"] for r in train])
    for C in (0.5, 1.0, 2.0, 4.0, 8.0):
        f1s = []
        for a, b in GroupKFold(n_splits=5).split(train, groups=gtrain):
            vec, clf = _fit([train[i] for i in a], True, C)
            val = [train[i] for i in b]
            f1s.append(_scores([r["label"] for r in val], _predict(vec, clf, val, True))["macro_f1"])
        score = float(np.mean(f1s))
        print(f"  C={C:<4} grouped 5-fold macro-F1 {score:.4f}")
        if score > best:
            best_c, best = C, score

    y = [r["label"] for r in test]
    lex = LexiconBackend()
    whole = [_lex_label(lex.predict([r["title"]])[0]) for r in test]
    window = [_lex_label(_window_probs(lex, r)) for r in test]
    vec_u, clf_u = _fit(train, False, best_c)
    unmasked = _predict(vec_u, clf_u, test, False)
    vec, clf = _fit(train, True, best_c)
    masked = _predict(vec, clf, test, True)

    conflict = [i for i, r in enumerate(test) if r["conflict"]]
    multi = [i for i, r in enumerate(test) if r["n_entities"] > 1]

    def sub(pred, idx):
        return _scores([y[i] for i in idx], [pred[i] for i in idx])

    systems = {
        "Lexicon, whole headline (naive)": whole,
        "Lexicon, entity window (Seismo v1)": window,
        "Trained, no target masking (ablation)": unmasked,
        "Trained, target-masked (shipped)": masked,
    }
    report = {
        "dataset": "SEntFiN 1.0 (Sinha et al., 2023): Economic Times headlines with entity-level labels",
        "file": path.name,
        "pairs": len(rows), "headlines": int(len(set(groups))),
        "label_mix": dict(Counter(r["label"] for r in rows)),
        "split": f"80/20 by headline, seed {SEED}; C={best_c} by grouped 5-fold CV on train (macro-F1 {best:.4f})",
        "test": {name: {"all": _scores(y, p), "multi_entity": sub(p, multi), "conflicting": sub(p, conflict)}
                 for name, p in systems.items()},
        "published_reference": "SEntFiN paper, test set: finBERT and RoBERTa fine-tuned reach about 93-94% "
                               "(GPU transformers); best lexicon + GBM about 85%.",
        "version": VERSION,
    }

    # Ship: refit on all data with the chosen C, store weights as JSON.
    vec_all, clf_all = _fit(rows, True, best_c)
    spec = to_spec(vec_all, clf_all, f"SEntFiN 1.0, {len(rows)} entity pairs ({path.name}); test metrics from an "
                   "80/20 headline split", report["test"]["Trained, target-masked (shipped)"])
    vocab = spec["vocabulary"]
    models = root / "models"
    models.mkdir(exist_ok=True)
    (models / "sentiment_target.json").write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")

    # The JSON scorer must reproduce scikit-learn exactly (up to rounding of the stored weights).
    from scipy.sparse import csr_matrix, hstack

    from seismo.nlp.target_model import TargetModelBackend

    backend = TargetModelBackend(models / "sentiment_target.json")
    probe = rows[:: max(1, len(rows) // 200)]
    texts, dense = _features(probe, True)
    ref = clf_all.predict_proba(hstack([vec_all.transform(texts), csr_matrix(dense)]).tocsr())
    order = [list(clf_all.classes_).index(c) for c in ("negative", "neutral", "positive")]
    got = backend.predict_targeted([(r["title"], r["target"], r["others"]) for r in probe])
    err = max(abs(g[i] - ref[n][order[i]]) for n, g in enumerate(got) for i in range(3))
    print(f"JSON scorer vs scikit-learn: max probability difference {err:.2e} on {len(probe)} pairs")
    if err > 1e-3:
        raise SystemExit("JSON scorer does not reproduce the trained model")
    out = root / "docs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sentiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{report['dataset']}\n{report['split']}\n")
    print(f"{'System':42s} {'acc':>6s} {'F1':>6s} | {'multi F1':>8s} | {'conflict F1':>11s}")
    for name, m in report["test"].items():
        print(f"{name:42s} {m['all']['accuracy']:6.3f} {m['all']['macro_f1']:6.3f} | "
              f"{m['multi_entity']['macro_f1']:8.3f} | {m['conflicting']['macro_f1']:11.3f}")
    print(f"\nWrote models/sentiment_target.json ({len(vocab)} features) and docs/results/sentiment_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
