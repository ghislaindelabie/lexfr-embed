"""Track-B v2 — the BM25/TF-IDF-hard, full-corpus, qrels-repaired successor to Track-B v1.

Forensics verdict on v1 (2026-07-06, results/day/trackb_forensics.json): a zero-ML TF-IDF
retriever solves 89.6% of v1 at rank 1 (98% at rank 10) because LLaMA-3 queries carry ~70%
of their content words verbatim from the gold article -> v1 measures lexical paraphrase
matching, and every dense model sits at the 0.97 ceiling. v2 fixes the *discriminative*
defect (it does NOT fix registre/professional realism — that remains P0):

  1. corpus     = ALL deduped LegalKit articles (~52k; trained articles as distractors only);
  2. pure-leak  : drop held-out queries whose gold TEXT is byte-identical to a trained
                  positive (~1.1% — already-seen text);
  3. hard filter: keep only queries where TF-IDF FAILS at rank 1 on that full corpus
                  (fail@1 ≈ 18% -> n ≈ 2.8k, MDE ≈ 0.018 at sd 0.35);
  4. qrels fix  : articles at TF-IDF cos > 0.8 of a gold are added as EXTRA golds
                  (multi-label) — the enlarged corpus would otherwise manufacture
                  false negatives (~8% of golds have such near-dups).

Build once with scripts/build_trackb_v2.py -> results/trackb_v2.json.gz; evaluate via
eval_extra --split trackb2. Caveats to attach to any v2 number: same-generator queries
(hard-in-distribution, not external truth); qrels still 1-few golds per query.
"""

from __future__ import annotations

import numpy as np

_NEARDUP_THRESHOLD = 0.8


def drop_pure_leaks(heldout_pairs: list[dict], trained_texts: set[str]) -> list[dict]:
    """Remove held-out pairs whose gold article text also appears as a trained positive."""
    return [p for p in heldout_pairs if p["positive"] not in trained_texts]


def _tfidf(corpus_texts: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(sublinear_tf=True, lowercase=True, dtype=np.float32)
    return vec, vec.fit_transform(corpus_texts)


def lexical_top1_fail_mask(queries: list[str], gold_idx: list[int], corpus_texts: list[str]) -> np.ndarray:
    """True where a zero-ML TF-IDF retriever does NOT rank the gold article first (= hard query)."""
    vec, xc = _tfidf(corpus_texts)
    xq = vec.transform(queries)
    fail = np.zeros(len(queries), dtype=bool)
    for s in range(0, xq.shape[0], 512):
        sims = (xq[s : s + 512] @ xc.T).toarray()
        top1 = sims.argmax(axis=1)
        wrong = top1 != np.asarray(gold_idx[s : s + len(top1)])
        no_signal = sims.max(axis=1) <= 0  # zero lexical overlap anywhere = definitionally hard
        fail[s : s + len(top1)] = wrong | no_signal
    return fail


def neardup_extra_golds(
    gold_idx: list[int], corpus_texts: list[str], threshold: float = _NEARDUP_THRESHOLD
) -> dict[int, set[int]]:
    """For each gold article index, corpus indices whose TF-IDF cosine exceeds `threshold`
    (excluding the gold itself) — unlabeled near-positives to fold into the qrels."""
    _, xc = _tfidf(corpus_texts)
    uniq = sorted(set(gold_idx))
    out: dict[int, set[int]] = {}
    for s in range(0, len(uniq), 256):
        rows = uniq[s : s + 256]
        sims = (xc[rows] @ xc.T).toarray()
        for local, g in enumerate(rows):
            close = np.where(sims[local] > threshold)[0]
            out[g] = {int(c) for c in close if c != g}
    return out


def build_v2_eval(heldout_pairs: list[dict], distractor_texts: list[str], trained_texts: set[str]):
    """Full v2 construction -> (queries, corpus, relevant) in the load_bsard shape.

    corpus = deduped(gold articles of surviving pool + distractor_texts); relevant is
    multi-label after near-dup repair. Deterministic (no sampling — the hard filter IS
    the selection).
    """
    pool = drop_pure_leaks(heldout_pairs, trained_texts)

    text2idx: dict[str, int] = {}
    corpus_texts: list[str] = []

    def _idx(t: str) -> int:
        if t not in text2idx:
            text2idx[t] = len(corpus_texts)
            corpus_texts.append(t)
        return text2idx[t]

    gold_idx = [_idx(p["positive"]) for p in pool]
    for t in distractor_texts:
        _idx(t)

    fail = lexical_top1_fail_mask([p["anchor"] for p in pool], gold_idx, corpus_texts)
    survivors = [(p, g) for (p, g), f in zip(zip(pool, gold_idx, strict=True), fail, strict=True) if f]

    extra = neardup_extra_golds([g for _, g in survivors], corpus_texts)
    corpus = {f"a{i}": t for i, t in enumerate(corpus_texts)}
    queries: dict[str, str] = {}
    relevant: dict[str, set[str]] = {}
    for i, (p, g) in enumerate(survivors):
        qid = f"q{i}"
        queries[qid] = p["anchor"]
        relevant[qid] = {f"a{g}"} | {f"a{e}" for e in extra.get(g, set())}
    return queries, corpus, relevant
