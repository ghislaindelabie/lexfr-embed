"""Stage-2 hard-negative mining — research §02/§03.

Noumon's lesson: for <~10k pairs, **1 filtered hard negative per query** is the sweet
spot; mining many raw negatives *regresses* results. Use a margin filter to avoid
false negatives.

A1 — false-negative denoising (implemented): the `relative_margin` filter on *embedder*
similarity is cheap but imperfect. French legal queries are frequently **multi-label**, so a
mined "negative" can actually be a real (unlabelled) relevant article — a *false negative* that
teaches the model the wrong boundary. Pass a **cross-encoder reranker** (`cross_encoder=`, a
model from a different family than the embedder) to `mine()`: candidates are RE-SCORED by the
reranker and the margin/`max_score` filters then apply to those scores, dropping candidates the
reranker rates ~as relevant as the positive (RocketQA NAACL-2021: 70% of mined negatives were
actually positives; denoising gave +10.35 MRR). Gate/sample for cost, and log the reject rate
(a high one flags a noisy mining step). See docs/training-data-strategy.md R4 and PROJECT_LOG A1.
"""

from __future__ import annotations


def pairs_to_anchor_positive_dict(train_pairs: list[dict]) -> dict:
    """Pure: [{anchor, positive, code?}, ...] -> {"anchor": [...], "positive": [...]}.

    Drops everything but anchor/positive (MNRL wants only those two columns). Raises on an
    empty list, or a pair missing 'anchor'/'positive'.
    """
    if not train_pairs:
        raise ValueError("no training pairs")
    anchors, positives = [], []
    for p in train_pairs:
        if "anchor" not in p or "positive" not in p:
            raise KeyError("each pair needs 'anchor' and 'positive'")
        anchors.append(p["anchor"])
        positives.append(p["positive"])
    return {"anchor": anchors, "positive": positives}


def mine(
    train_pairs,
    model,
    *,
    num_negatives: int = 1,
    relative_margin: float = 0.05,
    cross_encoder=None,
    max_score: float | None = None,
):
    """Mine `num_negatives` filtered hard negative(s) per query with `model` (the Stage-1 model).

    Returns an (anchor, positive, negative) triplet Dataset. `relative_margin` drops candidates
    too similar to the positive. When `cross_encoder` is given (A1 denoising), candidates are
    re-scored by the reranker and the margin/`max_score` filters apply to those scores, dropping
    false negatives (mined "negatives" the reranker rates ~as relevant as the positive). `max_score`
    is an optional hard cap on the reranker score above which a candidate is dropped.
    """
    from datasets import Dataset
    from sentence_transformers.util import mine_hard_negatives

    dataset = Dataset.from_dict(pairs_to_anchor_positive_dict(train_pairs))
    kwargs = {
        "num_negatives": num_negatives,
        "relative_margin": relative_margin,
        "sampling_strategy": "top",
        "use_faiss": True,
        "output_format": "triplet",
    }
    if cross_encoder is not None:  # A1: denoise on reranker scores (RocketQA-style)
        kwargs["cross_encoder"] = cross_encoder
        if max_score is not None:
            kwargs["max_score"] = max_score
    return mine_hard_negatives(dataset, model, **kwargs)


def mine_teacher_candidates(train_pairs, model, *, num_negatives: int = 1):
    """A1-bis Step A: mine `num_negatives` top-K candidates per query with the STAGE-1 embedder ONLY.

    Unlike `mine()`, this passes NO `cross_encoder` and NO `relative_margin`/`max_score` — the teacher
    reranker scores candidates in a SEPARATE explicit pass (Step B), so distillation sees soft labels
    for every candidate (including possible false negatives, which it handles softly). This split is the
    mitigation for SILENT-BUG #1: `mine_hard_negatives` only re-scores with a `cross_encoder` when a
    margin/`max_score` filter is set, so a naive `mine(..., cross_encoder=reranker)` with no filter would
    silently emit EMBEDDER cosine sims — degenerating into embedder self-distillation. Returns an
    `n-tuple` Dataset (anchor, positive, negative_1..K); ragged queries (< K negatives found) are dropped
    by `mine_hard_negatives` itself.
    """
    from datasets import Dataset
    from sentence_transformers.util import mine_hard_negatives

    dataset = Dataset.from_dict(pairs_to_anchor_positive_dict(train_pairs))
    return mine_hard_negatives(
        dataset,
        model,
        num_negatives=num_negatives,
        sampling_strategy="top",
        output_format="n-tuple",
        use_faiss=True,
    )
