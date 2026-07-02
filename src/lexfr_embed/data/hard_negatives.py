"""Stage-2 hard-negative mining — research §02/§03.

Noumon's lesson: for <~10k pairs, **1 filtered hard negative per query** is the sweet
spot; mining many raw negatives *regresses* results. Use a margin filter to avoid
false negatives.

Phase-2 enhancement (optional — documented for later use): the `relative_margin` filter
is cheap but imperfect. French legal queries are frequently **multi-label**, so a mined
"negative" can actually be a real (unlabelled) relevant article — a *false negative* that
teaches the model the wrong boundary if used. Before trusting mined negatives at scale,
add a second confirmation pass: score each (query, candidate) with a **cross-encoder
reranker** (e.g. a multilingual/legal cross-encoder) or an **LLM judge**, and DROP any
candidate it rates relevant above a threshold; keep only the confirmed-irrelevant ones.
Gate/sample it (cost), use a model from a different family than the embedder, and log how
many candidates it rejects (a high reject rate flags a noisy mining step). See
docs/training-data-strategy.md R4.
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


def mine(train_pairs, model, *, num_negatives: int = 1, relative_margin: float = 0.05):
    """Mine `num_negatives` filtered hard negative(s) per query with `model` (the Stage-1 model).

    Returns an (anchor, positive, negative) triplet Dataset. `relative_margin` drops candidates
    too similar to the positive (likely false negatives — see the module note above).
    """
    from datasets import Dataset
    from sentence_transformers.util import mine_hard_negatives

    dataset = Dataset.from_dict(pairs_to_anchor_positive_dict(train_pairs))
    return mine_hard_negatives(
        dataset,
        model,
        num_negatives=num_negatives,
        relative_margin=relative_margin,
        sampling_strategy="top",
        use_faiss=True,
        output_format="triplet",
    )
