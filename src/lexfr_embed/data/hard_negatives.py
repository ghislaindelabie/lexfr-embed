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


def mine(train_pairs, model, *, num_negatives: int = 1, relative_margin: float = 0.05):
    """Wrap sentence-transformers' mine_hard_negatives with our defaults.

    TODO(Phase 1): convert train_pairs -> a Dataset(anchor, positive), then:

        from sentence_transformers.util import mine_hard_negatives
        mined = mine_hard_negatives(
            dataset, model,
            num_negatives=num_negatives,        # 1 (research §03)
            relative_margin=relative_margin,    # neg <= 95% as similar as positive
            sampling_strategy="top",
            use_faiss=True,
        )
        return mined  # (anchor, positive, negative) triplets
    """
    raise NotImplementedError("Wrap sentence_transformers.util.mine_hard_negatives.")
