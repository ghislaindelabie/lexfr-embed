"""A1-bis — reranker->embedder distillation (MarginMSE). Research: PROJECT_LOG A1-bis, Tamber 2025.

The campaign proved plain/denoised hard-negative MNRL is null (A1 retired), so distillation is an
ADDITIVE third recipe, not a Stage-2 replacement: a strong cross-encoder teacher (`bge-reranker-v2-m3`)
scores mined (query, candidate) pairs OFFLINE, and the student embedder is trained to match the teacher's
*margins* via MarginMSELoss. Success is judged on the hit@k recall curve (lift hit@5 toward hit@50), not
NDCG alone.

Two design cruxes live here, both about SCALE MATCHING (see PROJECT_LOG risks #2):
  * the teacher score transform is applied EXPLICITLY (`apply_activation`), never left to CrossEncoder's
    implicit per-model default — raw logits (±10) against cosine margins (±2) would silently fail to fit;
  * the student similarity is COSINE (`util.pairwise_cos_sim`, not MarginMSE's dot-product default):
    Sigmoid teacher -> margin in [-1,1] vs cosine student -> margin in [-2,2] are comparable, so MSE fits.

The pure label math (`apply_activation`, `teacher_label_row`, `build_distill_dataset`) is hermetically
unit-tested (no torch/network); `make_distill_loss` is the thin loss wiring, tested by monkeypatch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from lexfr_embed.config import settings


def apply_activation(x: float, kind: str) -> float:
    """Map a raw reranker logit to the teacher score scale. kind: 'sigmoid' | 'identity' | 'tanh'.

    Sigmoid ([0,1]) is the default and the only scale MarginMSE fits well against a cosine student
    (see module docstring). Pure (stdlib `math`) so the label math stays torch-free and testable.
    """
    if kind == "sigmoid":
        return 1.0 / (1.0 + math.exp(-x))
    if kind == "identity":
        return float(x)
    if kind == "tanh":
        return math.tanh(x)
    raise ValueError(f"unknown activation {kind!r} (expected 'sigmoid' | 'identity' | 'tanh')")


def teacher_label_row(pos_score: float, neg_scores: list[float], activation: str) -> list[float]:
    """One MarginMSE label row: `[act(pos), act(neg_1), ..., act(neg_K)]` (length K+1, POSITIVE FIRST).

    This is the form MarginMSELoss auto-detects (labels.shape == (batch, K+1)) and internally converts
    to margins `act(pos) - act(neg_i)`; the pos-first ordering is load-bearing — swapping it flips the
    sign of every margin. `activation` is applied here so the transform is explicit and recorded in meta.
    """
    return [apply_activation(pos_score, activation)] + [apply_activation(n, activation) for n in neg_scores]


def build_distill_dataset(records: Sequence[dict]):
    """Shape cached teacher records into a MarginMSE-ready `datasets.Dataset`.

    Each record: `{"anchor", "positive", "negatives": list[str], "label": list[float]}` where
    `len(label) == len(negatives) + 1`. Columns produced: `anchor, positive, negative_1..K, label`
    with K = `settings.distill_num_negatives`. Rows with FEWER than K negatives are DROPPED (mirrors
    `mine_hard_negatives`' `indices_to_keep.all(dim=1)`): a ragged label column would otherwise crash
    the Arrow build or leak inconsistent shapes into MSE. The drop count is logged (a high count flags
    a too-small / duplicate-heavy corpus — PROJECT_LOG risk "ragged labels").
    """
    from datasets import Dataset

    k = settings.distill_num_negatives
    cols: dict[str, list] = {"anchor": [], "positive": [], "label": []}
    for j in range(1, k + 1):
        cols[f"negative_{j}"] = []

    kept = 0
    for r in records:
        negs = r["negatives"]
        if len(negs) < k:  # ragged / incomplete -> drop (guards the silent MSE bug)
            continue
        cols["anchor"].append(r["anchor"])
        cols["positive"].append(r["positive"])
        for j in range(1, k + 1):
            cols[f"negative_{j}"].append(negs[j - 1])
        cols["label"].append([float(x) for x in r["label"][: k + 1]])
        kept += 1

    dropped = len(records) - kept
    if dropped:
        print(f"[distill] dropped {dropped}/{len(records)} rows with < {k} negatives (ragged-label guard)")
    return Dataset.from_dict(cols)


def make_distill_loss(model):
    """A1-bis distill loss: `MarginMSELoss(model, similarity_fct=util.pairwise_cos_sim)`, wrapped in
    `MatryoshkaLoss` when `settings.distill_matryoshka` (mirrors `train.make_stage1_loss`).

    Cosine (not MarginMSE's dot-product default) is deliberate: on un-normalised bge-m3 embeddings a
    dot-product student against a [0,1] Sigmoid teacher is a scale mismatch that trains to ~nothing
    without crashing (documented non-default via `distill_student_sim="dot"`). MatryoshkaLoss.forward
    passes `labels` straight through to the inner loss, and only truncates a label column it mistakes
    for embeddings (size == model dim); the K+1-length label (2 for K=1) is far smaller, so it is safe.
    """
    from sentence_transformers import util
    from sentence_transformers.sentence_transformer.losses import MarginMSELoss, MatryoshkaLoss

    from lexfr_embed.train import build_matryoshka_dims

    sim_fct = util.pairwise_dot_score if settings.distill_student_sim == "dot" else util.pairwise_cos_sim
    base = MarginMSELoss(model, similarity_fct=sim_fct)
    if not settings.distill_matryoshka:
        return base
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dims = build_matryoshka_dims(get_dim(), settings.matryoshka_dims)
    return MatryoshkaLoss(model, base, matryoshka_dims=dims)
