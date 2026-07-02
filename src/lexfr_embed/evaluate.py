"""Retrieval evaluation — research §04/§05.

Primary: BSARD (`BSARDRetrieval`, French-language legal IR — *Belgian* law, CC-BY-NC-SA,
eval only). Report NDCG@10, R@100, MAP, MRR against: zero-shot baselines, BM25, the
existing `camembert-base-lleqa`, and the fine-tuned model — PLUS hybrid (BM25+dense) and
+reranker (research §05: a reranker is sometimes the cheaper win — report it honestly).

Also evaluate on a **French-national held-out set** (no public benchmark exists → we build
a small one; see DECISIONS in the response / Léo #1).
"""

from __future__ import annotations


def build_ir_evaluator(queries: dict[str, str], corpus: dict[str, str], relevant: dict[str, set]):
    """queries: qid->text ; corpus: cid->text ; relevant: qid->{cid,...}."""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        ndcg_at_k=[10],
        accuracy_at_k=[1, 10],
        precision_recall_at_k=[10, 100],
        map_at_k=[100],
        mrr_at_k=[10],
        name="bsard",
    )


def load_bsard(split: str = "test"):
    """Load BSARD -> (queries, corpus, relevant_docs) for InformationRetrievalEvaluator.

    Schema confirmed (2026-06-20): two configs. `corpus` (split "corpus", 22,633 rows):
    `id` (int), `article` (text). `questions` (train=886 / test=222 / synthetic=113,165):
    `id`, `question`, `article_ids` (COMMA-SEPARATED string of corpus ids). CC-BY-NC-SA,
    ungated, no token / no trust_remote_code. Eval only (NC). Phase 0: subset the questions.
    """
    from datasets import load_dataset

    from lexfr_embed.config import settings

    corpus_ds = load_dataset(settings.bsard_id, "corpus", split="corpus")
    q_ds = load_dataset(settings.bsard_id, "questions", split=split)
    corpus = {str(r["id"]): r["article"] for r in corpus_ds}
    queries = {str(r["id"]): r["question"] for r in q_ds}
    relevant = {str(r["id"]): {cid.strip() for cid in r["article_ids"].split(",")} for r in q_ds}
    return queries, corpus, relevant


def evaluate_model(model, queries, corpus, relevant) -> dict:
    """Run the IR evaluator and return the metric dict (NDCG@10, R@100, MAP, MRR...)."""
    evaluator = build_ir_evaluator(queries, corpus, relevant)
    return evaluator(model)


def per_query_ndcg_at_k(model, queries: dict, corpus: dict, relevant: dict, k: int = 10, batch_size: int = 32):
    """Per-query NDCG@k as a list (aligned to queries order) — feeds the paired bootstrap CI.

    Integration layer (needs a model + torch); the NDCG math itself is the pure, tested
    `metrics.ndcg_at_k`. Encodes normalised, ranks top-k by cosine via semantic_search.
    """
    from sentence_transformers import util

    from lexfr_embed.metrics import ndcg_at_k

    cids, qids = list(corpus), list(queries)
    corpus_emb = model.encode(
        [corpus[c] for c in cids], batch_size=batch_size, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    query_emb = model.encode(
        [queries[q] for q in qids], batch_size=batch_size, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    hits = util.semantic_search(query_emb, corpus_emb, top_k=k)
    return [ndcg_at_k([cids[h["corpus_id"]] for h in hits[i]], relevant.get(q, set()), k) for i, q in enumerate(qids)]


if __name__ == "__main__":
    raise SystemExit("TODO: wire load_bsard() + a model path, then print the metrics table.")
