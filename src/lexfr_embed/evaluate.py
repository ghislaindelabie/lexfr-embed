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


if __name__ == "__main__":
    raise SystemExit("TODO: wire load_bsard() + a model path, then print the metrics table.")
