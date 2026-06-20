"""End-to-end smoke test (gated behind --run-smoke; needs network + torch).

Proves the *pipeline* wiring on a tiny model + dummy data in seconds — encode, a 1-step
contrastive train, and an IR-eval call all run without error. Not a quality check.
"""

import pytest

pytestmark = pytest.mark.smoke


def test_pipeline_end_to_end_tiny():
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    pairs = [
        {"anchor": "délai de prescription", "positive": "Article 2224 du Code civil"},
        {"anchor": "résiliation du bail", "positive": "Article 1224 du Code civil"},
        {"anchor": "rupture du contrat de travail", "positive": "Article L1231-1 du Code du travail"},
        {"anchor": "vice caché", "positive": "Article 1641 du Code civil"},
    ]
    ds = Dataset.from_list(pairs)

    args = SentenceTransformerTrainingArguments(
        output_dir="results/_smoke",
        max_steps=1,
        per_device_train_batch_size=2,
        report_to="none",
        logging_steps=1,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=ds, loss=MultipleNegativesRankingLoss(model)
    )
    trainer.train()

    # eval call wiring
    from lexfr_embed.evaluate import evaluate_model

    queries = {"q1": "prescription"}
    corpus = {"a1": "Article 2224 du Code civil", "a2": "Article 1641 du Code civil"}
    relevant = {"q1": {"a1"}}
    metrics = evaluate_model(model, queries, corpus, relevant)
    assert isinstance(metrics, dict)
