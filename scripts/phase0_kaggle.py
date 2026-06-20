"""Phase-0 walking skeleton — runs the WHOLE pipeline end-to-end on free Kaggle GPU.

Goal (research §07): smallest model + LegalKit subset -> train -> eval on BSARD subset ->
a baseline->fine-tuned delta, for the Thu-25-Jun mentor gate. Prove the pipeline, not SOTA.

Paste into a Kaggle notebook cell (GPU: T4 x2). It checkpoints to the HF Hub so a session
timeout doesn't lose work. Cost: $0.

    !pip install -q sentence-transformers datasets
    # then run this file's main()

Outline (fill the two TODOs marked PHASE-0):
"""

from __future__ import annotations


def main() -> None:
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

    from lexfr_embed.config import settings
    from lexfr_embed.data.legalkit import load_legalkit
    from lexfr_embed.evaluate import evaluate_model, load_bsard

    # 1) DATA — small, stratified, deduped subset (research §03/§04)
    pairs = load_legalkit(subset_size=settings.phase0_subset)  # PHASE-0: keep small (~10-20k)
    ds = Dataset.from_list([{"anchor": p["anchor"], "positive": p["positive"]} for p in pairs])

    # 2) BASELINE — zero-shot eval BEFORE training (this delta is half the mentor gate)
    queries, corpus, relevant = load_bsard()  # PHASE-0: subset of questions + article corpus
    base_model_id = settings.base_model_id  # Phase 0: consider settings.base_model_key="smoke"
    model = SentenceTransformer(base_model_id)
    before = evaluate_model(model, queries, corpus, relevant)
    print("ZERO-SHOT:", before)

    # 3) TRAIN — Stage 1 only for Phase 0 (fast); CachedMNRL+Matryoshka in Phase 1
    loss = MatryoshkaLoss(model, MultipleNegativesRankingLoss(model), matryoshka_dims=settings.matryoshka_dims)
    args = SentenceTransformerTrainingArguments(
        output_dir="results/phase0",
        num_train_epochs=1,
        per_device_train_batch_size=settings.batch_size,
        learning_rate=settings.lr_full_ft,
        warmup_ratio=0.1,
        bf16=True,
        report_to=settings.report_to,
        save_strategy="epoch",  # checkpoint so a session timeout doesn't lose work
    )
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()

    # 4) EVAL AFTER — the delta to show the mentor
    after = evaluate_model(model, queries, corpus, relevant)
    print("FINE-TUNED:", after)
    print("DELTA NDCG@10:", after.get("bsard_cosine_ndcg@10", 0) - before.get("bsard_cosine_ndcg@10", 0))

    # 5) CHECKPOINT — push to HF Hub (set HF_TOKEN) so results survive the session
    # model.push_to_hub("ghislaindelabie/lexfr-embed-fr-phase0")  # TODO: confirm HF namespace


if __name__ == "__main__":
    main()
