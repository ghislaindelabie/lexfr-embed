"""Contrastive fine-tuning — research §03.

Stage 1: CachedMultipleNegativesRankingLoss wrapped in MatryoshkaLoss (in-batch negatives,
truncatable dims). Stage 2: 1 mined hard negative (lower LR). LoRA for bases > ~1B, else
full fine-tune. W&B via `report_to` if a key is set (graceful fallback).

This is the skeleton wired to the real sentence-transformers v5 API; the TODOs mark where
data wiring / tuning happen. Run small first (config.base_model_key="smoke", phase0 subset).
"""

from __future__ import annotations

from lexfr_embed.config import BASE_MODELS, settings


def build_model(base_model_id: str, use_lora: bool):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(base_model_id)
    if use_lora:
        # sentence-transformers has first-class PEFT support (research §03).
        from peft import LoraConfig, TaskType

        model.add_adapter(
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=settings.lora_r,
                lora_alpha=settings.lora_alpha,
                lora_dropout=0.1,
            )
        )
    return model


def stage1_loss(model):
    from sentence_transformers.losses import (
        CachedMultipleNegativesRankingLoss,
        MatryoshkaLoss,
    )

    base = CachedMultipleNegativesRankingLoss(model)
    return MatryoshkaLoss(model, base, matryoshka_dims=settings.matryoshka_dims)
    # ABLATION (Léo #2): swap CachedMNRL -> GISTEmbedLoss with a guide model (research §03).


def train_embedder(base_model_key: str | None = None, *, use_lora: bool | None = None):
    """Two-stage train. Returns the trained SentenceTransformer.

    TODO(Phase 1): wire real data:
      train_pairs = load_legalkit(settings.phase0_subset)        # Phase 0: small subset
      # Phase 1: + synthetic_queries + target_pairs, stratified + deduped
      ds1 = Dataset.from_list([{"anchor": p["anchor"], "positive": p["positive"]} for p in train_pairs])
      ds2 = hard_negatives.mine(train_pairs, model)              # (anchor, positive, negative)
    """
    from sentence_transformers import SentenceTransformerTrainingArguments

    key = base_model_key or settings.base_model_key
    base_id = BASE_MODELS[key]
    lora = use_lora if use_lora is not None else False  # set True for >1B bases (4B/7B)

    model = build_model(base_id, use_lora=lora)

    common = dict(
        output_dir=str(settings.results_dir / f"{key}"),
        bf16=True,
        learning_rate=settings.lr_lora if lora else settings.lr_full_ft,
        warmup_ratio=0.1,
        report_to=settings.report_to,
        seed=settings.seed,
        # batch_sampler=BatchSamplers.NO_DUPLICATES  # required for in-batch-negative losses
    )

    # --- Stage 1 (in-batch negatives) ---
    args1 = SentenceTransformerTrainingArguments(num_train_epochs=settings.epochs_stage1, **common)  # noqa: F841
    loss1 = stage1_loss(model)  # noqa: F841
    # trainer1 = SentenceTransformerTrainer(model, args1, train_dataset=ds1, loss=loss1)
    # trainer1.train()

    # --- Stage 2 (1 mined hard negative; same args as Stage 1 but epochs_stage2 + half LR) ---
    # trainer2 = SentenceTransformerTrainer(model, args2, train_dataset=ds2, loss=loss1)
    # trainer2.train()

    raise NotImplementedError(
        "Wire ds1/ds2 (see TODO) then enable the two trainer blocks. Smoke path is in scripts/phase0_kaggle.py."
    )


if __name__ == "__main__":
    train_embedder()
