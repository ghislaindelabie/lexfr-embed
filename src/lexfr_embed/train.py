"""Contrastive fine-tuning — Phase-1 two-stage recipe (research §03; blueprint).

Stage 1: MNRL wrapped in MatryoshkaLoss (in-batch negatives). Plain `MultipleNegativesRankingLoss`
is the DEFAULT (the validated phase0 path); `CachedMultipleNegativesRankingLoss` is opt-in via
`settings.use_cached_mnrl` (gated on the CPU smoke). Stage 2: 1 mined hard negative, half LR.
LoRA for large bases. Between stages: free memory + empty CUDA cache (phase0 pattern), and SAVE
the checkpoint (phase0 never saved — mandatory for Phase 1).

Pure helpers (`build_matryoshka_dims`, `stage_training_args`) are unit-tested in tests/test_train.py;
`train_embedder` is the thin two-stage integration, exercised by the marked smoke run (MiniLM, CPU).
"""

from __future__ import annotations

from lexfr_embed.config import BASE_MODELS, settings


def build_matryoshka_dims(model_dim: int, wanted: list[int]) -> list[int]:
    """Wanted dims <= the model's dim, always INCLUDING the full model dim, largest-first.

    Including `model_dim` ensures the full embedding is trained (sentence-transformers warns
    otherwise, since serving at native dim would then be under-trained).
    """
    return sorted({d for d in wanted if d <= model_dim} | {model_dim}, reverse=True)


def stage_training_args(stage: int, lora: bool, out_dir: str) -> dict:
    """Pure: SentenceTransformerTrainingArguments kwargs for a stage.

    Stage 2 halves the LR and uses `epochs_stage2`; both stages bf16. LR depends on the LoRA flag.
    """
    lr = settings.lr_lora if lora else settings.lr_full_ft
    if stage == 2:
        lr = 0.5 * lr
    return {
        "output_dir": out_dir,
        "num_train_epochs": settings.epochs_stage1 if stage == 1 else settings.epochs_stage2,
        "per_device_train_batch_size": settings.batch_size,
        "learning_rate": lr,
        "warmup_ratio": 0.1,
        "bf16": True,
        "report_to": settings.report_to,
        "seed": settings.seed,
    }


def build_model(base_model_id: str, use_lora: bool):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(base_model_id)
    model.max_seq_length = settings.max_seq_len
    if use_lora:
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


def make_stage1_loss(model):
    """Stage-1 loss: MNRL (plain default; CachedMNRL if `settings.use_cached_mnrl`) in MatryoshkaLoss."""
    from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dims = build_matryoshka_dims(get_dim(), settings.matryoshka_dims)
    if settings.use_cached_mnrl:
        from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

        base = CachedMultipleNegativesRankingLoss(model, mini_batch_size=settings.mini_batch_size)
    else:
        base = MultipleNegativesRankingLoss(model)
    return MatryoshkaLoss(model, base, matryoshka_dims=dims)


def _empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best-effort CPU/GPU-agnostic cleanup
        pass


def train_embedder(
    base_model_key: str | None = None,
    *,
    use_lora: bool = False,
    train_pairs: list[dict] | None = None,
    rehearsal_pairs: list[dict] | None = None,
    out_dir: str | None = None,
    max_steps: int | None = None,
):
    """Two-stage train; returns the trained model and saves it to `<out_dir>/final`.

    `train_pairs`: optional in-memory [{anchor, positive}] (smoke). If None, loads LegalKit.
    `rehearsal_pairs`: optional general (anchor, positive) pairs mixed in as an anti-forgetting
    floor (see data/rehearsal.py); None = no rehearsal (the smoke path stays offline).
    `max_steps`: optional per-stage step cap (smoke). Stage-1 checkpoint saved to `<out_dir>/stage1`
    so the forgetting canary can run on it before Stage 2 spends compute.
    """
    import gc

    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )

    from lexfr_embed.data import hard_negatives
    from lexfr_embed.data.hard_negatives import pairs_to_anchor_positive_dict

    on_cpu = not torch.cuda.is_available()

    def _device_adapt(a: dict) -> dict:
        if on_cpu:  # bf16 is GPU-only; the smoke / any CPU run must fall back to fp32
            a["bf16"] = False
            a["use_cpu"] = True
        return a

    key = base_model_key or settings.base_model_key
    base_id = BASE_MODELS[key]
    out_dir = out_dir or str(settings.results_dir / key)

    if train_pairs is None:
        from lexfr_embed.data.legalkit import load_legalkit

        train_pairs = load_legalkit(settings.phase0_subset)

    if rehearsal_pairs:  # anti-forgetting floor — both stages then see general pairs too
        from lexfr_embed.data.rehearsal import mix_rehearsal

        n_legal = len(train_pairs)
        train_pairs = mix_rehearsal(train_pairs, rehearsal_pairs, seed=settings.seed)
        print(f"[rehearsal] mixed {len(train_pairs) - n_legal} general into {n_legal} legal -> {len(train_pairs)}")

    model = build_model(base_id, use_lora=use_lora)

    # --- Stage 1: in-batch negatives ---
    ds1 = Dataset.from_dict(pairs_to_anchor_positive_dict(train_pairs))
    loss1 = make_stage1_loss(model)
    args1 = _device_adapt(stage_training_args(1, use_lora, f"{out_dir}/stage1"))
    if max_steps:
        args1["max_steps"] = max_steps
    SentenceTransformerTrainer(
        model=model, args=SentenceTransformerTrainingArguments(**args1), train_dataset=ds1, loss=loss1
    ).train()
    model.save_pretrained(f"{out_dir}/stage1")  # canary target before Stage 2

    del loss1
    gc.collect()
    _empty_cache()

    # --- Stage 2: mined hard negatives, half LR (num_negatives=0 -> plain pairs, isolates mining's value) ---
    if settings.num_negatives > 0:
        stage2_ds = hard_negatives.mine(
            train_pairs, model, num_negatives=settings.num_negatives, relative_margin=settings.hard_neg_relative_margin
        )
    else:
        stage2_ds = Dataset.from_dict(pairs_to_anchor_positive_dict(train_pairs))
    loss2 = make_stage1_loss(model)  # MNRL+Matryoshka also consumes (anchor, positive, negative)
    args2 = _device_adapt(stage_training_args(2, use_lora, f"{out_dir}/final"))
    if max_steps:
        args2["max_steps"] = max_steps
    SentenceTransformerTrainer(
        model=model, args=SentenceTransformerTrainingArguments(**args2), train_dataset=stage2_ds, loss=loss2
    ).train()
    model.save_pretrained(f"{out_dir}/final")
    return model


if __name__ == "__main__":
    train_embedder()
