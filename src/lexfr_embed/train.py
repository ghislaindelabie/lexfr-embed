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
    Stage 3 is the A1-bis distill stage: its own `distill_epochs`/`distill_lr` (LR defaults to the LoRA
    LR), but bf16/report_to/seed/warmup_ratio kept identical to 1/2 so it stays unit-tested the same way.
    """
    if stage == 3:  # A1-bis distill stage
        lr = settings.distill_lr
        epochs = settings.distill_epochs
    else:
        lr = settings.lr_lora if lora else settings.lr_full_ft
        if stage == 2:
            lr = 0.5 * lr
        epochs = settings.epochs_stage1 if stage == 1 else settings.epochs_stage2
    return {
        "output_dir": out_dir,
        "num_train_epochs": epochs,
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


def make_distill_loss(model):
    """A1-bis distill loss (MarginMSE ± Matryoshka) — delegates to `distill.make_distill_loss`.

    Kept as a thin re-export so `train.py` never imports the reranker/teacher: the distill stage
    touches only bge-m3 + LoRA + the cached labels, so the 16 GB training budget is unchanged.
    """
    from lexfr_embed.distill import make_distill_loss as _make_distill_loss

    return _make_distill_loss(model)


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
        denoiser = None
        if settings.denoise_negatives:  # A1: cross-encoder false-negative filter (RocketQA-style)
            from sentence_transformers import CrossEncoder

            denoiser = CrossEncoder(settings.denoise_reranker_id, max_length=settings.max_seq_len)
        stage2_ds = hard_negatives.mine(
            train_pairs,
            model,
            num_negatives=settings.num_negatives,
            relative_margin=settings.hard_neg_relative_margin,
            cross_encoder=denoiser,
            max_score=settings.denoise_max_score,
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


def _assert_distill_cache_meta(cache_dir: str) -> dict:
    """Load `<cache_dir>/meta.json` and fail loudly on any settings mismatch (cache/training alignment).

    A silent mismatch (wrong K, wrong reranker, a cache built over a different LegalKit subset/seed)
    would distill against the wrong positives/negatives and quietly understate the gain — so this is a
    hard gate, not a warning (PROJECT_LOG risks "cache/training misalignment", "wrong/stale miner ckpt").
    """
    import json
    from pathlib import Path

    meta_path = Path(cache_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} missing — build the teacher cache first: "
            "uv run --no-sync python scripts/build_distill_cache.py"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    checks = {
        "num_negatives": settings.distill_num_negatives,
        "reranker_id": settings.distill_reranker_id,
        "subset_size": settings.phase0_subset,
        "subset_seed": settings.seed,
    }
    mismatches = [f"{k}: cache={meta.get(k)!r} != settings={v!r}" for k, v in checks.items() if meta.get(k) != v]
    if mismatches:
        raise ValueError("distill cache / settings mismatch (rebuild the cache): " + "; ".join(mismatches))
    return meta


def distill_embedder(
    *,
    base_ckpt: str,
    cache_dir: str,
    out_dir: str,
    use_lora: bool,
    max_steps: int | None = None,
):
    """A1-bis: distill the teacher reranker's ranking into the embedder via cached MarginMSE labels.

    (1) load the embedder from `base_ckpt` (the saved stage-1/2 checkpoint) + (re)attach LoRA via
    `build_model`; (2) `load_from_disk(cache_dir)` and assert its `meta.json` matches `settings`
    (fail loudly); (3) build the MarginMSE(+Matryoshka) loss; (4) run ONE stage-3 trainer pass and
    save to `<out_dir>/distill`. The teacher reranker is NEVER imported here — training reads only the
    cache, so the 16 GB budget is bge-m3 + LoRA + labels (no reranker co-residency).
    """
    import datasets
    import torch
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )

    on_cpu = not torch.cuda.is_available()

    def _device_adapt(a: dict) -> dict:
        if on_cpu:  # bf16 is GPU-only; a CPU dry-run must fall back to fp32
            a["bf16"] = False
            a["use_cpu"] = True
        return a

    meta = _assert_distill_cache_meta(cache_dir)
    print(
        f"[distill] cache OK: {meta.get('n_rows')} rows, K={meta.get('num_negatives')}, "
        f"teacher={meta.get('reranker_id')}"
    )

    model = build_model(base_ckpt, use_lora=use_lora)
    ds = datasets.load_from_disk(cache_dir)
    loss = make_distill_loss(model)
    args = _device_adapt(stage_training_args(3, use_lora, f"{out_dir}/distill"))
    if max_steps:
        args["max_steps"] = max_steps
    SentenceTransformerTrainer(
        model=model, args=SentenceTransformerTrainingArguments(**args), train_dataset=ds, loss=loss
    ).train()
    model.save_pretrained(f"{out_dir}/distill")
    return model


if __name__ == "__main__":
    train_embedder()
