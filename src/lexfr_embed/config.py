"""Central configuration (pydantic-settings).

Reads from environment / `.env` (see `.env.example`). One place to change model
ids, paths, and the hyperparameters from the research report (§03). Keep training
scripts importing from here so Phase-0 (small) and Phase-1 (full) share one config.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

# Candidate base models (research §03). Phase 0 uses the small/fast one; Phase 1 sweeps.
BASE_MODELS = {
    "bge-m3": "BAAI/bge-m3",  # 568M, MIT, 8192 ctx, dense+sparse+ColBERT
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",  # 0.6B, Apache-2.0, MRL, 32k ctx
    "qwen3-4b": "Qwen/Qwen3-Embedding-4B",  # stretch (LoRA)
    "e5-mistral-7b": "intfloat/e5-mistral-7b-instruct",  # stretch (QLoRA), MIT
    # Phase-0 tiny smoke model (fast, multilingual incl. FR):
    "smoke": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

# Existing French-language legal retrievers (Belgian law) — baselines to beat (research §01).
BASELINE_MODELS = {
    "camembert-base-lleqa": "maastrichtlawtech/camembert-base-lleqa",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="LEXFR_EMBED_", extra="ignore")

    # --- secrets (also read without prefix from .env via aliases below) ---
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    wandb_api_key: str | None = Field(default=None, alias="WANDB_API_KEY")
    mistral_api_key: str | None = Field(default=None, alias="MISTRAL_API_KEY")

    # --- data ---
    legalkit_id: str = "louisbrulenaudet/legalkit"
    bsard_id: str = "maastrichtlawtech/bsard"  # eval only (CC-BY-NC-SA)
    data_dir: Path = REPO_ROOT / "data"
    results_dir: Path = REPO_ROOT / "results"

    # --- model choice ---
    base_model_key: str = "bge-m3"  # Phase-1 MVP base (fits 16GB full-FT/LoRA); "smoke"=MiniLM for the CPU wiring test
    matryoshka_dims: list[int] = [1024, 512, 256, 128, 64]

    # --- training hyperparameters (research §03 starting points) ---
    epochs_stage1: int = 2
    epochs_stage2: int = 1
    batch_size: int = 128  # effective batch (CachedMNRL decouples it from VRAM)
    mini_batch_size: int = 16  # CachedMNRL per-step micro-batch — bounds VRAM
    max_seq_len: int = 512  # training/eval sequence cap (Phase-0 tested 512 vs 1024; 512 is faster, 1024 a minor lever)
    rehearsal_frac: float = 0.07  # general FR/EN pairs mixed in from run 1 (anti-forgetting floor)
    lr_full_ft: float = 2e-5
    lr_lora: float = 1e-4
    lora_r: int = 16
    lora_alpha: int = 32
    use_lora_above_params: float = 1e9  # LoRA for bases > ~1B params
    hard_neg_relative_margin: float = 0.05
    num_negatives: int = 1  # Stage-2 mined hard negatives per pair; 0 = plain pairs (isolates mining's value)
    denoise_negatives: bool = False  # A1: rescore mined negatives with a cross-encoder, drop false negatives (RocketQA)
    denoise_reranker_id: str = "BAAI/bge-reranker-v2-m3"  # A1 cross-encoder (different family than the embedder)
    denoise_max_score: float | None = None  # A1: optional hard cap on reranker score above which a candidate is dropped
    use_cached_mnrl: bool = False  # Stage-1: plain MNRL default; CachedMNRL opt-in (gated on the CPU smoke)
    seed: int = 42

    # --- dataset sizing (research §03/§04) ---
    target_pairs: int = 90_000  # ~80–100k cap; Phase 0 uses a small subset
    phase0_subset: int = 15_000

    @property
    def base_model_id(self) -> str:
        return BASE_MODELS[self.base_model_key]

    @property
    def report_to(self) -> str:
        """Graceful: W&B only if a key is set AND wandb is importable (else off).

        Checking importability (not just the key) keeps CPU/smoke runs working without the
        `track` extra; the real run does `uv sync --extra track` to enable logging.
        """
        if not self.wandb_api_key:
            return "none"
        try:
            import wandb  # noqa: F401
        except ImportError:
            return "none"
        return "wandb"


settings = Settings()
