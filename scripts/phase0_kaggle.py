"""Phase-0 walking skeleton — SELF-CONTAINED, runs end-to-end on free Kaggle GPU.

Goal (research §07): smallest model + LegalKit subset -> train -> eval on BSARD ->
a baseline->fine-tuned NDCG@10 delta, for the mentor gate. Prove the pipeline, not SOTA.

This file has NO dependency on the (private) lexfr_embed package — it inlines the
confirmed LegalKit/BSARD field mappings so you can paste it into a Kaggle notebook cell
and run. Cost: $0.

PROVEN Kaggle results (1 epoch, 9.8k LegalKit pairs, BSARD test, max_seq 512):
  MiniLM (full-FT):  NDCG@10 0.055 -> 0.148  (recall@100 0.16 -> 0.42)
  BGE-M3 (LoRA):     NDCG@10 0.240 -> 0.292  (recall@100 0.59 -> 0.64)   <- representative
The strong base (BGE-M3) starts far higher and gains less in relative terms — expected.

Setup that makes it run on Kaggle:
- **Accelerator: GPU T4, NOT P100.** Kaggle's torch 2.10/cu128 dropped Pascal/sm_60, so a
  P100 fails with "CUDA error: no kernel image". Via the API:  kaggle kernels push --accelerator NvidiaTeslaT4
- **First notebook cell (BEFORE importing torch):** pin Kaggle's GPU-matched torch so pip
  can't replace it, use a single T4 (avoids DataParallel OOM), and reduce fragmentation:
      import os
      os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
      os.environ['CUDA_VISIBLE_DEVICES']='0'
      import torch; open('/tmp/c.txt','w').write(f'torch=={torch.__version__}\\n')
      !pip install -U -q sentence-transformers datasets peft -c /tmp/c.txt
      !pip uninstall -y -q torchao   # peft 0.19 raises on Kaggle's torchao 0.10<0.16 (unused for LoRA)
  then run this file (its __main__ guard calls main()).

Notes:
- T4 is Turing -> **fp16** (bf16 needs Ampere+). Training frees memory before the final
  eval encode (del trainer + empty_cache) to avoid a 16 GB OOM.
- Default base is now **BAAI/bge-m3** (568M) with **LoRA** (USE_LORA=True) so it fits a 16GB
  T4 — full fine-tune of 568M would OOM. For a fast pipeline smoke, set BASE_MODEL to the
  MiniLM fallback + USE_LORA=False (118M, full-FT, no prefix quirks).
"""

from __future__ import annotations

import random
import re
import unicodedata

# ----------------------------- CONFIG (tweak here) ----------------------------- #
BASE_MODEL = "BAAI/bge-m3"  # 568M, MIT, 1024-dim, 8192 ctx (MiniLM smoke fallback noted in docstring)
LEGALKIT_SUBSET = 12_000  # small Phase-0 subset (stratified by code)
MATRYOSHKA_DIMS = [1024, 512, 256, 128, 64]  # filtered to <= model dim at runtime
BSARD_SPLIT = "test"  # 222 held-out questions
MAX_SEQ_LEN = 512  # cap (BSARD articles can be huge) -> bounds eval time
USE_LORA = True  # LoRA for >=568M bases — fits a 16GB T4 (full-FT would OOM)
LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 16  # train batch (LoRA BGE-M3 on a 16GB T4)
EVAL_BATCH = 16  # eval-encode batch (BGE-M3 568M @ 1024-dim)
EPOCHS = 1
LR = 1e-4  # LoRA LR (higher than full-FT's 2e-5)
SEED = 42
PUSH_TO = None  # e.g. "ghislaindelabie/lexfr-embed-phase0" (needs HF_TOKEN)
# ------------------------------------------------------------------------------- #


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^\w ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_legalkit_pairs(n: int, seed: int = SEED) -> list[dict]:
    """LegalKit (CC-BY) -> deduped, code-stratified (anchor=query, positive=article).

    Confirmed schema (HF viewer 2026-06-20): split "train"; `query` = question,
    `output` = article text, `input` = "Code civil, art. 265-2" (code is the prefix).
    """
    from collections import defaultdict

    from datasets import load_dataset

    ds = load_dataset("louisbrulenaudet/legalkit", split="train")
    pairs, seen = [], set()
    for r in ds:
        q, a = (r.get("query") or "").strip(), (r.get("output") or "").strip()
        if not q or not a:
            continue
        key = (_norm(q), _norm(a))
        if key in seen:  # dedup (legal boilerplate -> false negatives)
            continue
        seen.add(key)
        code = (r.get("input") or "").split(", art.")[0].strip() or "unknown"
        pairs.append({"anchor": q, "positive": a, "code": code})

    # stratify by code (diversity > volume; LegalKit skews to Code du travail)
    rng = random.Random(seed)
    by_code: dict[str, list] = defaultdict(list)
    for p in pairs:
        by_code[p["code"]].append(p)
    per_code = max(1, n // max(1, len(by_code)))
    out: list[dict] = []
    for items in by_code.values():
        rng.shuffle(items)
        out.extend(items[:per_code])
    rng.shuffle(out)
    return out[:n]


def load_bsard(split: str = BSARD_SPLIT):
    """BSARD -> (queries, corpus, relevant_docs) for InformationRetrievalEvaluator.

    Confirmed schema: config "corpus" (id, article) 22,633 rows; config "questions"
    (id, question, article_ids = COMMA-separated corpus ids). CC-BY-NC-SA, eval only.
    """
    from datasets import load_dataset

    corpus_ds = load_dataset("maastrichtlawtech/bsard", "corpus", split="corpus")
    q_ds = load_dataset("maastrichtlawtech/bsard", "questions", split=split)
    corpus = {str(r["id"]): r["article"] for r in corpus_ds}
    queries = {str(r["id"]): r["question"] for r in q_ds}
    relevant = {str(r["id"]): {c.strip() for c in r["article_ids"].split(",") if c.strip()} for r in q_ds}
    return queries, corpus, relevant


def ir_eval(model, queries, corpus, relevant, name: str) -> dict:
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        ndcg_at_k=[10],
        accuracy_at_k=[1, 10],
        precision_recall_at_k=[10, 100],
        mrr_at_k=[10],
        map_at_k=[100],
        name=name,
        show_progress_bar=True,
        batch_size=EVAL_BATCH,
    )
    return evaluator(model)


def _ndcg10(metrics: dict) -> float:
    return next((v for k, v in metrics.items() if k.endswith("ndcg@10")), float("nan"))


def main() -> None:
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

    random.seed(SEED)

    # 1) DATA — small, stratified, deduped LegalKit subset
    pairs = load_legalkit_pairs(LEGALKIT_SUBSET)
    train_ds = Dataset.from_list([{"anchor": p["anchor"], "positive": p["positive"]} for p in pairs])
    print(f"train pairs: {len(train_ds)} across {len({p['code'] for p in pairs})} codes")

    # 2) EVAL SET + ZERO-SHOT BASELINE (this delta is half the mentor gate)
    queries, corpus, relevant = load_bsard()
    print(f"BSARD: {len(queries)} queries over {len(corpus)} articles")
    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = MAX_SEQ_LEN
    if USE_LORA:
        from peft import LoraConfig, TaskType

        # LoRA inits with B=0 → adds nothing at init, so zero-shot below == the base model.
        model.add_adapter(
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=0.1,
            )
        )
    before = ir_eval(model, queries, corpus, relevant, "zeroshot")
    print("ZERO-SHOT:", before)

    # 3) TRAIN — Stage 1: MNRL wrapped in Matryoshka (dims filtered to <= model dim)
    dim = model.get_sentence_embedding_dimension()
    dims = [d for d in MATRYOSHKA_DIMS if d <= dim] or [dim]
    loss = MatryoshkaLoss(model, MultipleNegativesRankingLoss(model), matryoshka_dims=dims)
    args = SentenceTransformerTrainingArguments(
        output_dir="phase0",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_ratio=0.1,
        fp16=True,  # T4 = fp16 (bf16 needs Ampere+)
        logging_steps=20,
        save_strategy="no",
        seed=SEED,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=loss)

    # --- timed training (matched 5060 Ti vs T4 benchmark; same script both sides) ---
    import time as _time

    import torch as _t
    gpu = _t.cuda.get_device_name(0) if _t.cuda.is_available() else "cpu"
    if _t.cuda.is_available():
        _t.cuda.reset_peak_memory_stats()
    _t0 = _time.perf_counter()
    trainer.train()
    if _t.cuda.is_available():
        _t.cuda.synchronize()
    train_wall_s = _time.perf_counter() - _t0
    vram_gb = _t.cuda.max_memory_allocated() / 1e9 if _t.cuda.is_available() else 0.0
    n_samples = len(train_ds) * EPOCHS
    print(
        f"BENCH gpu={gpu!r} pairs={len(train_ds)} epochs={EPOCHS} bs={BATCH_SIZE} seq={MAX_SEQ_LEN} "
        f"train_wall_s={train_wall_s:.1f} samples_per_s={n_samples / train_wall_s:.2f} "
        f"peak_vram_gb={vram_gb:.2f} torch={_t.__version__}"
    )

    # free training memory (optimizer/grads) before the final eval encode — avoids T4 OOM
    import gc

    import torch

    del trainer, loss
    gc.collect()
    torch.cuda.empty_cache()

    # 4) EVAL AFTER + the delta to show the mentor
    after = ir_eval(model, queries, corpus, relevant, "finetuned")
    print("FINE-TUNED:", after)
    b, a = _ndcg10(before), _ndcg10(after)
    print(f"\n=== BSARD NDCG@10: {b:.4f} -> {a:.4f}  (delta {a - b:+.4f}) ===")

    # 5) OPTIONAL — push to HF Hub so the result survives the Kaggle session (needs HF_TOKEN)
    if PUSH_TO:
        model.push_to_hub(PUSH_TO)


if __name__ == "__main__":
    main()
