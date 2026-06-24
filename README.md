# LexFR-Embed — a French legal-domain text embedder

> A domain-adapted French legal embedding model for retrieval/RAG (repo `lexfr-embed`, package `lexfr_embed`), fine-tuned with sentence-transformers + LoRA on openly-licensed French legal data and evaluated on BSARD. OpenClassrooms capstone + open release; seeds the LegalDataSpace (LDS) retrieval stack.

## Why

No open embedder exists for **French national law** (the Maastricht models cover *Belgian* law; JuriBERT is a masked-LM, not a retriever). General embedders drop sharply on French legal statute retrieval. A small domain fine-tune is cheap (cents–a few €) and can beat general/proprietary embedders by a meaningful margin. Full rationale, benchmarks, budget, and plan are in the **research bundle** (read these first):

- 📖 Research report (mobile): https://p710.tail3089b5.ts.net:8445/doc/law-embedder-report
- 📄 Mentor proposal (FR): https://p710.tail3089b5.ts.net:8445/doc/law-embedder-proposition
- Markdown sources: `../law-embedder/docs/` *(planning docs; to be consolidated into `docs/` here)*
- **In-repo specs:** `docs/data-and-evaluation.md` (what we train vs eval on — start here), `docs/eval-set-spec.md` (French eval set), `docs/publication-venues.md`

## Approach (one screen)

```
LegalKit (CC-BY) + DILA/jurisprudence (Etalab)
  → chunk + (query, article) pairs; augment with synthetic + practitioner-register queries
  → split by article id (no leakage); dedup
  → Stage 1: CachedMultipleNegativesRankingLoss wrapped in MatryoshkaLoss (bf16)
  → Stage 2: 1 filtered hard negative (mined)   ·   LoRA r16/α32 if base > 1B, else full FT
  → eval: InformationRetrievalEvaluator on BSARD + a French-national held-out set
  → ablate: MNRL vs GISTEmbedLoss ; dense vs hybrid(+BM25) vs +reranker ; Matryoshka dims
  → downsize: fp16 / int8-ONNX + Matryoshka dim truncation
  → serve: HF TEI / sentence-transformers behind FastAPI /search
```

Base-model candidates: **BGE-M3** (568M, MIT, 8192 ctx) and **Qwen3-Embedding-0.6B** (Apache-2.0, MRL); stretch: Qwen3-4B / e5-mistral-7B. See research report §03/§09.

## Quickstart

```bash
uv sync --extra dev            # install (add --extra track for W&B, --extra gpu for QLoRA)
make test                      # hermetic unit tests (no network)
make smoke                     # end-to-end walking skeleton on a tiny model (needs network)
```

## Layout

| Path | What |
|---|---|
| `src/lexfr_embed/config.py` | central config (pydantic-settings): model ids, paths, hyperparameters |
| `src/lexfr_embed/data/` | LegalKit loader, synthetic-query gen, hard-negative mining |
| `src/lexfr_embed/train.py` | SentenceTransformerTrainer: CachedMNRL + Matryoshka, Stage 1/2, LoRA |
| `src/lexfr_embed/evaluate.py` | `InformationRetrievalEvaluator` on BSARD + held-out |
| `src/lexfr_embed/quantize.py` | fp16 / int8-ONNX + Matryoshka dim variants |
| `src/lexfr_embed/serve.py` | FastAPI `/search` over a FAISS index (demo) |
| `scripts/phase0_kaggle.py` | the Phase-0 free-Kaggle walking skeleton (paste into a notebook cell) |
| `tests/` | hermetic unit tests + a marked smoke test |

## Plan (real calendar)

- **Phase 0 (→ Fri 26 Jun):** free Kaggle walking skeleton → mentor gate. **$0.** ✅ done.
- **Phase 1 (card ≈ 27 Jun → submit 7 Jul):** full-steam local serial + RunPod burst.
- **Phase 2 (post-7 Jul):** extended runs → arXiv preprint + HF model/dataset release.

Compute ladder: Kaggle-free → RunPod (crunch/parallel) → local GPU (serial). *Hardware is off the critical path — the whole project can finish on RunPod (~$150) if the card slips.*

## Licensing

Trained on **open data only** (LegalKit CC-BY, DILA/jurisprudence Etalab). **BSARD/LLeQA are CC-BY-NC-SA → evaluation only.** Code: Apache-2.0. The LDS production model is whichever variant benchmarks best, deployed as-is — no training on private/client corpora.
