# Optimization backlog — the summer experiment program

*A living register of hypotheses for tuning lexfr-embed across **all axes**, to run as cheap local-GPU experiments once a Phase-1 model "works" decently. Each lever is a testable hypothesis with prior evidence, expected effect, cost, and how to measure it. This seeds the "summer optimization plan" section of the eventual report. Pairs with [`training-data-strategy.md`](training-data-strategy.md) and [`data-and-evaluation.md`](data-and-evaluation.md).*

> **How to use:** pick the highest-priority `todo` lever, run it as a controlled experiment (change one thing, hold the rest fixed), score it on the scorecard below, log the result + status. Optimise for the best **trade-off across axes**, not the top single number.

## The multi-axis scorecard (what every experiment is judged on)

| Axis | Metric | Source |
|---|---|---|
| **Professional query→article** (headline) | NDCG@10, Recall@100 | Track P (to build) |
| **Graph / related-provisions** | NDCG@10 on citation pairs | Track G (to build) |
| **No general-language regression** (top priority) | retention Δ ≥ −0.02 (FR+EN) | `eval_general.py` |
| **Lay robustness** (secondary) | NDCG@10 | BSARD + service-public |
| **Efficiency** | latency, index size, dim | deployment |

Report as a **Pareto view**. A lever "wins" only if it improves an axis **without** tripping the retention guard. Track experiments in **W&B** (`report_to` already wired).

## Backlog — prioritised

### Tier 1 — high ROI / strong evidence (do first)

| # | Lever | Rationale | Prior evidence | Expected effect | Cost | How to test | Status |
|---|---|---|---|---|---|---|---|
| L1 | **Negative denoising** (cross-encoder / LLM judge before using a mined negative) | Mined "hard negatives" are often real answers (multi-label law) → false negatives teach the wrong boundary | RocketQA: ~70% of mined negs were false; denoising **+10.35 MRR@10** | Large on legal headline | Med (judge pass over mined negs) | with/without denoising at fixed neg count; measure Track P/G | todo |
| L2 | **Source-stratified batching** (fill each batch from one source) | Our training set mixes LegalKit + synthetic + structure; naive concat causes interference + cross-source false-negatives | Arctic-Embed **+3.23 NDCG@10** vs un-stratified (beat a 4× larger un-stratified batch) | Medium | Low (sampler change) | stratified vs concatenated, equal batch | todo |
| L3 | **Rehearsal-ratio tuning** (mix % general FR/EN pairs) | Directly serves the top priority (no general regression); too little → forgetting, too much → diluted legal gain | Ibrahim TMLR'24: 5% (weak shift)/25% (strong) recovers retention; 1% ≈ "almost perfect stability" | Protects retention axis | Low | {0, 10, 25}% × measure retention Δ + legal | todo |
| L4 | **Data-size ablation** (find *our* knee) | Answers "how big should the training set be"; β≈0.19 shallow power law ⇒ likely saturates early | Fang et al. SIGIR'24 (scaling laws for dense retrieval) | Sets the data budget | Med (several runs, cheap locally) | 5k→10k→25k→50k→100k→200k; plot legal NDCG + retention Δ + smooth contrastive eval-loss; **knee = smallest size within ~1pt of best legal while retention Δ≥0** | todo |
| **L5** | **★ Code-hierarchy structure pairs** (heading→body, book/title/chapter containment) as auxiliary contrastive signal | **The proven structural lever on our exact setting.** Extracted (no LLM → confound-free). Teaches legal structure the agent/professional use-case needs | **G-DSR / "Finding the Law" (Louis et al., EACL 2023, on BSARD, French/Belgian):** mAP **35→47**, mRP **27→40**, Recall@100 **+1.6pp** — large rank-aware (re-ranker-like) gains | Medium; rank-aware especially | Med (parse code hierarchy → pairs) | add hierarchy pairs (down-weighted) vs baseline; measure all legal axes | **todo (flagship)** |

### Tier 2 — medium (after Tier 1 / once a model works)

| # | Lever | Rationale | Prior evidence | Cost | Status |
|---|---|---|---|---|---|
| L6 | **Checkpoint soup / SLERP merge** (average best 2–4 checkpoints) | Free robustness + OOD/retention gain at no inference cost | WiSE-FT **+1.6 ID / +4–8.7 OOD**; Qwen3-Embedding ships SLERP **+1.77 MMTEB** | Low | todo |
| L7 | **Base × method sweep** (BGE-M3 full-FT vs LoRA; Qwen3-0.6B FT; Qwen3-4B LoRA; e5-mistral-7B QLoRA) | Find the best base/method/VRAM/forgetting trade-off; large bases need rent | LoRA forgets 6–18pp less (Biderman TMLR'24); 16GB fits BGE-M3 full-FT, forces LoRA at 4B+ | High (rent for ≥4B) | todo |
| L8 | **LR × epoch grid** | Noumon skipped it and regretted it; LoRA LR ~10× full-FT | Biderman; Noumon | Med | todo |
| L9 | **Hard-negative count** (1 denoised vs 2–8) | Count saturates then degrades; denoised >> raw | DPR (1), RocketQA (4), saturates ~15 | Low | todo |
| L10 | **Matryoshka dims + quantization** (efficiency axis) | Cheaper sovereign deployment; verify quality at the deployed dim | MRL; our recipe | Low | todo |
| L11 | **Reranker / hybrid (BM25 + dense)** as honest counter-evidence | A reranker is sometimes the cheaper win than fine-tuning (research §05) | research §05 | Med | todo |

### Tier 3 — later / speculative (Phase 2+)

| # | Lever | Rationale | Caveats | Status |
|---|---|---|---|---|
| L12 | **Renvoi cross-reference pairs** (article↔cited-article) | doc↔doc "related-provisions" signal, on-axis for agents; **unprecedented** as a training signal | Lower priority than L5 (hierarchy = proven, renvoi = speculative); **down-weight**, **filter procedural renvois** (false positives), **graph-split** leakage control, **ablate ruthlessly**; natural payoff is a related-provisions feature, not query→article | todo (after L5) |
| L13 | **Synthetic professional-register generation + multi-LLM jury** | Coverage + register diversity; jury = quality filter | Jury not literature-backed for IR labeling → validate vs human spot-checks first; use a different-family generator; consistency-filter with a non-base model | todo |
| L14 | **GISTEmbed vs MNRL loss ablation** | GISTEmbed (guided in-batch negatives) may beat CachedMNRL | research §03 | todo |
| L15 | **EU-French (EUR-Lex) augmentation** | LDS breadth; register/subject diversity | Phase 2 unconditionally; EUR-Lex under Commission Decision 2011/833/EU; with/without-EU ablation; may not lift the FR-national headline | todo |

## Notes on running the program

- **Local GPU economics make this viable:** a run is ~€0 of electricity, not $1–2 of RunPod → run a disciplined sweep all summer; rent only for ≥4B bases.
- **Discipline > volume:** change one variable per experiment; freeze+hash evals first; always run the retention guard; log to W&B.
- **This *is* the publication.** A systematic, multi-axis, confound-controlled ablation program is the methodological contribution for the NLLP-2026 / methods angle — not a single benchmark number.
- **Provenance:** levers L1–L11 are grounded in the 2026-06-25 capacity/scaling research; L5/L12 in the 2026-06-25 citation-structure research; full citations in `training-data-strategy.md` and the research agents' reports.
