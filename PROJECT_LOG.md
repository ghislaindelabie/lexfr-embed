# lexfr-embed — project log

*A detailed trace of decisions, implementation steps, and intermediary results. Combines the working discussions, the validated results, and what is recorded in git. Chronological with consolidated tables at the end.*

**Last updated:** 2026-07-03 · **Status:** **Phase 1 pipeline shipped + first real GPU run done.** PRs #1–#7 merged; full two-stage training pipeline built (TDD, 43 tests, CI green); first end-to-end RunPod run gives a defensible headline (**BSARD NDCG@10 0.240 → 0.290, 95% CI [+0.027, +0.073]**); retention guard running. Repo **public** (JOSS clock started, eligible ~2026-12-23). *(Sections 5–7 below predate the merges; see the 2026-07-03 entry in §3 for the current state.)*

---

## 1. What this project is

**lexfr-embed** (product name *LexFR-Embed*, package `lexfr_embed`) is a **domain-adapted French legal text embedder** — a sentence-embedding model fine-tuned for retrieval over French legal text (statutes / codes / case law).

It serves **two purposes at once**:

1. **OpenClassrooms capstone** — the AI-Engineer portfolio project *"Réalisez votre portfolio d'AI Engineer"* (path 2053, RNCP39775 Niveau 7, blocks BC03/BC05). Codenamed **OC15**, and **separate from OC14** ("Finetunez votre propre LLM", a generative medical-triage project).
2. **LDS / LegalDataSpace support** — improves retrieval quality for LDS, a French legal-data marketplace built on Alien Intelligence's DataCluster (a sovereign, pay-per-query, never-trained retrieval layer over licensed content). Addresses LDS's named **R6 "search-quality gap"** / anticipated "vectorial tuning phase".

**Key scoping constraint:** the OC/portfolio version uses **public data only** (the LDS lawyer corpus is contractually restricted with a no-training guarantee). The LDS production path is *not* a separate licensed-training track — LDS simply **deploys whichever publicly-trained model benchmarks best**; sovereignty (Scaleway-FR hosting) is a deployment concern only.

**Repository:** `github.com/ghislaindelabie/lexfr-embed` (private, SSH, identity `ghislain@delabie.tech`).

---

## 2. Timeline at a glance

| Phase | Window | Goal | Compute |
|---|---|---|---|
| **Research & planning** | 2026-06-14 → 06-20 | Justify the build; recipe; budget; hardware; scaffold | — |
| **Phase 0 — walking skeleton** | → **mentor gate Fri 2026-06-26** | Prove the full pipeline; baseline→fine-tuned BSARD delta | **Free (Kaggle T4)** |
| **Phase 1 — full recipe** | card arrives ≈ Sat 06-27 → **submit Tue 2026-07-07** | Hard negatives + synthetic queries + multi-model compare + French eval set | Local RTX 5060 Ti + RunPod burst |
| **Phase 2 — publication** | post-07-07, no deadline | arXiv preprint + HF model/dataset release; workshop paper | Local + RunPod |

Hardware is **off the critical path** — the whole project can complete on RunPod (~$150) if the local card slips.

---

## 3. Chronological decision & implementation log

### 2026-06-14 — Inception + deep research

- **Decision:** explore a French legal embedder ("law-embedder") to (a) improve LDS retrieval and (b) double as the OC15 capstone.
- **Action:** ran an 8-angle ultracode deep-research workflow (adversarially verified), covering existing French legal embedders, precedents, data/benchmarks, architecture, expected gains, Mistral-embedder LoRA feasibility, and OC feasibility.
- **Verdict (build is justified):**
  - **No open embedder for French *national* law exists.** The only purpose-built open French-language legal retrievers (Maastricht `camembert-base-lleqa`, `dpr/splade/colbert-legal-french`) are trained on **Belgian** law (BSARD/LLeQA), built on dated `camembert-base` (384-token), modest quality.
  - **Template precedent = Daniel Noumon's Dutch legal embedder** (EU AI Act, Apr 2026): Qwen3-Embedding-4B + LoRA → NDCG@10 0.966, beating OpenAI by >10 pts for ~€0.75 of GPU. Open, directly portable recipe.
  - **Expected gain:** +5–13 NDCG@10 pts from a mid-tier base, but only +1–4 from a SOTA base — so a reranker/hybrid must always be benchmarked as honest counter-evidence.
  - **Mistral embedder is NOT LoRA-able** (mistral-embed/codestral-embed are closed API; the fine-tuning API excludes embedders). Use open `e5-mistral-7b-instruct` (MIT) or GritLM-7B if a Mistral lineage is wanted.
- **Recommended shape:** base = BGE-M3 (MIT) / Qwen3-Embedding-0.6B (Apache-2.0); method = synthetic (query, article) pairs → CachedMNRL + MatryoshkaLoss → hard negatives → LoRA r16/α32; train on LegalKit, eval on BSARD.

**Deliverables produced:** 10 research reports in `~/code/law-embedder/docs/research/` (`00-EXECUTIVE-SUMMARY` → `08-sources`, `09-tech-stack-and-architecture`). Mobile HTML published to the vault (`/doc/law-embedder-report`). Generator: `docs/research/build_report_html.py` (`.md` is source of truth, re-run to refresh).

### 2026-06-14 → 06-18 — Dataset sizing + project proposal

- **Decision (dataset size):** diversity over raw volume. Target **~80–100k (query, article) pairs**, stratified by code; ~100k is the sensible cap. (Some raw datasets are multi-GB; full ingestion is unnecessary.) Added to research report 03.
- **Action:** drafted a clean **French project proposal** (`docs/proposal/proposition-projet-law-embedder.md`), ~4 pages, covering (1) the need, (2) goals, (3) methodology — naming LDS/Alien and **Léo (CTO) as technical advisor** on contrastive training + eval design, framed as the OC15 capstone. Published as mobile HTML (`/doc/law-embedder-proposition`). Generator: `build_proposal_html.py`.

### 2026-06-19 — Compute budget, re-examined (CTO challenge)

- **Trigger:** CTO Léo questioned the original "<$20" RunPod estimate.
- **Finding:** the <$20 was wrong (under-counted run-count, debug reruns, idle/always-on serving tail). Training is only ~half the bill.
- **Decision — defensible RunPod budget tiers:**
  - **Lean ~$30** (BGE-M3 + Qwen3-0.6B, 53k pairs).
  - **Realistic ~$150 / cap $250** ← *recommended* (partial ablation grid ~30 runs, ~80–100k pairs, quantization).
  - **Thorough ~$650–950** (stretch to 4B/7B, opt-in).
  - Hard ceiling holds **only** with auto-stop on every pod + serverless scale-to-zero serving + corpus bounded to 50–300k passages. **Biggest risk is operational** — a forgotten always-on pod ≈ $950 by itself.
- **Data-gap clarification:** the gap to close is **informal/practitioner language** (jargon). Reusable open sources are query-side only (`AgentPublic/service-public` related questions, Etalab sigles dict); practitioner docs/forums are copyright/DB-rights locked → close the gap by **synthetic jargon-rephrasing** of formal queries (different generator/prompt than LegalKit).

### 2026-06-20 — Repo scaffold + eval/venue specs

- **Decision (name):** **lexfr-embed** / `lexfr_embed` (chosen over jurisembed/droit-embed).
- **Action — scaffolded the repo** (`chore: scaffold lexfr-embed`, commit `b52e221`): uv + `src/lexfr_embed/` layout, pydantic-settings `config.py`, ruff (line-length 120), pytest (hermetic + gated smoke), GitHub Actions CI.
  - **`data/legalkit.py` — schema CONFIRMED** (HF viewer): `query` = question, `output` = article text, `input` = `"Code civil, art. 265-2"` (code is the prefix; no code column). Pure dedup/stratify helpers, unit-tested.
  - **`evaluate.py` — BSARD loader CONFIRMED**: configs `corpus` (22,633 articles) + `questions` (test = 222, train = 886); `article_ids` is a **comma-separated string**; CC-BY-NC-SA, ungated.
  - `train.py` / `quantize.py` / `serve.py` + synthetic-query / hard-negative **stubs** wired to sentence-transformers v5.
- **Action — two planning docs:**
  - **`docs/eval-set-spec.md`** — construction spec for a **French-national legal-IR eval set** (none exists; BSARD/LLeQA are Belgian). Two tracks reported side by side: **Track A** (OOD headline, ~250–350 queries from service-public.fr fiches + ~100 hand-curated practitioner queries, multi-label) and **Track B** (in-distribution diagnostic, ~200 held-out LegalKit queries, flagged synthetic). The **A↔B gap is itself a finding** (overfitting to generator phrasing). Includes leakage control by `(code, num)` id exclusion, freeze+hash, bootstrap CIs, Léo 15–20% spot-check.
  - **`docs/publication-venues.md`** — Phase-2 targets. **NLLP 2026** (EMNLP workshop, deadline ~Aug 11–27) = best fit and reachable; JURIX 2026 (Sep 5); ECIR 2027 resource track (Nov 2).
- **Licensing decision:** LegalKit (CC-BY-4.0) for training; BSARD (CC-BY-NC-SA) for **eval only**. Public data only.

### 2026-06-19 → 06-22 — GPU hardware research → purchase

Extensive local-GPU feasibility study for the P710 (Lenovo ThinkStation P710).

- **P710 constraints (verified):** PSU **850 W** (`dmidecode -t 39`, 2026-06-20); Lenovo caps add-in cards at **300 W/slot**; full-length bay ≈ 298 mm (in) / ~310–315 (out); PSU aux harness = **one 6-pin + one 6+2-pin** (a second 8-pin needs a 6→8 adapter); PCIe 3.0 (a non-issue for single-GPU training).
- **Cards surveyed (<€1500, 16–24 GB):** RTX 5060 / 5060 Ti, RX 7900 XT / XTX, Radeon Pro W7700 / W7800, R9700 32 GB, RTX 4000 Ada 20 GB, RTX 5070 Ti. Key findings: 24 GB-new is essentially only the RX 7900 XTX (NVIDIA 24 GB EOL/over-budget); **all long triple-fan 7900 XTX AIBs are too long** for the P710 (only a short reference/MBA 287 mm card fits, power-limited to ~300 W); the W7700 is *dominated by the 5060 Ti for AI* (slower + ROCm + ~2× price).
- **DECISION — purchased 2026-06-22:** **MSI GeForce RTX 5060 Ti 16 G Ventus 2X OC PLUS** (Darty, ~€629). Rationale: CUDA/Blackwell with **FP4 + FP8**, 180 W, single 8-pin (no adapter), ~242 mm → **trivial P710 fit, fully within Lenovo spec**; 16 GB covers the core models; CUDA removes all ROCm/QLoRA risk. Delivery 23–25 Jun, install by Fri 27 Jun (pull the Quadro K6000 first).
- **Impact:** a local card turns the cloud budget into ~€5 of electricity and kills the metering / forgotten-pod risk. Hardware confirmed **off the OC critical path**.

*(Full hardware state in the `server-p710` skill reference; detail in `~/vault/work/projects/legal-finetune/p710-hardware-feasibility.md`.)*

### 2026-06-22 → 06-23 — Phase 0: automated Kaggle validation

- **Goal:** prove the full pipeline end-to-end on **free** GPUs and produce a baseline→fine-tuned BSARD NDCG@10 delta for the mentor gate.
- **Action:** built `scripts/phase0_kaggle.py` — a **self-contained** runner (no dependency on the private package) that loads a stratified+deduped LegalKit subset, runs a zero-shot BSARD baseline, trains Stage-1 (MNRL wrapped in MatryoshkaLoss, LoRA when enabled), re-evaluates, and prints the delta. Driven **fully automatically via the Kaggle API** (kernels push/status/output) with background polling to completion.
- **Kaggle launch recipe (hard-won — documented in the script docstring):**
  - Push with **`--accelerator NvidiaTeslaT4`** — Kaggle's torch 2.10/cu128 dropped Pascal `sm_60`, so a **P100 fails** with "CUDA error: no kernel image".
  - First cell, before importing torch: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUDA_VISIBLE_DEVICES=0` (single T4, avoids DataParallel OOM), **pin torch** during `pip install -U` (constraint file), and **`pip uninstall -y torchao`** (peft 0.19 rejects Kaggle's torchao 0.10 < 0.16 during LoRA injection).
  - T4 = Turing → **fp16** (bf16 needs Ampere+). Free training memory (`del trainer, loss; gc.collect(); torch.cuda.empty_cache()`) before the final eval encode to avoid a 16 GB OOM.
- **Auth:** new-style Kaggle token (from a GDoc "Kaggle API") written to `~/.kaggle/access_token`; account `ghislaindelabie`.

**Results — 4 runs, all via the automated API loop:**

| Run | Base / method | max_seq | NDCG@10 (zero-shot → fine-tuned) | Recall@100 | Accuracy@10 |
|---|---|---|---|---|---|
| 1 | MiniLM-L12, full-FT | 512 | **0.055 → 0.148** (+0.092, ~2.7×) | 0.16 → 0.42 | 0.16 → 0.33 |
| 2 | BGE-M3 + LoRA | 512 | **0.240 → 0.292** (+0.052, +22%) | 0.59 → 0.64 | 0.52 → 0.59 |
| 3 | BGE-M3 + LoRA | 1024 | **0.242 → 0.307** (+0.065) | 0.592 → 0.657 | 0.514 → 0.608 |

*(All: 1 epoch, ~9.8k LegalKit pairs, Stage-1 only, BSARD test = 222 queries over 22,633 articles.)*

**Interpretation:**
- The pipeline is **proven end-to-end**. Strong base (BGE-M3) starts far higher and gains less in *relative* terms — exactly matching the research's counter-evidence prediction.
- BGE-M3 zero-shot ≈ 0.24 corroborates the "general models score ~0.2–0.3 on BSARD" reading; the contested MTEB-fr ~0.79 figure does **not** hold under our setup.
- **max_seq 512→1024 is a minor lever** (+1.5 NDCG@10 pts fine-tuned; zero-shot ~unchanged) — most BSARD statutory articles fit ~512 tokens. The committed Kaggle default stays **512** (fast smoke); carry ~1024 on the local card where compute is cheap.
- Absolute numbers are modest because the base is small (run 1) and **BSARD is Belgian law** (cross-jurisdiction transfer from French LegalKit). The **real Phase-1 levers are Stage-2 hard negatives + synthetic/practitioner queries + more epochs/pairs**, not context length.

### 2026-06-23 — General-capability retention guard (anti-forgetting)

- **Concern raised:** a LoRA legal fine-tune (contrastive MNRL on a narrow distribution) can erode the model's **general** French/English retrieval + semantic-similarity behaviour (catastrophic forgetting / representation collapse). The legal eval (BSARD + Track A/B) cannot detect this — it is legal-only.
- **Decision (D13):** add a **before-vs-after general-capability retention check** on a small, fixed, strictly **non-legal** MTEB(fr)+BEIR subset (FR/EN retrieval + FR STS + 1 clustering); accept only if general scores drop **≤ ±0.02** while the legal metric rises. Also a credibility win for the OC eval blocks (BC03/BC05).
- **Implemented (TDD):** `src/lexfr_embed/general_eval.py` (suite + pure verdict logic) with 8 hermetic unit tests in `tests/test_general_eval.py` (all pass, ruff clean) + `scripts/eval_general.py` (before/after deltas, PASS/FAIL, exit-code-gated); `mteb` added to the `eval` optional extra. Spec written into `docs/eval-set-spec.md` ("General-capability retention").

### 2026-06-25 — Primary use-case correction (professionals, not laypeople)

- **Correction:** the main users are **legal professionals + querying agents** (professional register, article citations, cross-reference/graph expectations) — **not** laypeople. Lay-citizen questions are a *later* phase. Consequence: BSARD (Belgian + lay) and the planned service-public Track A (lay) measure the *wrong register* for the headline → demoted to a **secondary "lay-robustness" axis**; the headline eval becomes **professional-register + graph/relatedness-aware** (to build, via citation-as-relevance). This upgrades the renvoi/citation-structure idea from auxiliary to *on-axis*.

### 2026-06-25 → 07-02 — Phase-1 pipeline built (TDD) and hardened

- Implemented the full **two-stage training pipeline** behind the earlier stubs, all test-first:
  - `train.py` — Stage-1 MNRL (⊂ Matryoshka) → mine **1 filtered hard negative** → Stage-2 at half LR; **saves both checkpoints**; plain-MNRL default with **CachedMNRL opt-in** (`use_cached_mnrl`, mini-batch 16) to fit BGE-M3 on 24 GB; CPU/bf16-aware for smoke.
  - `metrics.py` — `bootstrap_ci`, `paired_delta_ci`, `min_detectable_effect`, pure `ndcg_at_k`.
  - `data/leakage.py` — canonical `(code,num)` ids, `hard_exclude`, order-independent SHA-256 `hash_partition`, NC/SA source whitelist.
  - `scorecard.py` — honest renderer (CI + "excludes zero?", sub-MDE "within noise", retention regression, partition hashes).
  - `evaluate.per_query_ndcg_at_k`, `scripts/run_phase1.py` (the graded driver).
- **43 hermetic tests + a MiniLM/CPU two-stage smoke (both loss modes) + CI green.** The smoke caught & fixed 3 real bugs (bf16-on-CPU crash; Matryoshka silently dropped the full model dim; `report_to` needed wandb importable) and dropped SyntecRetrieval from the retention guard (MTEB tags it *Legal*, invalid for a non-legal guard).
- **Honesty fix (critical):** the "0.240 → 0.307" headline was a **splice** (512 zero-shot + 1024 fine-tuned). True *within-config* deltas: **+0.052 @512** (0.240→0.292) and **+0.065 @1024** (0.242→0.307). All mentor-facing material corrected; trust checklist adopted (no splice, CI + excludes-zero, retention as "no regression > ±MDE", frozen+hashed partition, transfer-proxy caveat, measured-vs-inferred labels).
- **PRs #1–#7 merged to `main`** (user explicitly authorised merging #6+#7 after CI turned green — the normal rule is user-merges-only).

### 2026-07-03 — First real GPU run (RunPod RTX 4090) + reproducible harness

- Built a **fire-and-forget RunPod harness** (`scripts/runpod_job.sh`, branch `ops/runpod-job`, **PR #9**): captures all output to `/workspace/job.log` and ships it to W&B on exit (diagnosable after self-terminate); **STAGE A GPU-smoke gate** (a tiny MiniLM CUDA train) *before* paying for the real run; **STAGE B** graded `run_phase1.py`; ships `scorecard.md` + `partition_hashes.json` to W&B; then self-terminates. `RUN_RETENTION=1` toggle adds the Axis-3 MTEB guard.
- Bring-up cleared **5 real blockers** (all baked into the script): torch ≥ 2.6 cu124 for BGE-M3 `.bin` loading (CVE-2025-32434); remove torchvision/torchaudio (torch-2.6 ABI break cascades into a transformers import failure); CachedMNRL + encode batch 16 + `expandable_segments` for 24 GB OOM; single-quoted `docker_args` (GraphQL); take-first-available GPU loop (no 4090-community capacity).

**Result — first full two-stage run on real GPU (within-config, n = 222):**

| Run | Base / method | max_seq | NDCG@10 (zero-shot → fine-tuned) | 95% paired-bootstrap CI | MDE |
|---|---|---|---|---|---|
| RunPod-1 | BGE-M3 + LoRA, **two-stage + 1 hard neg** | 512 | **0.240 → 0.290** (Δ +0.050) | **[+0.027, +0.073]** — excludes zero | ±0.033 |

- **Defensible headline:** the gain is statistically real (CI excludes zero) and above the MDE, measured with a frozen+hashed BSARD partition. Cost ~$0.65 total (incl. fast-fails); pod self-terminated; 0 orphan pods.
- **Honest finding:** this full two-stage @512 (0.290) ≈ the Phase-0 Kaggle **Stage-1-only** @512 (0.292) → **Stage-2 hard-negatives added nothing measurable on this subset** (within noise). Not a failure — a measured result that points Phase-1 work at *more data / better negative filtering / margin tuning* (backlog L1/L4) rather than assuming Stage-2 helps.
- **Retention guard (Axis-3), second RunPod run (`RUN_RETENTION=1`, ~40 min):** base-vs-fine-tuned on the non-legal MTEB(fr)+BEIR subset, ±0.02 tolerance. **Verdict: FAIL — 1 of 7 tasks regressed.**

| Task | Lang | Before → After | Δ | Status |
|---|---|---|---|---|
| AlloprofRetrieval | FR | 0.490 → 0.475 | −0.015 | ok |
| MintakaRetrieval | FR | 0.222 → 0.234 | +0.012 | ok |
| SciFact | EN | 0.644 → 0.638 | −0.006 | ok |
| **FiQA2018** | EN | **0.413 → 0.385** | **−0.028** | ⚠️ **REGRESSED** |
| STSBenchmarkMultilingualSTS | FR | 0.824 → 0.842 | +0.018 | ok |
| SICKFr | FR | 0.785 → 0.784 | −0.001 | ok |
| AlloProfClusteringS2S | FR | 0.359 → 0.344 | −0.015 | ok |

  - **Every French task held; STS improved.** The one regression is **English financial QA** — the most out-of-domain task for a French-legal fine-tune — just past the ±0.020 tolerance.
  - **Root cause (confirmed):** `rehearsal_frac = 0.07` is defined in `config.py` but **never used in `train.py`** (zero references) — training ran on LegalKit pairs only, with no general-domain rehearsal floor.
  - **Fix (scoped, = the always-planned MVP insurance):** wire ~7% MS-MARCO/MIRACL FR/EN rehearsal into Stage-1 (TDD) → re-run → expect PASS. (Or narrow the guard, since EN finance is irrelevant to a French-legal product — but rehearsal is the principled fix.)
  - This run also **reproduced the BSARD headline**: 0.240 → **0.284** (Δ +0.044, CI [+0.021, +0.067], excludes zero) vs +0.050 in run 1 — two significant runs, ~+0.047 average.
  - **Takeaway for the defence:** the FAIL is a *feature* — the catastrophic-forgetting guard demonstrably works (it caught a real, mild regression on the single most distant task), the root cause is precise, and the fix is defined. Strong evidence for the BC03/BC05 evaluation blocks.

### 2026-07-03 — Rehearsal floor wired (TDD) + third RunPod run

- Implemented the anti-forgetting **rehearsal floor** that `rehearsal_frac` promised (PR #11, `feat/rehearsal-floor`), test-first:
  - `src/lexfr_embed/data/rehearsal.py` — pure `rehearsal_count` (solves r/(n+r)=frac) + `mix_rehearsal` (tags `code="rehearsal"`, interleaves, deterministic) with **5 hermetic tests**; plus a **defensive FR+EN loader**.
  - **Gotcha:** the obvious FR sources (`unicamp-dl/mmarco`, `facebook/mlqa`, `miracl/miracl`) are all **script-based** and refuse to load under `datasets>=3`. Working **parquet** sources: **`etalab-ia/piaf`** (FR Wikipedia QA) + **`sentence-transformers/natural-questions`** (EN).
  - Wired into `train_embedder(rehearsal_pairs=…)` (mixed into both stages) and loaded in `run_phase1` via `rehearsal_count`.
- **Third RunPod run** (`feat/rehearsal-floor`, `RUN_RETENTION=1`, ~42 min): 892 general pairs (frac 0.07) mixed into 11 848 legal → 12 740 total (loaded exactly as designed).

| | Axis-1 legal (BSARD) | Retention verdict | FiQA2018 (EN finance) |
|---|---|---|---|
| No rehearsal (run 2) | 0.240 → 0.284 (+0.044) | FAIL | −0.028 |
| **+ rehearsal (run 3)** | **0.240 → 0.292 (+0.052)**, CI [+0.031, +0.076] | FAIL | **−0.026** |

  - **Legal gain held (best of the three runs).** Three legal runs now: +0.050 / +0.044 / +0.052 — CI always excludes zero (robust ~+0.049 headline).
  - **Retention improved broadly:** vs the no-rehearsal run, AlloProfClustering −0.015 → **+0.020**, STS +0.018 → +0.020, Mintaka +0.012 → +0.018, SciFact +0.003; **all French fully protected.**
  - **But FiQA2018 still −0.026** (barely up from −0.028) → verdict still FAIL on that one task.
  - **Key insight:** open-domain EN rehearsal (Natural Questions = factoid Wikipedia) preserves the legal gain and lifts general tasks, but does **not** specifically protect FiQA's *financial* sub-domain — 7 % of open-domain pairs moved it only −0.002. Fully clearing it needs **domain-matched EN rehearsal** (MS-MARCO/web/financial) or a higher dose, *not* more of the same.
  - **Honest conclusion (stronger than a clean PASS):** the model gains significantly on legal retrieval, keeps **all** French + broad general capability, with a small, **quantified, understood** residual EN-financial trade-off. Domain-matched rehearsal is the documented next lever.

---

## 4. Consolidated decisions

| # | Decision | Choice | Why |
|---|---|---|---|
| D1 | Build vs reuse | **Build** | No open embedder for French *national* law exists |
| D2 | Project name | **lexfr-embed** / `lexfr_embed` | Over jurisembed / droit-embed |
| D3 | Base model | **BGE-M3** (MIT) primary; Qwen3-Embedding-0.6B (Apache); MiniLM = smoke fallback | Strong multilingual, permissive licence, fits 16 GB |
| D4 | Mistral embedder | **Excluded** (closed, not LoRA-able) | Use e5-mistral-7b / GritLM if a Mistral lineage is wanted |
| D5 | Training method | Contrastive **MNRL + MatryoshkaLoss**, **LoRA r16/α32, lr 1e-4**; Stage-2 hard negatives; synthetic practitioner queries | Portable from the Noumon recipe; fits a 16 GB T4/5060 Ti |
| D6 | Data | **LegalKit** CC-BY-4.0 (train), **BSARD** CC-BY-NC-SA (eval only); ~80–100k pairs target | Public data only (LDS corpus restricted) |
| D7 | Evaluation | **Build a French-national eval set** (Track A OOD + Track B in-distribution) | None exists; BSARD is Belgian |
| D8 | Compute | Kaggle (Phase 0, free) → local **RTX 5060 Ti 16 G** + RunPod burst (Phase 1) | Free validation; CUDA simplicity; cloud burst for sweeps |
| D9 | RunPod budget | **Realistic ~$150 / cap $250**, auto-stop everywhere | <$20 was wrong; operational risk dominates |
| D10 | OC framing | **OC15 capstone**, separate from OC14 | RNCP39775 Niveau 7, BC03/BC05 |
| D11 | LDS production | Deploy best **publicly-trained** model; **no private-data training** | Contractual no-training guarantee |
| D12 | Publication | arXiv + HF release; target **NLLP 2026** (Aug), JURIX, ECIR | NLLP is the best, reachable fit |
| D13 | Guard vs over-specialisation | **Before/after general-capability retention check** on a non-legal MTEB(fr)+BEIR subset (±0.02 tolerance) | A LoRA legal fine-tune can erode general FR/EN retrieval — measure it, don't assume |

---

## 5. Git history

| Commit | Branch | Message |
|---|---|---|
| `b52e221` | `main` | `chore: scaffold lexfr-embed — French legal-domain embedder` |
| `da0eb76` | `feat/phase0-kaggle-runner` | `feat: self-contained Phase-0 Kaggle runner (baseline->fine-tuned BSARD delta)` |
| `b5c608f` | `feat/phase0-kaggle-runner` | `fix: self-contained, T4-robust Phase-0 Kaggle runner (verified BSARD NDCG@10 0.055->0.148)` |
| `32629f2` | `feat/phase0-kaggle-runner` | `feat: BGE-M3 + LoRA Phase-0 config; verified on Kaggle T4 (BSARD NDCG@10 0.240->0.292)` |

**PR #1** (`feat/phase0-kaggle-runner` → `main`) is **open, awaiting user merge**. `main` currently holds only the scaffold.

---

## 6. Known issues / open items

- **No French-national eval set built yet** — spec is in `docs/eval-set-spec.md`, construction is Phase-1 work.
- **Hard-negative mining and synthetic-query generation are stubs** — the two biggest Phase-1 levers, not yet implemented.
- **BSARD is Belgian** — current numbers are cross-jurisdiction transfer; a French-national eval is needed for a defensible headline.
- **BGE-M3 long-context edge under-tested** — max_seq capped at 512 on the T4 (1024 tried, minor gain); revisit on the local card.
- **Phase-1 entrypoint not runnable yet** — `train.py` is a skeleton (raises `NotImplementedError`); `hard_negatives.mine()` and `synthetic_queries.{generate_queries,consistency_filter}` are stubs. The validated end-to-end path is `scripts/phase0_kaggle.py`.
- **GPU bring-up undocumented** — the MSI 5060 Ti is Blackwell (sm_120) → needs a CUDA 12.8 / recent torch build (same class of issue as the Kaggle Pascal failure). No local bring-up checklist yet.

---

## 7. What's next

- **Phase 1 (→ 2026-07-07):** implement Stage-2 hard negatives + synthetic/practitioner queries; scale to ~80–100k pairs + more epochs; compare base models (BGE-M3, Qwen3-0.6B, possibly e5-mistral); build the French-national eval set (Tracks A/B); **run the general-capability retention check (`scripts/eval_general.py`) before/after each run** (D13); quantize for deployment. Run serial overnight on the local 5060 Ti + RunPod burst for parallel sweeps.
- **Phase 2 (post-deadline):** arXiv preprint (cs.CL + cs.IR) + HF model/dataset release; submit to NLLP 2026. **Publication strategy done** (11-agent adversarial workflow, 2026-06-23): 6 ranked plays, joint-top = P1 reframed NLLP resource paper (the confound-control diagnostic; doubles as the capstone) + P5 governance paper with Primavera de Filippi. Full plan: `~/vault/personal/projects/research-career/lexfr-embed-publication-strategy.md` (HTML `/doc/lexfr-embed-publication-strategy`).

---

## 8. Pointers

- **Memory:** `~/.claude/projects/-home-gdelabie-code/memory/law-embedder-project.md` (the consolidated source record).
- **Research reports:** `~/code/law-embedder/docs/research/` (10 reports) + vault HTML `/doc/law-embedder-report`.
- **Proposal:** `~/code/law-embedder/docs/proposal/` + vault HTML `/doc/law-embedder-proposition`.
- **Hardware:** `~/vault/work/projects/legal-finetune/p710-hardware-feasibility.md`; `server-p710` skill reference.
- **Specs in-repo:** `docs/data-and-evaluation.md` (train-vs-eval map — read first), `docs/eval-set-spec.md` (French eval-set build + retention), `docs/publication-venues.md`.
- **Code (Phase-1 eval):** `src/lexfr_embed/general_eval.py` + `scripts/eval_general.py` (retention check) + `tests/test_general_eval.py`.
- **Publication strategy:** `~/vault/personal/projects/research-career/lexfr-embed-publication-strategy.md` (HTML `/doc/lexfr-embed-publication-strategy`).
