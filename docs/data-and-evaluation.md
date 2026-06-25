# Data & evaluation — what we train on vs what we measure

*The most important discipline in this project: keep **training** and **evaluation** cleanly separated, focus legal quality on **French** sources, and **guarantee no regression on general language**. This doc is the single source of truth for which dataset plays which role. Pairs with [`eval-set-spec.md`](eval-set-spec.md) (how the French eval sets are built), [`training-data-strategy.md`](training-data-strategy.md) (what to train on), and [`optimization-backlog.md`](optimization-backlog.md) (experiments).*

## 0. Primary use-case (read this first — it drives everything below)

**The main users are legal PROFESSIONALS and querying AGENTS**, not laypeople. They query with **professional semantics** (jargon, article citations, abbreviations like `art. 1240 C. civ.`) and **expect the model to understand the legal graph/structure** (renvois / cross-references → "give me this article *and* its related provisions"). **Lay-citizen questions are a *later* phase, not the headline goal.** This reframes the whole eval: the headline is **professional query→article** + **graph relatedness**, and lay sets become a **secondary "robustness" axis**.

## 1. The map (which dataset plays which role)

Every dataset has exactly **one** role. A dataset used in training is **never** used in evaluation.

| Dataset | Role | Lang / jurisdiction | Queries | Licence | Status |
|---|---|---|---|---|---|
| **LegalKit** (`louisbrulenaudet/legalkit`) | 🟩 **TRAIN** | French / national | LLM-generated | CC-BY-4.0 | ✅ in use |
| Synthetic **professional-register** queries | 🟩 TRAIN (Phase-1 augment) | French | LLM-generated, **consistency-filtered** | derived | 🔴 to build |
| Hard negatives | 🟩 TRAIN (Stage-2) | French | mined + **denoised** | derived | 🔴 to build |
| Renvoi / hierarchy structure pairs | 🟩 TRAIN (Phase-2 ablation) | French | **extracted** (no LLM) | derived | 🟡 backlog |
| **Professional query→article** | 🟦 **EVAL — HEADLINE** | French / national | real/curated **professional** *or* confound-controlled synthetic | mixed | 🔴 **to build (the hard one)** |
| **Graph / related-provisions** | 🟦 **EVAL — graph axis** | French | **extracted** citations/renvois (GerDaLIR-style) | open | 🔴 to build (confound-free) |
| **General-retention suite** | 🟨 **EVAL — no-regression guard** | FR + EN, non-legal | MTEB(fr) + BEIR | open | ✅ coded (`general_eval.py`) |
| **BSARD** (`maastrichtlawtech/bsard`) | 🟦 EVAL — lay robustness *(secondary)* + external transfer | French / **Belgian** | real **lay** citizen | CC-BY-NC-SA | ✅ in use (eval only) |
| service-public.fr "Track A" | 🟦 EVAL — lay robustness *(secondary)* | French / national | real **lay** citizen | Licence Ouverte 2.0 | 🟡 later phase |
| `tax-retrieval-benchmark` (Brulé Naudet) | 🟦 EVAL — French external (optional) | French / national (tax) | synthetic | CC-BY *(verify)* | 🟡 candidate |

Three colours = three questions: 🟩 *what shapes the weights*, 🟦 *is it good at the French legal job*, 🟨 *did it forget general language*.

## 2. What we TRAIN on

- **LegalKit** — ~53k French `(query → article)` pairs (`query`=question, `output`=article, `input`=`"Code civil, art. X"`). Queries are **LLM-generated** → the *same-generator confound* must be controlled at eval time (§3, and `eval-set-spec.md`). Keep as backbone, **audit-don't-trust** (single LLaMA-3-70B generator, coverage skew).
- **Phase-1 augmentations** (not yet built): **professional-register** synthetic queries (jargon, citations, elliptical) for under-covered codes, each passing a **consistency filter** (kept only if a *different-family* retriever ranks its source article top-k); plus **1 denoised hard negative**/query.
- **Phase-2 (backlog):** **extracted** structure pairs — code **hierarchy** (heading→body; the *proven* lever, see backlog) and **renvoi** cross-references (speculative). No LLM, so confound-free.
- **Public data only.** No LDS private corpus ever trains the model.

## 3. What we EVALUATE on — a four-axis suite

The job is **professional query→article retrieval + graph understanding, without forgetting general language**. One number can't capture that; we report a **suite**, as a Pareto view (best *trade-off*, not best single score).

### 3a. Professional query→article — 🔴 THE HEADLINE (and the hard one)
What a lawyer/agent actually types: jargon, citations, scenario+citation. **This is unbuilt and genuinely hard**, because *real* professional queries are scarce/private (same wall as the LDS corpus). Options: a small **real/curated** professional query set (best, expensive); or **confound-controlled synthetic** (generate eval queries with a *different generator* than training, and **report the same-vs-different-generator gap** — that gap *is* the confound measurement). Precedents: JuriFindIT (expert queries), COLIEE.

### 3b. Graph / related-provisions — 🔴 doc↔doc, **confound-free**
"Given this article/citation, retrieve the right + related provisions." Built by **extracting real citations** (renvois in statutes, citations in jurisprudence) as gold — **GerDaLIR / CLERC** style. No LLM → **no generator confound**, and the relevance is *real professional behaviour*. Directly tests the agent/graph use-case. **Caveat:** this measures **doc↔doc relatedness**, which is a real axis — but it **does not replace 3a** (a full article is not a short query). Don't let this clean, easy-to-build eval quietly become the headline in place of 3a (we already made the "measure-what's-available" mistake once).

### 3c. General-language retention — 🟨 the no-regression guard (top priority)
Contrastive fine-tuning on narrow legal text can degrade general FR/EN retrieval + similarity; the legal eval can't see it. So **every run reports a before→after retention regression** on a fixed, strictly **non-legal** suite:

| Family | Tasks (MTEB) | Why |
|---|---|---|
| FR retrieval | AlloprofRetrieval, SyntecRetrieval, MintakaRetrieval | the deployed capability, non-legal |
| EN retrieval | SciFact, FiQA2018 (BEIR) | multilingual base → confirm EN didn't collapse |
| FR STS | STSBenchmarkMultilingualSTS, SICKFr | semantic-geometry sanity |
| FR clustering | AlloProfClusteringS2S | optional extra signal |

**Acceptance contract:** the legal metric rises **while** each general task drops by **no more than ±0.02** (≈ noise). Larger drop ⇒ over-specialised → mitigate (lower LR / fewer epochs · LoRA adapter-on-vs-off · rehearsal ~5–25% *relevant* general pairs · checkpoint soup). Verify at the **deployed Matryoshka dim**. Code: `general_eval.py` + `scripts/eval_general.py` (PASS/FAIL, exit-gated).

### 3d. Lay robustness — 🟦 secondary axis (was the old "headline")
**BSARD** (real lay, **Belgian**) and a future service-public.fr set (real lay, French) measure whether the model *also* handles laypeople. Keep them — but as a **secondary robustness axis and a later-phase concern**, not the headline. BSARD is also a useful **external cross-jurisdiction transfer** signal (leakage-free, because Belgian). Our validated Phase-0 number lives here (BSARD NDCG@10 0.24→0.31).

### The confound, stated precisely (so we don't fool ourselves)
- A **synthetic** eval is a fine **diagnostic** but a dangerous **headline** — it inflates scores by rewarding the generator's style.
- A train/eval **split does NOT fix the confound** — the confound is shared *style/distribution*, not shared *items*. Only a **non-synthetic** or **different-generator** eval measures reality.
- **Multiple generators mitigate** generator-*specific* bias but do **not** close the synthetic-vs-*real* gap. For eval, prefer a small **real/extracted** set (IR floor ~50–150 queries) over a large multi-generator synthetic one.
- **Extraction** (3b) is confound-free, but solves the **graph** axis only — not 3a.

## 4. Train ↔ eval leakage discipline

- **ID-level filter:** for the professional eval, gold `(code, num)` ids must be **hard-excluded** from the training split (not merely "preferred"); report any unavoidable overlap only as a separate labelled secondary metric.
- **Graph leakage (for 3b):** citations form a **graph** — pair-level disjointness is *not enough*. Split the **graph** (disjoint article sets / connected components) so a train article never cites an eval article; dedup near-duplicate/consolidated articles.
- **Synthetic diagnostic (Track B):** in-distribution by construction → partition by article id **and** row; flag synthetic; dedup synthetic queries against **eval-set query text**, not just LegalKit `output` strings.
- **Freeze + hash** every eval set before any mining/relabeling; never used in training or HPO.
- **Whitelist:** BSARD and any NC/SA source are mechanically barred from the training / mining / consistency-filter / synthesis-seed indexes (LDS is commercial).

## 5. Metrics

- **Legal:** NDCG@10 + Recall@100 primary; also MRR, R@10. **Bootstrap CIs** (sizes near the IR floor → treat small deltas cautiously). Run a **power analysis**: if the minimum detectable Δ exceeds ±0.02 at the eval size, state the guard detects only gross regression.
- **Retention:** per-task main metric, before→after Δ with the ±0.02 verdict.
- Always report at the **deployed embedding dimension** (Matryoshka).

## 6. Where each lives in code

| Concern | Code |
|---|---|
| Train data load (LegalKit) | `src/lexfr_embed/data/legalkit.py` |
| Synthetic / professional queries | `src/lexfr_embed/data/synthetic_queries.py` *(stub)* |
| Hard-negative mining (+ Phase-2 false-neg judge) | `src/lexfr_embed/data/hard_negatives.py` *(stub)* |
| Legal eval (BSARD + held-out) | `src/lexfr_embed/evaluate.py` |
| French eval-set construction | `docs/eval-set-spec.md` — *builders to write* |
| General-language retention guard | `src/lexfr_embed/general_eval.py` + `scripts/eval_general.py` |
| Validated end-to-end reference | `scripts/phase0_kaggle.py` |

## 7. Limitations to state honestly (mentor + paper)

The headline professional eval is the hardest to source (real pro queries are private); the graph eval is doc↔doc (a *capability*, not the query→article *job*); BSARD is lay + Belgian; eval sizes sit near the IR floor → report CIs; coverage skews to source-rich codes (show a per-code breakdown). The retention suite is a *subset* of MTEB — a guard, not a leaderboard claim.

## 8. Open decisions

1. **Professional-eval design** — real/curated vs confound-controlled synthetic (or both)? How many queries, who curates (Léo?), what register mix?
2. **Build the graph/citation eval (3b)?** Confound-free and on-axis for the agent use-case; needs graph-split + procedural-renvoi filtering. Recommended for Phase 2.
3. **Adopt `tax-retrieval-benchmark`** as an external French data point (also must-cite prior art)? Pending size/licence check.
4. **Retention tolerance** — keep ±0.02, or stricter at the deployed dim? Run the power analysis to confirm it's meaningful at our eval size.
5. **Lay phase** — when (if ever) does lay-citizen retrieval become a headline rather than a robustness axis?
