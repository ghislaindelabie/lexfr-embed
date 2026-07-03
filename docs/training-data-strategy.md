# Training-data strategy — what we train on, and why

*A decision-ready, adversarially-hardened answer to: what training signal does the embedder need? Keep LegalKit or self-generate? Apply OC14-style multi-LLM convergence? Follow Noumon and use EU docs? Use EU-French law? Produced by an 11-agent workflow (5 scouts → synthesis → 3 adversarial critics → finalize), 2026-06-24. Pairs with [`data-and-evaluation.md`](data-and-evaluation.md) (train-vs-eval map) and [`eval-set-spec.md`](eval-set-spec.md).*

> **Status: proposal with open decisions** (see §8). Nothing here is locked until you sign off the MVP scope.

> **⚠️ Update (2026-06-25) — primary use-case correction.** The main users are **legal professionals + querying agents** (professional register, citation/graph expectations), **not laypeople**; lay-question→article is a *later* phase. This re-frames evaluation: the headline is now **professional query→article + graph relatedness**, and the lay sets (BSARD, service-public) become a **secondary "robustness" axis** — see the revised [`data-and-evaluation.md`](data-and-evaluation.md) and [`eval-set-spec.md`](eval-set-spec.md) for the four-axis suite. It also **upgrades the renvoi / citation-graph idea from "auxiliary" to "on-axis"** (professionals/agents leverage the graph) — though **code *hierarchy* is the higher-priority, evidence-backed structural lever** (G-DSR/BSARD: mAP 35→47), logged as flagship **L5** in [`optimization-backlog.md`](optimization-backlog.md) ahead of the speculative renvoi (L12). The register framing in **R5 / §3 below predates this correction** (it treats lay as the headline); read it through the professional lens.

## TL;DR — verdict

**BUILD, but ship a tight MVP first and hold the research-grade layers for Phase 2.** The original design was directionally right on every axis; the adversarial pass did not overturn a single verdict — it **falsified one load-bearing number**, showed several "literature-backed" framings are weaker than claimed, and proved the Phase-1 plan was **~2× over-scoped** for the ~9 working days to the 2026-07-07 OC submission. So the *justifications and sequencing* change more than the conclusions.

| Q | Verdict |
|---|---|
| **Pairs vs document exposure / DAPT?** | **Pairs only.** No MLM/DAPT/TSDAE stage. (Document *structure* → free pairs is a Phase-2 lever.) |
| **Keep LegalKit, or self-generate?** | **Keep as backbone, audit-don't-trust.** Self-generate for *coverage + register*, not volume — Phase 1.5+. |
| **OC14 multi-LLM convergence?** | **Partial.** Diversity via a persona taxonomy from one *different-family* generator; free consistency filter. The cross-family *jury* is **not** literature-backed for IR labeling → Phase 2. |
| **Follow Noumon (EU docs train/eval)?** | **Yes to the split discipline + portable recipe lessons**; his 0.966 is directional precedent, not a target. |
| **EU-French law as a training source?** | **Yes in principle, but cut entirely from the OC submission** → Phase 2. |

### ⏱️ Deadline-critical actions (do these regardless)
1. **Gate-protect** `scripts/phase0_kaggle.py` + its BSARD 0.240→0.307 delta as the **frozen Friday mentor-gate artifact**. Do all of the below on branches that cannot break it.
2. **Canary (cheap, day-1):** run `scripts/eval_general.py` on the **existing** Phase-0 fine-tuned checkpoint. If even that small Stage-1-only fine-tune already trips the ±0.02 retention guard, you learn it *now* — with time to pivot (lower LR / stronger LoRA / more rehearsal) — not on 07-04.
3. **Cut to MVP** (see §5/§6): one base (BGE-M3), full audited LegalKit + **1** mined hard negative + a baked-in small general-rehearsal floor + a ~150-query Track A + one Phase-0-informed LR. Everything else is Phase 2.

---

## 1. What a contrastive embedder learns from (the core question)

A retrieval bi-encoder (BGE-M3, Qwen3-Embedding) does **not** learn from text the way a language model does. It learns from **relational** signal: `(anchor query → relevant passage)` pairs. The contrastive loss (MNRL / CachedMNRL) does two things per batch — pull each query's vector toward its positive article, and push it away from every *other* article in the batch (in-batch negatives) plus any explicit hard negative. **A lone, unpaired article produces no gradient.** So "train on the codes" can only mean: (a) **mine pairs and hard negatives** from them, or (b) **index** them for eval. Feeding raw articles directly would be masked-language-modelling (DAPT) — a *different* objective.

**Why DAPT / raw-document exposure is the wrong lever *for us*** (a regime argument, not a law of nature):

1. **The record actually says DAPT often *helps* retrieval — when you have no in-domain labels.** Honest reading of GPL (Wang et al., NAACL 2022, Table 1, supervised target→MS-MARCO): no-pretrain 45.2, **MLM 46.7 (+1.5), TSDAE 49.2 (+4.0)**, TSDAE+GPL 52.9. *(An earlier draft of this analysis misquoted this as "MLM hurts by −4.8"; that was false and is corrected here — see §7.)* So the anti-DAPT case is **not** "exposure hurts retrievers."
2. **The case that survives is the regime mismatch.** Those gains are for the *zero-in-domain-labels* setting. We are the opposite: 53k→~90k supervised in-domain pairs, a base that already did pair-pretraining, and a $150 budget. With supervised pairs in hand, a separate unsupervised stage is **low-ROI** and competes for budget/time better spent on pairs + one good hard negative.
3. **The base already did the exposure stage, pair-shaped, for free.** Every strong base (E5, BGE-M3, Qwen3-Embedding, Arctic) = massive weakly-supervised contrastive **pair** pretraining → small supervised fine-tune with mined hard negatives. Choosing BGE-M3 **inherits** that. Re-running exposure on 90k pairs with $150 is the from-scratch tactic, not the strong-base tactic.

**Practical upshot:** the entire training signal is well-curated `(query → positive)` pairs + **1 margin-filtered hard negative** per query, under CachedMNRL wrapped in MatryoshkaLoss. Raw codes / jurisprudence / EUR-Lex earn their keep **only** as (i) the pool to *synthesise* pairs from and (ii) the pool to *mine hard negatives* from — never as raw training rows.

**One nuance the critique added (and it's right):** "document exposure" ≠ "MLM/DAPT". There is a third, **objective-aligned** option — **free structure-derived contrastive pairs** mined from document structure (heading→body, article↔cross-referenced article / *renvoi*). Pair-shaped, no LLM cost, no MLM. Not MVP, but a legitimate "expose to the codes" lever for the Phase-2 ablation menu — possibly the cheapest per-dollar way to teach *renvoi* structure.

**For your top priority (no general-language regression)** the levers are **not** DAPT but: **LoRA** (low-rank acts as an implicit regulariser; adapter-off instantly recovers the base), 1–2 epochs at modest LR, a **small general-rehearsal floor baked in from run 1** (not held in reserve — there's no slack to react to a late trip), the exit-gated retention guard at the deployed Matryoshka dim, and optionally model merging. **DAPT would make forgetting worse, not better.**

---

## 2–6. The five decisions

### R1 — Pairs only; no DAPT · *confidence: High*
**Verdict:** pairs-only training rows; **no** MLM/DAPT/TSDAE/SimCSE stage; **yes** to document *structure* as a Phase-2 source of free pairs.
**How:** keep the recipe — CachedMNRL ⊂ MatryoshkaLoss (dims [1024,768,512,256,128,64]), LoRA r16/α32, batch 128 / mini-batch 4, Stage-1 ~3 epochs + Stage-2 (1 hard neg) ~2 epochs at lower LR. Use raw codes / `AgentPublic/legi` / EUR-Lex **only** to mine pairs+negatives and to index for eval. If a mentor asks "why no domain pretraining?", give the **regime** answer (label-rich, strong base, tight budget) and cite GPL's *actual* Table 1.
**Effort/risk:** zero for MVP (removes a tempting stage) · low risk.

### R2 — Keep LegalKit as backbone, audit-don't-trust · *confidence: High (keep); Medium (own-data share)*
**Verdict:** MVP = full deduped + per-code-stratified LegalKit, **no** new generation. Phase 1.5/2 → rebalance toward ~50–60% LegalKit / ~40–50% own data, self-generating for **coverage + register + dedup**, not volume.
**Why:** LegalKit (53k pairs, CC-BY-4.0, *real* Légifrance text) is the best ready-made French national resource — but it's a **single-generator** (LLaMA-3-70B) backbone with **no datasheet**, unaudited legal relevance (the screen was fluency), and coverage skew (travail 17% / civil 5.4% / pénal 2.3%). Field mapping is already correct in `legalkit.py`.
**How:** MVP — all ~53k after `dedup_pairs` + `stratify_by_code`; partition by canonical `(code,num)`; Léo spot-checks 50–100 pairs to quantify relevance error. Phase 1.5 — generate to a per-code **floor**; **gate** register-diversity generation on an embedding-space separability check on a small pilot *before* committing 25–30k pairs (if Mistral & LLaMA queries aren't separable, the rationale collapses). Consistency-filter with a **different-family** model; dedup the **merged** pool and dedup synthetic queries against **Track-A/B query text**, not just LegalKit outputs.
**Effort/risk:** low MVP · medium Phase-1.5 · main risk = Mistral substituting its own register bias (the pilot gate settles it).

### R3 — Multi-LLM convergence: partial · *confidence: High*
**Verdict:** get diversity from a **persona/prompt taxonomy** out of **one different-family** generator (Mistral); use the **free consistency filter** (with a non-base model) for quality. The cross-family **relevance jury + Gecko relabeling are Phase 2**, reframed as a project-specific finding — **not** a literature-backed default.
**Why:** OC14's two halves transfer unequally. (a) Multiple *generator* models: weak support — SOTA embedders get query diversity from a prompt/persona taxonomy out of *one* strong model. (b) Consensus *filtering*: the headline citation (Verga 2024 PoLL) judges **generated-answer** semantic containment, **not** retrieval relevance — a different task. So the jury is an unproven add-on *and* the main new API cost — the "cargo-culting?" risk made real.
**How:** MVP/1.5 — persona taxonomy (lay-citizen / practitioner-abbreviated / procedural / definitional / fact-scenario) × code, generated with Mistral (breaks the LLaMA-3 monoculture); free consistency filter on the 5060 Ti with a **different-family** model (avoid circularity), log discards. Phase 2 only — a 2–3 model cross-family jury **excluding LLaMA-3** (the LegalKit generator family, to avoid re-importing self-preference bias), calibrated against Léo's human spot-checks *first*, reported with Fleiss/Krippendorff; Gecko relabeling **only** after both eval tracks are frozen+hashed with hard exclusion of eval gold ids.
**Effort/risk:** low MVP · medium Phase-2 API (capped to post-consistency survivors).

### R4 — Follow Noumon's *discipline*, not his number · *confidence: High*
**Verdict:** yes to the **split discipline** + portable recipe lessons; treat 0.966 as directional precedent, not a target. **Correction:** Noumon split *one* corpus by chunk-ID 85/15 and used a *different* document (GDPR) as the genuine held-out test.
**Why:** his credible evidence was **cross-document transfer** (+7–11 nDCG on unseen GDPR), not the in-corpus 0.966 (single-generator-style confound, 340 queries / 85 chunks, no CIs, and achieved with **Qwen3-Embedding-4B** — his 560M-class model scored lower). Portable lessons: **ID-level split** (never pair-level); **1 filtered hard negative** (his 5-raw-negs *regressed* 0.9398 vs 0.9463 — direct support for our choice); the exact loss stack; and **run the LR sweep he skipped and regretted**. Our analog of his GDPR held-out test already exists: **BSARD** (external Belgian) = cross-corpus transfer; **Track A** = headline.
**How:** keep `(code, num)` split, per-code stratified; Stage-2 near-verbatim (1 filtered negative, mined *with* the Stage-1 model, `relative_margin` ~0.05–0.1, `sampling_strategy='top'`, `use_faiss=True`); one Phase-0-informed LR for MVP, the 3-LR×3-epoch grid is Phase 2.
**Phase-2 false-negative guard (documented for later):** the `relative_margin` filter is cheap but imperfect, and French legal queries are often **multi-label** — so a mined "negative" can be a real unlabelled answer. Before trusting mined negatives at scale, add a confirmation pass that scores each `(query, candidate)` with a **cross-encoder reranker** (or an **LLM judge**, from a different family than the embedder) and **drops candidates it rates relevant**; gate/sample it for cost and log the reject rate (a high one flags a noisy mining step). Hook: `src/lexfr_embed/data/hard_negatives.py`.
**Effort/risk:** low MVP · caveat: he won at 2,284 pairs on one doc; at 16 codes/90k, enforce per-`(code,num)` partition and quantify near-duplicate rate first.

### R5 — EU-French law: yes in principle, cut from the submission · *confidence: High (cut); Medium (lift)*
**Verdict:** Phase 2 unconditionally. EUR-Lex stays the recommended EU source; cite **Commission Decision 2011/833/EU**, not a blanket "CC-BY-4.0".
**Why:** EU-French law is genuinely part of the LDS/practitioner surface and adds register/subject diversity, and EUR-Lex has clean leakage (no shared `(code,num)` id space with national evals). But it's the **least aligned** source with the top priority (national-legal quality), the strategy itself concedes the gain *might be diversity rather than Track-A lift*, and Formex/CELLAR parsing is brand-new plumbing. 2–3 days of a 9-day window on a maybe-no-lift source is a misallocation.
**How (Phase 2):** source only from EUR-Lex under Decision 2011/833/EU (attribution + no-endorsement in the dataset card; check third-party-IP carve-outs per CELEX doc); parse Formex → articles/recitals; same Mistral + persona + consistency pipeline; **subject**-stratified; split by CELEX/document id + version-dedup across consolidations; optional held-out **Track C** (diagnostic only); **with/without-EU ablation** on Track A + retention guard. **Avoid:** JRC-Acquis (non-commercial), MultiEURLEX/mteb-eurlex (classification, CC-BY-SA copyleft); verify LEMUR's true licence before reuse.
**Effort/risk:** medium, fully off-critical-path, Phase 2 only.

---

## 5. Recommended data mix

**Don't chase volume past ~100k** — spend effort on dedup, per-code stratification, and hard-negative quality, where returns are real.

**MVP mix (ships by 2026-07-07 — single base BGE-M3, single LR):**
- **LegalKit backbone:** full deduped + per-code-stratified ~50–53k pairs. Léo spot-checks 50–100; partition by `(code,num)`. *Essentially all the training data for MVP.*
- **General rehearsal:** ~5–10% MS-MARCO/MIRACL FR/EN pairs **baked in from run 1** — proactive no-regression insurance (no slack to react late).
- **Hard negatives:** exactly **1** margin-filtered negative/query (margin ~0.05–0.1, `top`, faiss, mined *with* the Stage-1 model). **Never 5+ raw** (Noumon regressed).
- **Self-generated / EU / jury / Gecko:** 0%.

**Phase-1.5 / 2 target mix (~90k, after MVP ships and the diversity gate passes):**
- LegalKit (audited, partitioned): ~48–53k (~55%)
- Self-generated national balancing + register (Mistral, persona taxonomy, per-code floor, non-base consistency filter, *gated*): ~25–30k (~30%)
- EU-French (EUR-Lex, Decision 2011/833/EU, subject-stratified, ablated): ~10–12k (~12%)
- General rehearsal floor: ~5–10% (widen to 15% only if the guard still trips)
- *Optional:* free structure-derived (heading→body, *renvoi*) pairs as an ablation arm

Dedup the **merged** pool before training; dedup synthetic queries against **Track-A/B query text** too.

---

## 6. Phase-1 build plan (deadline-aware)

1. **Gate-protect** `phase0_kaggle.py` + its 0.240→0.307 delta as the frozen Fri mentor artifact; all R1–R5 work on branches that can't break it. *Phase 0 is the gate; R1–R5 target 07-07 only.*
2. **Canary (day-1):** retention suite on the existing Phase-0 checkpoint → detect forgetting early.
3. **Freeze+hash** the train/eval `(code,num)` partition **first**, before any synthesis/mining/relabel. Make Track-A leakage **hard-exclude** (gold ids disjoint from training); report unavoidable overlaps only as a separate labelled metric. Quantify overlap rate + per-code disjoint size first.
4. **Whitelist** indexable/seedable corpora (LegalKit, DILA/Légifrance; Phase-2 CC-BY EUR-Lex). **BSARD and any NC/SA source are mechanically barred** from the mining index, the consistency-filter index, the relabel pool, and the synthesis seed pool — commercial-incompatible text can't leak into a training artifact.
5. **MVP trainer:** make `train.py` run two-stage end-to-end locally (or extend `phase0_kaggle.py` as fallback). **Lock BGE-M3** and fix `config.py` (currently defaults to `qwen3-0.6b`). Defer multi-base + e5-mistral-7b QLoRA.
6. **MVP data:** full deduped + stratified LegalKit + the baked-in ~5–10% rehearsal floor. No new synthetic for MVP.
7. **MVP hard negatives** (highest-ROI, ~20 LOC): wrap `mine_hard_negatives` for 1 filtered negative.
8. **MVP eval:** Track A at the **~150-query IR floor** the spec cites (not 250–350), real service-public.fr, multi-label, Léo verifies 15–20%; Track B held-out LegalKit (~150–200, diagnostic). Run `eval_general.py` before/after every run, exit-gated, at the deployed Matryoshka dim. **Power analysis:** compute the minimum-detectable NDCG@10/Spearman delta at Track-A size before declaring ±0.02 meaningful; if MDE>0.02, state the guard catches only gross regression.
9. **MVP GPU:** schedule the 5060 Ti **sm_120 bring-up** as a dated task with a CUDA 12.8 / recent-torch checklist (same class as the Kaggle Pascal failure). RunPod fallback with auto-stop; one LR.
10. **Ship MVP**, then Phase 1.5/2 sequenced: (a) embedding-space diversity gate → persona synthetic gen with a non-base filter; (b) the LR×epoch grid; (c) cross-family jury (no LLaMA-3) + Gecko behind frozen evals; (d) EUR-Lex + ablation; (e) structure-derived pairs. Frame the confound-control design (A↔B gap + retention guard + leakage discipline) as the OC report's central methodological contribution and the NLLP-2026 seed.

---

## 7. What the adversarial hardening caught (so you can trust the rest)

| # | Original claim | Reality | Fix |
|---|---|---|---|
| 1 | "MLM-before-contrastive *hurt* −4.8 nDCG (46.7 vs 51.5)" | **False.** GPL Table 1 (supervised regime): MLM **+1.5**, TSDAE **+4.0**. | Anti-DAPT case re-grounded on the **regime** argument; cite GPL honestly. |
| 2 | Cross-family relevance **jury** is literature-backed (Verga PoLL) | PoLL judges **generated-answer** containment, not IR relevance | Jury demoted to Phase-2 project-finding, validated vs Léo's spot-checks first |
| 3 | "InPars+ 70B underperformed 8B" → small models suffice | Paper attributes gains to **CPO+DSPy prompts**, not model-size inversion | Dropped; re-grounded on E5-mistral/Qwen3 prompt-taxonomy practice |
| 4 | Noumon 0.966 as a target; "€0.75 proves method beats volume" | 0.966 was **Qwen3-4B**, 340-query single-doc, no CIs | Keep only verified portable lessons (1 filtered neg; real 5-neg regression; run the LR sweep) |
| 5 | Consistency-filter with BGE-M3 to train BGE-M3 | **Circular** (self-selection) | Mandate a **different-family** filter model + discard logging |
| 6 | EU prefer-not-overlap leakage; "EUR-Lex = CC-BY-4.0" | "Prefer" is too soft; licence is **Decision 2011/833/EU** | **Hard-exclude** leakage; cite the real instrument + third-party-IP carve-out |
| 7 | Plan presented as Phase-1-buildable | **~2× over-scoped** for ~9 days | Explicit MVP cut; synthetic/jury/Gecko/EUR-Lex/LR-grid → Phase 2 |

Critique severities: project-fit **High**, confound/leakage/licence **High**, evidence/methodology **Medium**. No verdict was overturned.

---

## 8. Open decisions (your call)

1. **MVP scope sign-off** *(biggest)* — confirm: BGE-M3 only; full audited LegalKit + 1 hard neg + ~5–10% rehearsal; Track A ~150 q; one LR = the gradeable 07-07 deliverable; synthetic/LR-grid/jury/Gecko/EUR-Lex all Phase 2.
2. **Base lock** — BGE-M3 for submission (and fix `config.py`'s `qwen3-0.6b` default)?
3. **Rehearsal posture** — accept baking a ~5–10% general-rehearsal floor into run 1 vs holding in reserve?
4. **Léo availability (dated dependency)** — can he verify 15–20% of Track A in-window? If not, pre-agree the BSARD + Track-B fallback headline.
5. **Retention realism** — if the power analysis shows MDE > 0.02 at Track-A size, enlarge the suite or restate the guard as gross-regression-only?
6. **Synthetic-query redistribution licence** — do Mistral/LLaMA-3 output-use terms permit a CC-BY-4.0 release of generated queries for a *commercial* (LDS) artifact? Verify before any public dataset release.
7. **EUR-Lex inclusion (Phase 2)** — worth the Formex/CELLAR plumbing given the conceded maybe-no-Track-A-lift, or keep it purely as an LDS-breadth play settled by the ablation?

---

## 9. Key references

- **GPL** (Wang et al., NAACL 2022) — Table 1: MLM +1.5, TSDAE +4.0, TSDAE+GPL +7.7 nDCG@10 (no-label regime). https://ar5iv.labs.arxiv.org/html/2112.07577
- **Replacing Judges with Juries (PoLL)** (2024) — panel judges generated-answer containment, *not* IR relevance. https://arxiv.org/html/2404.18796
- **InPars+** (2025) — gains from CPO+DSPy prompts, not small-beats-large. https://arxiv.org/abs/2508.13930
- **Noumon — fine-tuning embeddings on legal/regulatory text** (2025) — 0.9658 was Qwen3-4B; 1 filtered neg beats 5 raw. https://danielnoumon.com/blog/embedding-finetuning/
- **Commission Decision 2011/833/EU** — the actual EUR-Lex reuse instrument (attribution + no-endorsement + third-party-IP carve-out). https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32011D0833
- **STARD** (2024) — lay queries hard in absolute terms (R@100 ~0.91); supports A↔B gap as a hypothesis to *test*. https://arxiv.org/abs/2406.15313
- **BSARD** (2022) — Belgian French statutory retrieval; eval-only, CC-BY-NC-SA (must stay out of any training/mining index). https://arxiv.org/abs/2108.11792
- **LegalKit** (`louisbrulenaudet/legalkit`, 2024) — ~53k FR (query→article) pairs, CC-BY-4.0, LLaMA-3-70B-generated over Légifrance; the audited training backbone. https://huggingface.co/datasets/louisbrulenaudet/legalkit
