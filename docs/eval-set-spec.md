# French-national legal-IR evaluation set — construction spec

*For Léo's approval. No public French-**national** legal-IR benchmark exists (BSARD/LLeQA are Belgian), so we build a small, credible held-out set. Researched 2026-06-20.*

## Design: two tracks, reported side by side

| Track | Role | Source | ~Size |
|---|---|---|---|
| **A — OOD, headline** | the number we defend | **service-public.fr fiches** (real citizen questions → cited Légifrance articles via `LienExterne`, Licence Ouverte 2.0, multi-label) **+ ~100 hand-curated practitioner queries** | ~250–350 queries |
| **B — in-distribution, diagnostic** | overfitting check, clearly flagged | **held-out LegalKit slice** (LLM-generated queries → article) | ~200 queries |

The **A↔B gap is itself a finding** (large gap ⇒ overfitting to the generator's phrasing).

**Why this is enough:** IR methodology floor is ~50 queries (≥150 for good power); MLEB (2025 SOTA legal benchmark) ships NDCG@10 tasks of just **65–500 queries**. The binding constraint is **label quality + corpus realism**, not raw count.

## Corpus

Index the **full article set of the in-scope codes** (~10–30k articles — BSARD/LLeQA/STARD run at 22k–55k), keyed by canonical `(code, num)` from Légifrance / LegalKit dumps so corpus, labels, and training share one ID space.

## Code coverage (report as a table; avoid travail/commerce skew)

Civil · Travail · Pénal · Commerce · Consommation · Fiscal/monétaire (+ housing/social where service-public fiches are rich).

## Labelling / verification

1. Auto-extract `(fiche question, cited Légifrance article ids)` from the service-public XML `LienExterne` refs; (b)/(d) authored directly against Légifrance ids.
2. Normalise every label to `(code, article num)` — the same key LegalKit exposes.
3. **Léo spot-checks 15–20 %** of Track A: confirm the cited article genuinely answers the question; drop fiches whose citation set is too broad. Record the clean/agreement rate.
4. Treat Track A as **multi-label** (a query may have >1 relevant article).

## Leakage control (critical — we train on LegalKit)

- Build the set of `(code, num)` article ids used as **answers in the LegalKit training split**; for Track A **prefer queries whose gold articles are NOT in that set**; where overlap is unavoidable, **flag and report metrics with/without** them.
- Track B is in-distribution by construction — partition by article id **and** row; label it synthetic.
- Text-dedup eval queries vs training `output` strings; drop near-duplicates.
- **Freeze + hash** the eval set; never used in training/HPO.

## Metrics

**NDCG@10 + Recall@100** primary (MTEB/MLEB convention); also MRR, R@10. **Bootstrap confidence intervals** (sizes are above the IR floor but below the high-power regime). Report Track A and B side by side.

## General-capability retention (catastrophic-forgetting guard)

Contrastive fine-tuning (MNRL) on a narrow *legal* distribution can degrade the model's **general** French/English behaviour — catastrophic forgetting / representation collapse toward legalese. The legal eval above will **not** catch this (it is legal-only). So every Phase-1 run also reports a **before-vs-after retention regression** on a small, fixed, strictly **non-legal** benchmark subset (MTEB(fr) + BEIR).

| Family | Tasks (MTEB) | Why |
|---|---|---|
| **FR retrieval** | AlloprofRetrieval, SyntecRetrieval, MintakaRetrieval | the deployed capability (general, non-legal) |
| **EN retrieval** | SciFact, FiQA2018 (BEIR) | multilingual base → confirm EN didn't collapse |
| **FR STS** | STSBenchmarkMultilingualSTS, SICKFr | semantic-geometry sanity check |
| **FR clustering** | AlloProfClusteringS2S | optional extra signal |

**Protocol:** score the base model, then the fine-tuned model, on the identical suite; report per-task Δ. **Acceptance:** the legal metric rises meaningfully **while** each general task drops by **no more than ±0.02** (absolute NDCG@10 / Spearman / V-measure ≈ within noise). A larger drop ⇒ over-specialised. Verify at the **deployed Matryoshka dim** (e.g. 256/512), not just full 1024.

**If it regresses:** lower LR / fewer epochs · keep LoRA and report adapter-on vs adapter-off (toggling the adapter off recovers the base) · *rehearsal*: mix ~5–15 % general pairs (MS-MARCO / MIRACL slice) into training · or accept it *iff* the product only ever serves legal queries (state the scope explicitly).

**Implementation:** `scripts/eval_general.py` (before/after deltas + PASS/FAIL, exit-code-gated) over `src/lexfr_embed/general_eval.py` (the suite + pure verdict logic, unit-tested in `tests/test_general_eval.py`); needs the `mteb` `eval` extra. The retention suite is deliberately **legal-free** (BSARD / Track A/B handled above). Bonus: an explicit forgetting check strengthens the OC evaluation blocks (BC03/BC05).

## The "LLM-eval just rewards the generator's style" risk

Real for us — training (LegalKit) and a tempting eval source share the LLM-query paradigm. STARD shows lay queries crater retrieval (R@100 ≈ 0.91) vs near-saturation on synthetic. Mitigations (all in the recipe): headline is non-synthetic (Track A); report both distributions; Léo spot-check; any synthetic augmentation uses a **different generator/prompt than LegalKit** + human verification; ID-level leakage filter.

## Limitations to state honestly in the report

Purpose-built (not a community standard); Track A relevance is fiche-level/multi-label (coarser than single-article) and reflects service-public editorial coverage; Track B is synthetic/in-distribution (diagnostic only); sizes sit above the IR minimum but below high-power → report CIs, treat small deltas cautiously; coverage skewed to source-rich codes (show per-code breakdown).

## Checklist (Léo sign-off)

- [ ] Corpus: full in-scope code articles (~10–30k), keyed by `(code, num)`
- [ ] Track A: ~250–350 q (service-public fiches + ~100 hand-curated), multi-label
- [ ] Track B: ~200 held-out LegalKit q, partitioned by id+row, flagged synthetic
- [ ] Code-coverage table; no travail/commerce skew
- [ ] Leakage filter (exclude/flag LegalKit-answer ids; text-dedup); freeze+hash
- [ ] Léo verifies 15–20 % of Track A; record clean rate
- [ ] NDCG@10 + R@100 (+MRR, R@10), bootstrap CIs, A vs B side by side
- [ ] Limitations paragraph

## Sources

BSARD (ACL 2022) · LLeQA (AAAI 2024) · STARD (EMNLP 2024 Findings) · GerDaLIR (NLLP 2021) · LegalBench-RAG (2024) · MLEB (2025) · IR power analyses (Webber; Buckley & Voorhees) · service-public.fr open data (Licence Ouverte 2.0, daily XML; "Textes de référence – Légifrance" linking since 2023) · LegalKit (`louisbrulenaudet/legalkit`).
