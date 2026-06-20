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
