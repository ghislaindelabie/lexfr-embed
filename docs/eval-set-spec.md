# French-national legal-IR evaluation set — construction spec

*For Léo's approval. No public French-**national** legal-IR benchmark exists (BSARD/LLeQA are Belgian), so we build our own. **Revised 2026-06-25** to the corrected primary use-case: the headline is **professional query→article + graph relatedness**; lay-citizen sets are a secondary robustness axis. See [`data-and-evaluation.md`](data-and-evaluation.md) for the four-axis map.*

## Tracks (build priority order)

| Track | Role | Source | ~Size | Confound-free? |
|---|---|---|---|---|
| **P — professional query→article** | 🟦 **HEADLINE** (the job) | real/curated **professional** queries (jargon, citations, scenario+cite) **or** confound-controlled synthetic | ~150 (IR floor) | only if real/curated, or via different-generator control |
| **G — graph / related-provisions** | 🟦 graph axis (agent use-case) | **extracted** citations/renvois as gold (GerDaLIR/CLERC style) | a few k pairs | ✅ yes (no LLM) |
| **R — lay robustness** *(secondary)* | 🟦 "does it also handle laypeople" | **BSARD** (Belgian) + later service-public.fr fiches (`LienExterne`→Légifrance, Licence Ouverte 2.0, multi-label) | BSARD 222 / SP ~250 | ✅ (real lay) |
| **B — synthetic diagnostic** | 🟦 overfitting check, *flagged* | held-out LegalKit slice (LLM-generated) | ~150–200 | ❌ diagnostic only |

**The P↔B gap (and a same-generator vs different-generator gap on synthetic) is itself a finding** — it measures how much score is real skill vs generator-style memorisation.

**Why small is enough:** IR floor ~50 queries (≥150 for decent power); MLEB (2025 SOTA legal benchmark) ships NDCG@10 tasks of just **65–500 queries**. Binding constraint = **label quality + register/realism**, not raw count.

## Track P — the professional headline (the hard, unbuilt one)

Real professional queries are scarce/private (same wall as the LDS corpus). Two viable paths, ideally combined:
1. **Real/curated:** a small set of genuine practitioner queries (Léo / a jurist authors ~100–150), each mapped to gold `(code, num)` articles. Best signal; expensive.
2. **Confound-controlled synthetic:** generate eval queries with a **different generator family than training** (and a professional persona), human-verify a sample, and **report the same-generator vs different-generator gap**. The gap quantifies generator-style inflation.

Precedents: JuriFindIT (expert-authored), COLIEE (exam register). Register: jargon, abbreviations, article citations, elliptical phrasing — *not* full-sentence lay questions.

## Track G — graph / related-provisions (confound-free)

Use **real citations as relevance** (GerDaLIR/CLERC pattern): extract renvois between statutory articles (and citations in jurisprudence) → `(article, cited/related article)` gold pairs. Tests whether the model encodes the legal graph the agents traverse.
- **Filter procedural/generic renvois** (*"sous réserve de l'article X"*) — they link topically-unrelated articles (false positives).
- **Graph leakage control (critical):** pair-level disjointness is **not enough** — split the **graph** (disjoint article sets / connected components) so a train article never cites an eval article; dedup near-duplicate/consolidated articles.
- This is a **doc↔doc** eval — a real capability axis, but **not a substitute for Track P** (a full article ≠ a short query).

## Corpus

Index the **full article set of the in-scope codes** (~10–30k articles — BSARD/LLeQA/STARD run at 22k–55k), keyed by canonical `(code, num)` from Légifrance / LegalKit dumps so corpus, labels, and training share one ID space.

## Code coverage (report as a table; avoid travail/commerce skew)

Civil · Travail · Pénal · Commerce · Consommation · Fiscal/monétaire (+ housing/social where sources are rich).

## Labelling / verification

1. **Track P:** authored/curated against Légifrance ids (real) or generated-then-verified (synthetic). **Track R (service-public):** auto-extract `(fiche question, cited Légifrance ids)` from the XML `LienExterne` refs. **Track G:** extract renvoi/citation edges.
2. Normalise every label to `(code, article num)` — the same key LegalKit exposes.
3. **Léo spot-checks 15–20%** of Tracks P and R: confirm the cited article genuinely answers the query; drop over-broad citation sets. Record the clean/agreement rate.
4. Treat P and R as **multi-label** (a query may have >1 relevant article).

## Leakage control (critical — we train on LegalKit)

- **Hard-exclude** (not merely "prefer") the headline eval's gold `(code, num)` ids from the LegalKit **training split**; report any unavoidable overlap only as a separate labelled secondary metric.
- **Graph (Track G):** split by graph component (see Track G above), not by pair.
- **Track B** is in-distribution by construction → partition by article id **and** row; flag synthetic; dedup synthetic queries vs training `output` strings **and** vs eval-set query text.
- **Freeze + hash** every eval set before any mining/relabeling; never used in training/HPO.
- **Whitelist:** BSARD and any NC/SA source are mechanically barred from the training / mining / consistency-filter / synthesis-seed indexes (LDS is commercial).

## Metrics

**NDCG@10 + Recall@100** primary (MTEB/MLEB convention); also MRR, R@10. **Bootstrap confidence intervals** (sizes near the IR floor). Report all tracks side by side. **Power analysis:** compute the minimum detectable Δ at each track's size before declaring small deltas meaningful.

## General-capability retention (catastrophic-forgetting guard)

Contrastive fine-tuning on narrow legal text can degrade **general** FR/EN behaviour; the legal tracks can't see it. So every run reports a **before→after retention regression** on a fixed, strictly **non-legal** MTEB(fr)+BEIR subset:

| Family | Tasks (MTEB) | Why |
|---|---|---|
| **FR retrieval** | AlloprofRetrieval, SyntecRetrieval, MintakaRetrieval | the deployed capability (general, non-legal) |
| **EN retrieval** | SciFact, FiQA2018 (BEIR) | multilingual base → confirm EN didn't collapse |
| **FR STS** | STSBenchmarkMultilingualSTS, SICKFr | semantic-geometry sanity check |
| **FR clustering** | AlloProfClusteringS2S | optional extra signal |

**Acceptance:** the legal metric rises **while** each general task drops by **no more than ±0.02** (≈ noise) at the **deployed Matryoshka dim**. Larger drop ⇒ over-specialised → mitigate (lower LR / fewer epochs · LoRA adapter-on-vs-off · rehearsal ~5–25% *relevant* general pairs · checkpoint soup). **Implementation:** `scripts/eval_general.py` over `src/lexfr_embed/general_eval.py` (unit-tested; `mteb` `eval` extra). Strengthens OC blocks BC03/BC05.

## The "LLM-eval just rewards the generator's style" risk

Real for us — training (LegalKit) and a tempting eval source share the LLM-query paradigm. STARD shows lay queries crater retrieval (R@100 ≈ 0.91) vs near-saturation on synthetic. **A train/eval split does NOT fix this** (the confound is shared *style*, not shared *items*). Mitigations: headline (Track P) is real/curated *or* different-generator-controlled with the gap reported; Track G is extraction-based (confound-free); Léo spot-check; any synthetic uses a **different generator/prompt than LegalKit** + human verification; ID-level + graph leakage filters.

## Limitations to state honestly in the report

Track P is the hardest to source (real pro queries are private) and may lean on confound-controlled synthetic; Track G is doc↔doc (a capability, not the query→article job); Track R is lay + (BSARD) Belgian; sizes near the IR floor → report CIs; coverage skews to source-rich codes. Purpose-built, not a community standard.

## Checklist (Léo sign-off)

- [ ] **Track P** (headline): ~150 professional queries — real/curated and/or different-generator-controlled synthetic; gold `(code,num)`; gap reported
- [ ] **Track G**: extracted citation/renvoi pairs; procedural renvois filtered; **graph-split** (not pair-split)
- [ ] **Track R** (secondary): BSARD + (later) service-public fiches, multi-label
- [ ] **Track B**: held-out LegalKit, partitioned by id+row, flagged synthetic
- [ ] Corpus: full in-scope code articles (~10–30k), keyed by `(code, num)`; code-coverage table
- [ ] Leakage: hard-exclude headline ids; graph-split G; text-dedup; freeze+hash; NC/SA whitelist
- [ ] Léo verifies 15–20% of P and R; record clean rate
- [ ] NDCG@10 + R@100 (+MRR, R@10), bootstrap CIs, power analysis, all tracks side by side
- [ ] Retention guard before/after at deployed dim
- [ ] Limitations paragraph

## Sources

BSARD (ACL 2022) · LLeQA (AAAI 2024) · STARD (EMNLP 2024 Findings) · **GerDaLIR (NLLP 2021 — citation-as-relevance)** · **CLERC (NAACL 2025)** · G-DSR / "Finding the Law" (EACL 2023 — French/Belgian, hierarchy lever) · JuriFindIT (Findings-EACL 2026 — expert queries) · MLEB (2025) · IR power analyses (Webber; Buckley & Voorhees) · service-public.fr open data (Licence Ouverte 2.0) · LegalKit (`louisbrulenaudet/legalkit`).
