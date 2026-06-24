# Data & evaluation — what we train on vs what we measure

*The most important discipline in this project: keep **training** and **evaluation** cleanly separated, focus legal quality on **French** sources, and **guarantee no regression on general language**. This doc is the single source of truth for which dataset plays which role. Pairs with [`eval-set-spec.md`](eval-set-spec.md) (how the French eval set is built) and [`publication-venues.md`](publication-venues.md).*

## 1. The map (read this first)

Every dataset has exactly **one** role. A dataset used in training is **never** used in evaluation.

| Dataset | Role | Lang / jurisdiction | Queries | Licence | Status |
|---|---|---|---|---|---|
| **LegalKit** (`louisbrulenaudet/legalkit`) | 🟩 **TRAIN** | French / **national** | LLM-generated | CC-BY-4.0 | ✅ in use |
| Synthetic + practitioner-register queries | 🟩 TRAIN (Phase-1 augment) | French | LLM-generated, **consistency-filtered** | derived (own) | 🔴 to build |
| Hard negatives | 🟩 TRAIN (Stage-2) | French | mined from the corpus | derived (own) | 🔴 to build |
| **BSARD** (`maastrichtlawtech/bsard`) | 🟦 **EVAL — legal, transfer** | French / **Belgian** | real citizen | CC-BY-NC-SA | ✅ in use (eval only) |
| `tax-retrieval-benchmark` (Brulé Naudet, MTEB) | 🟦 EVAL — legal, French (optional) | French / national (**tax**) | synthetic | CC-BY *(verify)* | 🟡 candidate |
| **Track A** | 🟦 **EVAL — legal, French, headline** | French / national, multi-code | **real** (service-public.fr) + curated practitioner | Licence Ouverte 2.0 | 🔴 to build |
| **Track B** | 🟦 EVAL — legal, French, diagnostic | French | held-out LegalKit (**synthetic**) | derived | 🔴 to build |
| **General-retention suite** | 🟨 **EVAL — no-regression guard** | FR + EN, **non-legal** | MTEB(fr) + BEIR | various open | ✅ coded (`general_eval.py`) |

Three colours = three different questions: 🟩 *what shapes the weights*, 🟦 *is it good at French legal retrieval*, 🟨 *did it forget general language*.

## 2. What we TRAIN on

- **LegalKit** — ~53k French `(query → article)` pairs. Fields: `query` (the question), `output` (the article text), `input` (`"Code civil, art. 265-2"` → the code is the prefix). The **queries are LLM-generated**, which is why generator-style overfitting is a real risk we must control for at eval time (see §3b and the *same-generator confound* in `eval-set-spec.md`).
- **Phase-1 augmentations** (not yet built): synthetic `(query → article)` pairs for under-covered codes, plus **practitioner-register** rephrasings (jargon, `art. 1240 C. civ.`, elliptical phrasing) — every synthetic query passes a **consistency filter** (kept only if a baseline retriever ranks its source article in the top-k). Hard negatives are mined from the same corpus (1 filtered negative/query).
- **Public data only.** We train exclusively on openly-licensed sources (LegalKit CC-BY; any DILA/Légifrance text under Etalab). **No LDS private/client corpus ever enters training** — the LDS production model is simply whichever publicly-trained variant benchmarks best.

## 3. What we EVALUATE on — two distinct questions

### 3a. "Is it good at **French legal** retrieval?" (the point of the project)

| Set | What it tells us | Honest caveat |
|---|---|---|
| **BSARD** | Real lay-citizen questions → statutory articles. Our current headline number (zero-shot 0.24 → fine-tuned 0.31 NDCG@10). | **Belgian** law → this is a *cross-jurisdiction transfer* proxy, not French-national. |
| **Track A** *(to build)* | The defensible French-national number: real service-public.fr questions cited to Légifrance, multi-code, multi-label. | Relevance is fiche-level (coarser than single-article); reflects service-public editorial coverage. |
| **Track B** *(to build)* | Diagnostic only: held-out LegalKit (synthetic, in-distribution). The **A↔B gap is itself a finding** (large gap ⇒ overfitting to the generator's phrasing). | Synthetic / in-distribution — never the headline. |
| `tax-retrieval-benchmark` *(candidate)* | An **external, citable French-national** data point (tax domain). Adopting it strengthens French coverage **and** pre-empts a reviewer objection (it is must-cite prior art). | Tax-only; synthetic queries; verify size/composition/licence before adopting. |

> **Why BSARD alone isn't enough:** it's Belgian. The French-national signal comes from Track A (real) + optionally `tax-retrieval-benchmark` (external). That is the "French legal focus" you asked for.

### 3b. "Did it **forget general language**?" (the no-regression guard — top priority right now)

Contrastive fine-tuning on a narrow legal distribution can pull the embedding space toward legalese and **degrade general French/English retrieval + similarity**. The legal eval above cannot see this. So **every Phase-1 run reports a before-vs-after retention regression** on a small, fixed, strictly **non-legal** suite:

| Family | Tasks (MTEB) | Why |
|---|---|---|
| FR retrieval | AlloprofRetrieval, SyntecRetrieval, MintakaRetrieval | the deployed capability, non-legal |
| EN retrieval | SciFact, FiQA2018 (BEIR) | multilingual base → confirm EN didn't collapse |
| FR STS | STSBenchmarkMultilingualSTS, SICKFr | semantic-geometry sanity |
| FR clustering | AlloProfClusteringS2S | optional extra signal |

**Acceptance contract:** the legal metric rises **while** each general task drops by **no more than ±0.02** (absolute NDCG@10 / Spearman / V-measure ≈ within noise). A larger drop ⇒ over-specialised → mitigate (lower LR / fewer epochs · LoRA adapter-on-vs-off · rehearsal with ~5–15 % general pairs). Verify at the **deployed Matryoshka dim** (e.g. 256/512), not just full 1024. Implemented in `src/lexfr_embed/general_eval.py` + `scripts/eval_general.py` (PASS/FAIL, exit-code-gated).

## 4. Train ↔ eval leakage discipline

Because we **train on LegalKit** and one tempting eval source is also LegalKit-derived:

- **ID-level filter:** collect the `(code, num)` article ids used as *answers* in the LegalKit **training** split; for **Track A**, prefer queries whose gold articles are **not** in that set; where overlap is unavoidable, **flag and report metrics with/without** them.
- **Track B** is in-distribution by construction → partition by article id **and** row; label it synthetic.
- **Text-dedup** eval queries against training `output` strings; drop near-duplicates.
- **Freeze + hash** every eval set; never used in training or HPO.
- **BSARD is external** (Belgian corpus, different text) → naturally leakage-free vs LegalKit training — a useful property, even though it's a transfer proxy.

## 5. Metrics

- **Legal:** NDCG@10 + Recall@100 primary (MTEB/MLEB convention); also MRR, R@10. **Bootstrap CIs** (set sizes sit above the IR floor but below high-power → treat small deltas cautiously). Report Track A and B **side by side**.
- **Retention:** same per-task main metric, reported as before → after Δ with the ±0.02 verdict.
- Always report at the **deployed embedding dimension**.

## 6. Where each lives in code

| Concern | Code |
|---|---|
| Train data load (LegalKit) | `src/lexfr_embed/data/legalkit.py` |
| Synthetic / practitioner queries | `src/lexfr_embed/data/synthetic_queries.py` *(stub)* |
| Hard-negative mining | `src/lexfr_embed/data/hard_negatives.py` *(stub)* |
| Legal eval (BSARD + held-out) | `src/lexfr_embed/evaluate.py` |
| French eval set construction | `docs/eval-set-spec.md` (Track A/B) — *builder to write* |
| General-language retention guard | `src/lexfr_embed/general_eval.py` + `scripts/eval_general.py` |
| Validated end-to-end reference | `scripts/phase0_kaggle.py` |

## 7. Limitations to state honestly (mentor + paper)

BSARD is Belgian (transfer proxy); Track A relevance is fiche-level/multi-label; Track B is synthetic/in-distribution (diagnostic only); eval sizes are above the IR minimum but below high-power → report CIs; coverage skews to source-rich codes (show a per-code breakdown). The retention suite is a *subset* of MTEB, not the full benchmark — it's a guard, not a general-embedding leaderboard claim.

## 8. Open decisions

1. **Adopt `tax-retrieval-benchmark` as an eval set?** Adds a real French-national (tax) data point and is must-cite prior art — recommended, pending a check of its size/composition/licence.
2. **Track-A code coverage** — confirm the in-scope codes (Civil · Travail · Pénal · Commerce · Consommation · Fiscal + housing/social) and the ~250–350 query target.
3. **Retention tolerance** — is ±0.02 the right bar, or stricter for the deployed dim?
4. **Retention training-mix** — if regression appears, is rehearsal (mixing general pairs) acceptable for the OC/LDS scope, or do we accept legal-only specialisation and state it?
