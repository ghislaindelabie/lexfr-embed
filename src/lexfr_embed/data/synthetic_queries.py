"""Synthetic query generation (Promptagator-style) — research §03/§04.

Two jobs:
  1. Generate (query -> article) pairs for under-covered codes to balance LegalKit.
  2. Diversify the *register*: rephrase formal queries into **practitioner language**
     (jargon, abbreviations like "art. 1240 C. civ.", elliptical phrasing), grounded in
     the open sigles/abbreviation dictionary + service-public `related_questions` style.

Why synthetic: real practitioner Q&A / documents have no open licence (forums are
copyright + DB sui-generis). Synthetic rephrasing sidesteps that cleanly (research §04).

Generator: Mistral batch API (<$6, no GPU) by default; a local open LLM is an alt.
Apply **consistency filtering** (keep a query only if its source article is retrieved
in the top-k by a baseline retriever) — the single most important quality step.
"""

from __future__ import annotations

PROMPTAGATOR_SYSTEM = (
    "Tu es juriste français. Génère {n} requêtes de recherche variées qu'un avocat "
    "taperait pour retrouver l'article suivant. Types: factuelle, définitionnelle, "
    "procédurale, cas pratique. Utilise le registre praticien (jargon, abréviations "
    "type 'art. 1240 C. civ.', 'Cass. com.'). Une requête par ligne, sans numéro."
)


def build_prompt(article_text: str, n: int = 4) -> list[dict]:
    """Build a chat prompt for the generator. (Pure — unit-testable.)"""
    return [
        {"role": "system", "content": PROMPTAGATOR_SYSTEM.format(n=n)},
        {"role": "user", "content": article_text},
    ]


def generate_queries(articles: list[str], n_per_article: int = 4) -> list[dict]:
    """TODO(Phase 1): call the Mistral batch API with build_prompt(), parse lines,
    return [{"anchor": query, "positive": article}]. Then run consistency_filter()."""
    raise NotImplementedError("Wire the Mistral batch API (needs MISTRAL_API_KEY).")


def consistency_filter(pairs: list[dict], retriever, k: int = 20) -> list[dict]:
    """TODO(Phase 1): keep a synthetic pair only if `retriever` ranks its `positive`
    article in the top-k for its `anchor` query. Drops hallucinated/off-topic queries."""
    raise NotImplementedError("Implement consistency filtering with a baseline retriever.")
