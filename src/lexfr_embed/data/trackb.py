"""Track-B — a large, powered, IN-DISTRIBUTION French-legal retrieval eval built from LegalKit.

The campaign proved BSARD (n=222, Belgian, lay, dev=test) cannot discriminate configs
(MDE ~0.05 >> the ~0.01-0.05 effects we chase). Track-B holds out LegalKit pairs that NO
trained subset saw (leak-free), then serves them in the SAME (queries, corpus, relevant)
shape as evaluate.load_bsard, so eval_extra's powered/rerank/matryoshka modes consume it
directly (`--split trackb`).

HONEST caveat — state it wherever a Track-B number appears: the queries are LLaMA-3-generated
(same generator family as the training pairs), so Track-B measures IN-DISTRIBUTION retrieval,
NOT the external professional truth. Its value is twofold: (a) large n -> tiny MDE (a *powered*
eval, unlike BSARD); (b) the BSARD(external transfer) vs Track-B(in-distribution) gap objectifies
the same-generator confound. It is a DIAGNOSTIC, not the publishable benchmark — that remains the
LEGI-renvois / jurisprudence-visa / expert-labeled roadmap (see PROJECT_LOG / lexfr-bilan-pistes).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_WS = re.compile(r"\s+")

# (subset_size, seed) pairs the campaign actually trained on (campaign_queue.json).
# Held-out = complement of the UNION -> leak-free for any evaluated checkpoint.
CAMPAIGN_SUBSETS: tuple[tuple[int, int], ...] = (
    (6_000, 42),
    (15_000, 42),
    (15_000, 43),
    (15_000, 44),
    (15_000, 45),
    (30_000, 42),
    (30_000, 43),
    (45_000, 42),
)


def _norm(s) -> str:
    return _WS.sub(" ", str(s).strip().lower())


def pair_id(pair: dict) -> str:
    """Stable text-based identity of a LegalKit pair.

    Matches the loader's dedup key `(_norm(anchor), _norm(positive))`, so a held-out pair is
    genuinely absent from training (LegalKit's loaded pairs carry no article number to key on).
    """
    return f"{_norm(pair['anchor'])}\x1f{_norm(pair['positive'])}"


def build_holdout(all_pairs: list[dict], trained_ids: Iterable[str]) -> list[dict]:
    """Pairs whose identity is NOT in any trained subset — the leak-free eval pool."""
    trained = set(trained_ids)
    return [p for p in all_pairs if pair_id(p) not in trained]


def build_trackb_eval(heldout_pairs: list[dict], n_queries: int | None = None, seed: int = 42):
    """Return (queries, corpus, relevant) — same shape as evaluate.load_bsard.

    corpus   = ALL held-out articles, deduped by text, as realistic distractors (deterministic
               cids by first-seen order);
    queries  = a code-stratified sample of size n_queries (all held-out, if None or >= available);
    relevant = {qid: {cid}} of each query's gold article (multi-label emerges naturally when
               several held-out queries share one article).
    """
    from lexfr_embed.data.legalkit import stratify_by_code

    text2cid: dict[str, str] = {}
    corpus: dict[str, str] = {}
    for p in heldout_pairs:
        t = p["positive"]
        if t not in text2cid:
            cid = f"a{len(text2cid)}"
            text2cid[t] = cid
            corpus[cid] = t

    if n_queries is None or n_queries >= len(heldout_pairs):
        sampled = list(heldout_pairs)
    else:
        sampled = stratify_by_code(heldout_pairs, n_queries, seed=seed)

    queries: dict[str, str] = {}
    relevant: dict[str, set[str]] = {}
    for i, p in enumerate(sampled):
        qid = f"q{i}"
        queries[qid] = p["anchor"]
        relevant[qid] = {text2cid[p["positive"]]}
    return queries, corpus, relevant


def trained_ids(subsets_seeds: Iterable[tuple[int, int]] = CAMPAIGN_SUBSETS) -> set[str]:
    """Union of pair identities across every trained (subset_size, seed) — needs `datasets`."""
    from lexfr_embed.data.legalkit import load_legalkit

    ids: set[str] = set()
    for n, s in subsets_seeds:
        ids.update(pair_id(p) for p in load_legalkit(n, seed=s))
    return ids


def load_trackb(n_queries: int | None = 5_000, seed: int = 42, subsets_seeds=CAMPAIGN_SUBSETS):
    """Live builder (hits HF): full LegalKit -> leak-free held-out -> (queries, corpus, relevant)."""
    from lexfr_embed.data.legalkit import load_legalkit

    all_pairs = load_legalkit(None)
    held = build_holdout(all_pairs, trained_ids(subsets_seeds))
    return build_trackb_eval(held, n_queries=n_queries, seed=seed)
