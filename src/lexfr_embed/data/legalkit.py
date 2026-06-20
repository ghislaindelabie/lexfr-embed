"""Load and shape LegalKit into (anchor, positive) training pairs.

`louisbrulenaudet/legalkit` ships ~53k LLM-generated (query -> article) pairs across
16 French codes, CC-BY-4.0 (research §04). The network loader is thin; the value is in
the **pure helpers** below (dedup, stratify), which are unit-tested in tests/test_data.py.

Format rule (sentence-transformers): column *order* matters, not names — (anchor, positive).
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

Pair = dict  # {"anchor": str, "positive": str, "code": str}


def _norm(text: str) -> str:
    """Normalise for dedup: lowercase, strip accents, drop punctuation, collapse spaces.

    Order matters: drop punctuation BEFORE collapsing whitespace + stripping, so a
    trailing "?"/"!" can't leave a stray trailing space.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^\w ]+", " ", text)  # punctuation -> space
    return re.sub(r"\s+", " ", text).strip()  # collapse + trim last


def dedup_pairs(pairs: list[Pair]) -> list[Pair]:
    """Drop pairs whose (normalised anchor, normalised positive) was already seen.

    Legal text is full of boilerplate / amended near-duplicates; in-batch-negative
    losses turn those into false negatives (research §03), so dedup is mandatory.
    """
    seen: set[tuple[str, str]] = set()
    out: list[Pair] = []
    for p in pairs:
        key = (_norm(p["anchor"]), _norm(p["positive"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def stratify_by_code(pairs: list[Pair], target_n: int, seed: int = 42) -> list[Pair]:
    """Sample ~target_n pairs balanced across `code` (research §03: diversity > volume).

    LegalKit is skewed toward Code du travail (~17%); over-training on one sub-domain
    hurts generalisation (NitiBench, research §05). This caps each code's share so the
    embedder sees every register, not just labour law.
    """
    import random

    rng = random.Random(seed)
    by_code: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        by_code[p.get("code", "unknown")].append(p)
    if not by_code:
        return []

    per_code = max(1, target_n // len(by_code))
    out: list[Pair] = []
    for _code, items in by_code.items():
        rng.shuffle(items)
        out.extend(items[:per_code])
    rng.shuffle(out)
    return out[:target_n]


def load_legalkit(subset_size: int | None = None, *, stratify: bool = True, seed: int = 42):
    """Load LegalKit -> deduped, optionally stratified list[Pair]. Requires `datasets`.

    Schema confirmed from the live HF viewer (2026-06-20): config "default", split "train",
    ~53k rows, CC-BY-4.0. Fields: `query` (the question), `output` (the article text),
    `input` (e.g. "Code civil, art. 265-2" — the code name is the prefix), `num`. There is
    no dedicated code column -> derive it from `input`.
    NB: the singular repo `louisbrulenaudet/legalkit` is THIS flat pair dataset; the per-code
    `louisbrulenaudet/code-*` repos have a DIFFERENT schema (raw articles, no query) — don't mix.
    """
    from datasets import load_dataset

    from lexfr_embed.config import settings

    ds = load_dataset(settings.legalkit_id, split="train")
    pairs: list[Pair] = [
        {
            "anchor": row["query"],
            "positive": row["output"],
            "code": (row.get("input") or "").split(", art.")[0].strip() or "unknown",
        }
        for row in ds
    ]
    pairs = [p for p in pairs if p["anchor"] and p["positive"]]
    pairs = dedup_pairs(pairs)
    if stratify and subset_size:
        pairs = stratify_by_code(pairs, subset_size, seed=seed)
    elif subset_size:
        pairs = pairs[:subset_size]
    return pairs
