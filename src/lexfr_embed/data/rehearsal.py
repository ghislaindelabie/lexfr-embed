"""Rehearsal floor — the anti-forgetting insurance the retention guard proved we need.

Fine-tuning contrastively on legal pairs ONLY drifts the model's farthest capabilities
(the Axis-3 guard caught an EN-financial-QA regression). The classic fix is **rehearsal**:
mix a small fraction of *general* (query, positive) retrieval pairs into training so the
model keeps rehearsing the general skill while it specialises. `settings.rehearsal_frac`
(≈0.07) sets the target share.

Pure helpers (`rehearsal_count`, `mix_rehearsal`) are unit-tested (no network).
`load_rehearsal_pairs` is a thin, defensive integration loader (general FR + EN retrieval
pairs from public HF datasets); it is exercised only on a real run.
"""

from __future__ import annotations

import random

REHEARSAL_CODE = "rehearsal"  # tag so mixed-in pairs are auditable / excludable downstream


def rehearsal_count(n_legal: int, frac: float) -> int:
    """How many rehearsal pairs to add so they are `frac` of the *combined* set.

    Solving r/(n+r)=frac gives r = frac/(1-frac)·n. So 7 % of the total, not 7 % of legal.
    """
    if frac <= 0.0 or n_legal <= 0:
        return 0
    return round(frac / (1.0 - frac) * n_legal)


def mix_rehearsal(legal_pairs: list[dict], rehearsal_pairs: list[dict], seed: int = 42) -> list[dict]:
    """Tag rehearsal pairs, concatenate with the legal pairs, and deterministically shuffle.

    Returns the legal list unchanged (same object) when there is nothing to mix, so the
    smoke / rehearsal-off path is a no-op. Legal pairs are never mutated.
    """
    if not rehearsal_pairs:
        return legal_pairs
    tagged = [
        {**p, "code": p.get("code", REHEARSAL_CODE)} if p.get("code") else {**p, "code": REHEARSAL_CODE}
        for p in rehearsal_pairs
    ]
    combined = [*legal_pairs, *tagged]
    random.Random(seed).shuffle(combined)  # interleave so batches see both distributions
    return combined


def load_rehearsal_pairs(n: int, seed: int = 42) -> list[dict]:
    """Load ~`n` general (anchor, positive) retrieval pairs — FR + EN, public, ungated.

    Defensive: tries several sources and returns whatever loads, so one dataset being
    unavailable never blocks a run. Splits `n` across languages to guard FR *and* EN.
    """
    if n <= 0:
        return []
    from datasets import load_dataset

    # (repo, config, split, query_col, positive_col) — small slices; ungated, PARQUET (no dataset
    # scripts, which HF datasets>=3 refuses to run). Verified loadable 2026-07-03.
    sources = [
        ("etalab-ia/piaf", None, "train", "question", "context"),  # FR QA over Wikipedia (general retrieval)
        ("sentence-transformers/natural-questions", None, "train", "query", "answer"),  # EN general retrieval
    ]
    per_source = max(1, n // len(sources))
    out: list[dict] = []
    for repo, config, split, qcol, pcol in sources:
        want = n - len(out) if repo == sources[-1][0] else per_source
        if want <= 0:
            continue
        try:
            sliced = f"{split}[:{want * 3}]"  # over-fetch, then filter empties
            ds = load_dataset(repo, config, split=sliced) if config else load_dataset(repo, split=sliced)
            picked = 0
            for row in ds:
                q, p = (row.get(qcol) or "").strip(), (row.get(pcol) or "").strip()
                if q and p:
                    out.append({"anchor": q, "positive": p, "code": REHEARSAL_CODE})
                    picked += 1
                    if picked >= want:
                        break
        except Exception as e:  # noqa: BLE001 - a missing/renamed dataset must not kill the paid run
            print(f"[rehearsal] source {repo} unavailable ({str(e)[:80]}) — skipping")
    random.Random(seed).shuffle(out)
    return out[:n]
