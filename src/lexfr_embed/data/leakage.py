"""Train/eval leakage-control spine — pure, hermetic.

Hard-exclude eval gold (code, num) ids from training; order-independent partition hashes
(written to results/partition_hashes.json BEFORE any hard-negative mining); and a whitelist
that bars NC/SA sources (BSARD/LLeQA) from any training / mining / synthesis index.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

_WS = re.compile(r"\s+")
_BARRED = ("bsard", "lleqa")  # CC-BY-NC-SA -> never enters a training/mining/synthesis index


def _norm(s) -> str:
    return _WS.sub(" ", str(s).strip().lower())


def canonical_id(code, num) -> str:
    """Canonical article id 'code|num' — case/whitespace-insensitive, idempotent."""
    return f"{_norm(code)}|{_norm(num)}"


def hard_exclude(train_ids: Iterable[str], eval_ids: Iterable[str]) -> set[str]:
    """Training id set with every eval id removed — hard exclusion, not 'prefer'."""
    return set(train_ids) - set(eval_ids)


def hash_partition(ids: Iterable[str]) -> str:
    """Order-independent sha256 over the deduped, sorted canonical ids (integrity fingerprint)."""
    h = hashlib.sha256()
    for i in sorted(set(ids)):
        h.update(i.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def is_barred_source(name: str) -> bool:
    """True if a source is non-commercial (NC/SA) and must never enter a training/mining index."""
    n = (name or "").lower()
    return any(b in n for b in _BARRED)


def filter_pairs_by_ids(pairs: list[dict], barred_ids: Iterable[str], id_key: str = "id") -> list[dict]:
    """Drop pairs whose `id_key` value is in `barred_ids` (e.g. eval gold excluded from training)."""
    barred = set(barred_ids)
    return [p for p in pairs if p.get(id_key) not in barred]
