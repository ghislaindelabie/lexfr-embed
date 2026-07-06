"""Build the Track-B v2 artifact -> results/trackb_v2.json.gz (deterministic, CPU-only).

Recipe + rationale in lexfr_embed.data.trackb_v2. Run once (a few minutes):
    CUDA_VISIBLE_DEVICES="" uv run --no-sync python scripts/build_trackb_v2.py
Evaluate with: uv run --no-sync python scripts/eval_extra.py --mode powered --split trackb2 ...
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

OUT = Path("results/trackb_v2.json.gz")


def main() -> None:
    from lexfr_embed.data.legalkit import load_legalkit
    from lexfr_embed.data.trackb import build_holdout, trained_ids
    from lexfr_embed.data.trackb_v2 import build_v2_eval

    t0 = time.time()
    all_pairs = load_legalkit(None)
    tids = trained_ids()
    held = build_holdout(all_pairs, tids)
    held_gold_texts = {p["positive"] for p in held}
    trained_texts = {p["positive"] for p in all_pairs if p["positive"] not in held_gold_texts}
    # distractors = every article that is not a held-out gold (includes trained positives)
    distractors = sorted(trained_texts)
    print(f"[build] pool={len(held)} heldout-gold-texts={len(held_gold_texts)} distractors={len(distractors)}")

    queries, corpus, relevant = build_v2_eval(held, distractor_texts=distractors, trained_texts=trained_texts)
    n_multi = sum(1 for r in relevant.values() if len(r) > 1)
    print(f"[build] v2: queries={len(queries)} corpus={len(corpus)} multilabel={n_multi}")

    payload = {
        "queries": queries,
        "corpus": corpus,
        "relevant": {q: sorted(r) for q, r in relevant.items()},
        "meta": {
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recipe": "fail@1 TF-IDF filter on full corpus; pure-leak golds dropped; neardup>0.8 multi-label",
            "n_queries": len(queries),
            "n_corpus": len(corpus),
            "n_multilabel": n_multi,
            "corpus_hash": hashlib.sha256("\n".join(sorted(corpus.values())).encode()).hexdigest()[:16],
            "queries_hash": hashlib.sha256("\n".join(sorted(queries.values())).encode()).hexdigest()[:16],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"[build] wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
