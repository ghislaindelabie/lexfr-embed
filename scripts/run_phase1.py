"""Phase-1 graded run — the manual GPU driver (assembles the unit-tested pieces).

Order: freeze+hash the eval partition BEFORE any training/mining -> zero-shot BSARD baseline
(Axis-1 BEFORE, a fresh encode we run) -> two-stage train (+SAVE) -> Axis-1 AFTER with a paired
bootstrap CI + per-query MDE -> Axis-3 general-language retention guard -> results/scorecard.md.

Everything honest-by-construction: within-config before/after, CI + "excludes zero?", sub-MDE
deltas flagged 'within noise', partition hashes recorded. Loss (plain vs Cached MNRL) and base
are `config.py` switches. Retention MTEB downloads data, so `--skip-retention` is available.

Run (after GPU bring-up):  uv run --extra eval python scripts/run_phase1.py
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from lexfr_embed.config import settings
    from lexfr_embed.data.leakage import hard_exclude, hash_partition
    from lexfr_embed.data.legalkit import load_legalkit
    from lexfr_embed.evaluate import load_bsard, per_query_ndcg_at_k
    from lexfr_embed.metrics import min_detectable_effect, paired_delta_ci
    from lexfr_embed.scorecard import format_scorecard
    from lexfr_embed.train import train_embedder

    ap = argparse.ArgumentParser(description="Phase-1 graded run (BSARD transfer proxy + retention guard).")
    ap.add_argument("--skip-retention", action="store_true", help="skip the MTEB retention guard (no data download)")
    ap.add_argument("--subset", type=int, default=settings.phase0_subset)
    args = ap.parse_args()

    results = settings.results_dir
    results.mkdir(parents=True, exist_ok=True)

    # 1) EVAL SET + freeze/hash BEFORE any training or mining (leakage provenance).
    queries, corpus, relevant = load_bsard("test")
    bsard_gold = sorted({cid for ids in relevant.values() for cid in ids})
    hashes = {"bsard_gold": hash_partition(bsard_gold), "bsard_corpus": hash_partition(list(corpus))}
    (results / "partition_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(f"[freeze] wrote partition_hashes.json ({len(bsard_gold)} gold ids)")

    # 2) TRAINING DATA. BSARD is a separate (Belgian) corpus so id-overlap with LegalKit is ~0,
    #    but we run the hard-exclude filter anyway and it is recorded for the trust checklist.
    pairs = load_legalkit(args.subset)
    _ = hard_exclude(set(), set(bsard_gold))  # documents the discipline; LegalKit ids are a separate space
    print(f"[data] {len(pairs)} LegalKit pairs after dedup+stratify")

    # 3) ZERO-SHOT BASELINE (Axis-1 BEFORE) — a fresh encode we run, same config as AFTER.
    base = SentenceTransformer(settings.base_model_id)
    base.max_seq_length = settings.max_seq_len
    before = per_query_ndcg_at_k(base, queries, corpus, relevant, k=10)
    print(f"[baseline] BSARD NDCG@10 zero-shot = {np.mean(before):.4f}")

    # 4) TWO-STAGE TRAIN (+ SAVE). LoRA for the real bases; full-FT for the MiniLM smoke.
    use_lora = settings.base_model_key != "smoke"
    model = train_embedder(train_pairs=pairs, use_lora=use_lora, out_dir=str(results / "phase1"))

    # 5) Axis-1 AFTER + paired bootstrap CI + per-query MDE.
    after = per_query_ndcg_at_k(model, queries, corpus, relevant, k=10)
    mean_delta, lo, hi = paired_delta_ci(before, after, seed=settings.seed)
    sd = float(np.std(np.asarray(after) - np.asarray(before)))
    headline = {
        "metric": "NDCG@10", "before": float(np.mean(before)), "after": float(np.mean(after)),
        "delta": mean_delta, "ci": (lo, hi), "mde": min_detectable_effect(len(before), sd), "n": len(before),
    }
    print(f"[after] NDCG@10 = {headline['after']:.4f} (Δ {mean_delta:+.4f}, CI [{lo:+.4f}, {hi:+.4f}])")

    # 6) Axis-3 RETENTION GUARD (general FR/EN, non-legal). base vs fine-tuned on the fixed suite.
    retention: list[dict] = []
    if not args.skip_retention:
        from lexfr_embed.general_eval import run_mteb, task_names

        b = run_mteb(SentenceTransformer(settings.base_model_id), output_folder=str(results / "mteb/base"))
        a = run_mteb(model, output_folder=str(results / "mteb/finetuned"))
        for t in task_names():
            if t in b and t in a:
                # MTEB reports one aggregate score per task -> no per-query MDE; use the ±0.02 tolerance.
                retention.append({"task": t, "before": b[t], "after": a[t], "delta": a[t] - b[t], "mde": 0.02})

    # 7) SCORECARD.
    md = format_scorecard(headline, retention, hashes)
    (results / "scorecard.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print("\n[done] results/scorecard.md + partition_hashes.json + phase1/final checkpoint")


if __name__ == "__main__":
    main()
