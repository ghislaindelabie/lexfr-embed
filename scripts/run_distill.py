"""A1-bis graded run — distill the teacher reranker into the fine-tuned embedder, judged on the
recall curve (PROJECT_LOG A1-bis, criterion LOCKED).

Order (mirrors run_phase1.py's honest before/after + paired CI): assert the OFFLINE teacher cache is
present (else point to scripts/build_distill_cache.py) -> pre-distill recall curve BEFORE on the
fine-tuned checkpoint -> `train.distill_embedder` (MarginMSE ± Matryoshka, reads only the cache) ->
recall curve AFTER on Track-B v2 + BSARD with a PAIRED Δhit@5 vs BEFORE -> scorecard JSON.

Success is judged on the curve, not NDCG: the distilled student's hit@5 lifts toward the pre-distill
hit@50 (a shallower reranker gate suffices) and the paired Δhit@5 CI on Track-B v2 excludes zero.
Literature (Tamber 2025) predicts a PARTIAL close — a within-noise Δ is reported honestly, never
overclaimed (like the A1 null). Run eval_general separately for the general-language retention guard.

    uv run --extra eval python scripts/run_distill.py                 # after build_distill_cache.py
    uv run --extra eval python scripts/run_distill.py --splits trackb2,test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    import sys

    from lexfr_embed.config import settings
    from lexfr_embed.train import distill_embedder

    # eval_extra lives in scripts/ (not the package) — put it on the path, then import as a sibling.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_extra import mode_recall_curve  # noqa: E402

    ap = argparse.ArgumentParser(description="A1-bis distillation graded run (recall-curve criterion).")
    ap.add_argument("--base-ckpt", default=settings.distill_miner_ckpt, help="pre-distill fine-tuned checkpoint")
    ap.add_argument(
        "--cache-dir", default=str(settings.distill_cache_dir), help="teacher cache (build_distill_cache.py)"
    )
    ap.add_argument("--out-dir", default=str(settings.results_dir / "distill"), help="distilled checkpoint output dir")
    ap.add_argument("--splits", default="trackb2,test", help="comma list; trackb2=powered Track-B v2, test=BSARD")
    ap.add_argument("--no-lora", action="store_true", help="full-FT distill instead of LoRA (default LoRA)")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if not (cache_dir / "meta.json").exists():
        raise SystemExit(
            f"[run_distill] no teacher cache at {cache_dir} — build it first:\n"
            "    uv run --no-sync python scripts/build_distill_cache.py"
        )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = args.splits.split(",")

    # 1) BEFORE — pre-distill recall curve on the fine-tuned checkpoint (the paired reference).
    before = mode_recall_curve(args.base_ckpt, splits)
    before_path = out / "recall_before.json"
    before_path.write_text(json.dumps({"result": before}, indent=2), encoding="utf-8")
    print(f"[run_distill] wrote {before_path}")

    # 2) DISTILL — MarginMSE(±Matryoshka), reads only the cache (reranker never co-resident).
    distill_embedder(
        base_ckpt=args.base_ckpt,
        cache_dir=str(cache_dir),
        out_dir=str(out),
        use_lora=not args.no_lora,
    )

    # 3) AFTER — recall curve on the distilled checkpoint + paired Δhit@5 vs BEFORE.
    after = mode_recall_curve(f"{out}/distill", splits, reference=str(before_path))

    scorecard = {
        "base_ckpt": args.base_ckpt,
        "cache_meta": json.loads((cache_dir / "meta.json").read_text(encoding="utf-8")),
        "splits": splits,
        "before": before,
        "after": after,
    }
    (out / "scorecard.json").write_text(json.dumps(scorecard, indent=2, default=float), encoding="utf-8")
    print(f"\n[run_distill] done — {out}/distill + scorecard.json")
    for split in splits:
        pd = after[split].get("paired_delta_hit5")
        if pd:
            verdict = "EXCLUDES 0" if pd["excludes_zero"] else "within noise"
            print(f"  {split:9} Δhit@5 = {pd['delta']:+.4f} CI[{pd['ci'][0]:+.4f},{pd['ci'][1]:+.4f}] ({verdict})")


if __name__ == "__main__":
    main()
