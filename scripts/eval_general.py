"""General-capability retention check — before vs after legal fine-tuning.

Loads the base model and the fine-tuned model, runs the SAME small non-legal
MTEB(fr) + BEIR subset on both, and prints + writes the delta table with a PASS/FAIL
verdict. Phase-1 guard against catastrophic forgetting (see docs/eval-set-spec.md
"General-capability retention"). Exit code is non-zero on FAIL so it can gate a
training pipeline / CI step.

Usage:
  uv run --extra eval python scripts/eval_general.py \
      --base BAAI/bge-m3 \
      --finetuned results/bge-m3-lora \
      --out results/general_retention.md --batch-size 32
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    from sentence_transformers import SentenceTransformer

    from lexfr_embed.general_eval import (
        DEFAULT_TOLERANCE,
        build_report,
        format_markdown,
        missing_tasks,
        run_mteb,
    )

    ap = argparse.ArgumentParser(description="General-capability retention check (catastrophic-forgetting guard).")
    ap.add_argument("--base", required=True, help="Base model id/path (the 'before' model).")
    ap.add_argument("--finetuned", required=True, help="Fine-tuned model id/path (the 'after' model).")
    ap.add_argument("--out", default="results/general_retention.md", help="Markdown report output path.")
    ap.add_argument("--results-dir", default="results/mteb", help="Where MTEB writes its raw per-task JSON.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Max allowed absolute drop per task.")
    args = ap.parse_args()

    print(f"[1/2] base      = {args.base}")
    before = run_mteb(
        SentenceTransformer(args.base), output_folder=f"{args.results_dir}/before", batch_size=args.batch_size
    )
    print(f"[2/2] finetuned = {args.finetuned}")
    after = run_mteb(
        SentenceTransformer(args.finetuned), output_folder=f"{args.results_dir}/after", batch_size=args.batch_size
    )

    rows, passed = build_report(before, after, tolerance=args.tolerance)
    md = format_markdown(rows, passed, tolerance=args.tolerance)
    print("\n" + md)

    missing = missing_tasks(before, after)
    if missing:
        print(f"\n⚠️  Not scored (failed to run in one/both passes): {', '.join(missing)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
