#!/usr/bin/env bash
# Presentation eval batch — triangulate 3 models across 4 splits, plus reranker counter-evidence.
# Splits: test/traintest (BSARD, external transfer) | trackb (leak-free LegalKit, in-distribution, powered)
#         | tax (louisbrulenaudet tax-retrieval, external FR professional; NB Lemone home-turf).
# Inference only, no training. Resumable: skips a run whose --out JSON already exists.
set -uo pipefail
cd /home/gdelabie/code/lexfr-embed
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p results/eval_extra
LOG=results/eval_extra/batch.log

run() {  # run <out_json> <args...>
  local out="$1"; shift
  if [ -s "$out" ]; then echo "SKIP (exists): $out" | tee -a "$LOG"; return; fi
  echo "=== $(date -u +%H:%M:%S) $* --out $out ===" | tee -a "$LOG"
  uv run --no-sync python scripts/eval_extra.py "$@" --out "$out" 2>&1 | tee -a "$LOG"
}

BASE="bge-m3"; FT="results/phase1/final"; LEMONE="louisbrulenaudet/lemone-embed-pro"

# 1) powered: NDCG@10 + Recall + bootstrap CI + MDE, all models x all splits
for M in "$BASE" "$FT" "$LEMONE"; do
  run "results/eval_extra/powered_$(basename "$M").json" --mode powered --model "$M" --splits test,traintest,trackb,tax
done

# 2) rerank counter-evidence on the powered new evals (+ test) — base & fine-tuned only
for M in "$BASE" "$FT"; do
  for S in trackb tax test; do
    run "results/eval_extra/rerank_$(basename "$M")_${S}.json" --mode rerank --model "$M" --split "$S"
  done
done

# 3) matryoshka dim-curve on a POWERED eval (trackb) — base & fine-tuned
for M in "$BASE" "$FT"; do
  run "results/eval_extra/matryoshka_$(basename "$M").json" --mode matryoshka --model "$M" --split trackb
done

echo "=== $(date -u +%H:%M:%S) BATCH DONE ===" | tee -a "$LOG"
