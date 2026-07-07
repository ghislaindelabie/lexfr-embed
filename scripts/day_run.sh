#!/usr/bin/env bash
# Day pipeline 2026-07-06 — ONE driver process owns the GPU (last night's A1 OOMs came from
# two independent pollers racing for it; a single sequential owner is the structural fix).
# flock is belt-and-suspenders against future concurrent sessions.
# Order (by information value):
#   1. paired evals   — base vs phase1/final on traintest/trackb/tax with per-query NDCG
#                       (enables PAIRED deltas: ~2x more sensitive than unpaired CIs)
#   2. A1off          — 30k/seed42/frac0.07 control (= CONF-1 replicate), checkpoint KEPT
#   3. A1on           — same + cross-encoder denoised mining (the A1 experiment)
#   4. REH resume     — retention curve arms (tree is clean now; dirty-gate passes)
# No rerank stages for A1 (trackb rerank costs ~2h and measured ~0 at the ceiling).
set -uo pipefail
cd /home/gdelabie/code/lexfr-embed
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_PROJECT=lexfr-embed-sweeps
mkdir -p results/day
LOG=results/day/day.log
say() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* ===" | tee -a "$LOG"; }

exec 9>/tmp/lexfr-gpu.lock
flock -n 9 || { say "FATAL: GPU lock held by another process — refusing to start"; exit 1; }

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv -l 30 > results/day/gpu_telemetry.csv 2>/dev/null &
GPUMON=$!
trap 'kill $GPUMON 2>/dev/null' EXIT

# --- 1) paired evals (skip if already done) ---
for spec in "bge-m3:base" "results/phase1/final:final"; do
  M="${spec%%:*}"; TAG="${spec##*:}"
  OUT="results/day/paired_${TAG}.json"
  [ -s "$OUT" ] && { say "SKIP paired $TAG"; continue; }
  say "PAIRED EVAL $TAG (traintest,trackb,tax)"
  uv run --no-sync python scripts/eval_extra.py --mode powered --model "$M" \
    --splits traintest,trackb,tax --out "$OUT" 2>&1 | tee -a "$LOG"
done

# --- 2+3) A1 experiment: matched pair, checkpoints kept ---
run_a1() {  # <tag> <denoise>
  local tag="$1" den="$2" ck="results/day/ck_$1"
  [ -f "results/day/${tag}.done" ] && { say "SKIP $tag"; return 0; }
  say "TRAIN $tag (denoise=$den, 30k, seed 42)"
  LEXFR_EMBED_SEED=42 LEXFR_EMBED_DENOISE_NEGATIVES="$den" LEXFR_EMBED_USE_CACHED_MNRL=true \
  WANDB_NAME="A1-$tag-30k-s42" \
    uv run --no-sync python scripts/run_phase1.py --subset 30000 --skip-retention --out-dir "$ck" 2>&1 | tee -a "$LOG"
  [ -d "$ck/final" ] || { say "ERROR $tag: no checkpoint — eval skipped"; return 1; }
  say "EVAL $tag (powered, per-query)"
  uv run --no-sync python scripts/eval_extra.py --mode powered --model "$ck/final" \
    --splits traintest,trackb,tax --out "results/day/powered_${tag}.json" 2>&1 | tee -a "$LOG"
  touch "results/day/${tag}.done"
  say "DONE $tag"
}
run_a1 A1off false
run_a1 A1on true

# --- 4) REH resume (clean tree -> dirty-gate passes; reruns only non-ok arms) ---
if [ ! -f results/day/reh.done ]; then
  say "RESUME REH (run_campaign)"
  uv run --no-sync python scripts/run_campaign.py 2>&1 | tee -a "$LOG"
  touch results/day/reh.done
fi

say "DAY RUN DONE"
