#!/usr/bin/env bash
# Unattended overnight GPU pipeline. STRICTLY SEQUENTIAL — never two GPU jobs at once
# (org rate limits + 16 GB card). It first WAITS for the running eval_batch to finish
# (gates on its "BATCH DONE" marker, then re-confirms the GPU is free), then:
#   A1 experiment  — train denoise=OFF vs ON @30k/seed42, eval each on the POWERED evals
#                    (Track-B MDE~0.017 can finally DETECT whether denoised negatives help)
#   REH resume     — finish the rehearsal retention curve (REH-07/REH-15) via the orchestrator
# Resume-safe: every step skips if its .done/-.json output already exists. All logged;
# GPU telemetry (util/mem/power/temp) sampled every 30 s to results/overnight/gpu_telemetry.csv.
set -uo pipefail
cd /home/gdelabie/code/lexfr-embed
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_PROJECT=lexfr-embed-sweeps
mkdir -p results/overnight
LOG=results/overnight/overnight.log
say() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* ===" | tee -a "$LOG"; }

# --- GPU telemetry logger (background nvidia-smi loop; killed on exit) ---
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv -l 30 > results/overnight/gpu_telemetry.csv 2>/dev/null &
GPUMON=$!
trap 'kill $GPUMON 2>/dev/null' EXIT

wait_for_batch() {  # gate on the running eval_batch's completion marker (~3h cap)
  say "waiting for eval_batch 'BATCH DONE' marker"
  for _ in $(seq 1 720); do
    grep -q "BATCH DONE" results/eval_extra/batch.log 2>/dev/null && { say "eval_batch finished"; return 0; }
    sleep 15
  done
  say "WARN: eval_batch marker not seen after ~3h — will gate on GPU-free instead"
}

wait_gpu_free() {  # block until no compute apps AND <800 MiB (belt-and-suspenders, ~1h cap)
  for i in $(seq 1 240); do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ "$apps" -eq 0 ] && [ "${used:-9999}" -lt 800 ] && { say "GPU free (${used}MiB, iter $i)"; return 0; }
    sleep 15
  done
  say "WARN: GPU still busy after ~1h — proceeding anyway"
}

run_a1_arm() {  # <tag> <denoise:true|false>
  local tag="$1" den="$2" ck="results/overnight/ck_$1"
  if [ -f "results/overnight/${tag}.done" ]; then say "SKIP A1 $tag (done)"; return 0; fi
  wait_gpu_free
  say "TRAIN A1 $tag (denoise=$den, subset=30000, seed=42)"
  LEXFR_EMBED_SEED=42 LEXFR_EMBED_DENOISE_NEGATIVES="$den" LEXFR_EMBED_USE_CACHED_MNRL=true \
  WANDB_NAME="A1-$tag" \
    uv run --no-sync python scripts/run_phase1.py --subset 30000 --skip-retention --out-dir "$ck" 2>&1 | tee -a "$LOG"
  if [ ! -d "$ck/final" ]; then say "ERROR A1 $tag: no checkpoint at $ck/final — skipping eval"; return 1; fi
  say "EVAL A1 $tag: powered (test,trackb,tax)"
  uv run --no-sync python scripts/eval_extra.py --mode powered --model "$ck/final" \
    --splits test,trackb,tax --out "results/overnight/powered_${tag}.json" 2>&1 | tee -a "$LOG"
  say "EVAL A1 $tag: rerank (trackb)"
  uv run --no-sync python scripts/eval_extra.py --mode rerank --model "$ck/final" \
    --split trackb --out "results/overnight/rerank_${tag}_trackb.json" 2>&1 | tee -a "$LOG"
  rm -rf "$ck/stage1" "$ck"/checkpoint-* 2>/dev/null  # keep only the LoRA final adapter
  touch "results/overnight/${tag}.done"
  say "DONE A1 $tag"
}

say "OVERNIGHT START"
wait_for_batch          # let the triangulation batch finish
wait_gpu_free

# --- A1: does DENOISED hard-negative mining beat plain mining? (measured on a powered eval) ---
run_a1_arm A1off false  # matched control: same 30k/seed42, no cross-encoder denoising
run_a1_arm A1on  true   # A1: cross-encoder (bge-reranker-v2-m3) drops false negatives

# --- REH: resume the retention curve (orchestrator re-runs only not-ok arms from the ledger) ---
if [ ! -f "results/overnight/reh.done" ]; then
  wait_gpu_free
  say "RESUME REH via run_campaign.py (resumes REH-07/REH-15)"
  uv run --no-sync python scripts/run_campaign.py 2>&1 | tee -a "$LOG"
  touch "results/overnight/reh.done"
fi

# --- morning summary: pull the key numbers from both logs into one file ---
say "writing SUMMARY.txt"
{
  echo "# Overnight summary  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo "## Triangulation batch (base / fine-tuned / Lemone x test/traintest/trackb/tax)"
  grep -hE "\[powered\]|\[rerank\]|\[matryoshka\]" results/eval_extra/batch.log 2>/dev/null | grep -v "Loading weights"
  echo "## A1 experiment (denoise OFF vs ON @30k/seed42)"
  grep -hE "\[powered\]|\[rerank\]" results/overnight/overnight.log 2>/dev/null | grep -v "Loading weights"
} > results/overnight/SUMMARY.txt 2>/dev/null

say "OVERNIGHT DONE"
