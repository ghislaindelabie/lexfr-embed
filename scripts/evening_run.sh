#!/usr/bin/env bash
# Evening pipeline — BLOCKS on the GPU flock until day_run.sh releases it, then evaluates
# every available checkpoint on Track-B v2 (the lexical-hard eval). First dense pass on v2:
# also confirms the forensics agent's projected base score (~0.60-0.80).
set -uo pipefail
cd /home/gdelabie/code/lexfr-embed
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p results/day
LOG=results/day/evening.log
say() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* ===" | tee -a "$LOG"; }

say "waiting for GPU lock (day_run still owns it)"
exec 9>/tmp/lexfr-gpu.lock
flock 9   # blocking — wakes exactly when day_run exits
say "lock acquired — starting trackb2 evals"

for spec in "bge-m3:base" "results/phase1/final:final" \
            "results/day/ck_A1off/final:A1off" "results/day/ck_A1on/final:A1on"; do
  M="${spec%%:*}"; TAG="${spec##*:}"
  OUT="results/day/trackb2_${TAG}.json"
  [ -s "$OUT" ] && { say "SKIP $TAG"; continue; }
  if [ "$M" != "bge-m3" ] && [ ! -d "$M" ]; then say "MISSING checkpoint $M — skipping $TAG"; continue; fi
  say "EVAL trackb2 $TAG"
  uv run --no-sync python scripts/eval_extra.py --mode powered --model "$M" \
    --splits trackb2 --out "$OUT" 2>&1 | tee -a "$LOG"
done
say "EVENING DONE"
