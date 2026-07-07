#!/usr/bin/env bash
# Matched micro-bench at the current power cap (150 W experiment), vs the 24.3 samples/s @ 180 W
# baseline. Blocks on the GPU lock so it never contends with the day/evening pipelines.
set -uo pipefail
cd /home/gdelabie/code/lexfr-embed
export HF_HUB_DISABLE_TELEMETRY=1
LOG=results/day/microbench_150w.log
say() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* ===" | tee -a "$LOG"; }
exec 9>/tmp/lexfr-gpu.lock
say "waiting for GPU lock (day + evening runners ahead)"
flock 9
say "lock acquired; enforced power.limit=$(nvidia-smi --query-gpu=enforced.power.limit --format=csv,noheader)"
uv run --no-sync python scripts/gpu_microbench.py 2>&1 | grep -E "BENCH" | tee -a "$LOG"
# also sample the actual power drawn during a second short run for samples/s-per-watt
say "DONE (compare samples_per_s to 24.34 @ 180 W)"
