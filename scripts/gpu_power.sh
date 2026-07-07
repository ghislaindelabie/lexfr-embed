#!/usr/bin/env bash
# Per-job GPU power hook for the P710 RTX 5060 Ti.
#
# The LAB DEFAULT is 150 W (systemd gpu-power-cap.service + persistence mode). Rationale:
# our workloads are memory-bandwidth-bound, so capping core power 180->150 W costs ~1-3%
# throughput for -14% power and ~5 C cooler/quieter (measured 2026-07-06). Memory-bound jobs
# therefore need NO wrapper — they just run at 150 W.
#
# Use this wrapper ONLY for a job you've MEASURED to be compute(core-clock)-bound, to lift the
# cap for that job's duration; the 150 W default is restored on exit, even on crash/kill (trap).
#
#   Usage:   scripts/gpu_power.sh <watts> -- <command...>
#   Example: scripts/gpu_power.sh 180 -- uv run --no-sync python scripts/big_batch_encode.py
#   (valid range on this card: 150-180 W)
set -uo pipefail
DEFAULT_W=150
watts="${1:?usage: gpu_power.sh <watts> -- <command...>}"
shift
[ "${1:-}" = "--" ] && shift
[ "$#" -ge 1 ] || {
  echo "usage: gpu_power.sh <watts> -- <command...>" >&2
  exit 2
}
restore() { sudo -n nvidia-smi -pl "$DEFAULT_W" >/dev/null 2>&1 || true; }
trap restore EXIT INT TERM
sudo -n nvidia-smi -pl "$watts" >/dev/null 2>&1 || echo "gpu_power: warn: could not set ${watts}W" >&2
echo "gpu_power: cap=${watts}W for [$*] — restores ${DEFAULT_W}W on exit" >&2
"$@"
