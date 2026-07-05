"""Background GPU telemetry logger for training runs.

W&B's auto system metrics cover GPU util/mem/power/temp, but NOT throttle reasons, pstate,
clock speeds, fan, or Xid errors. This poller fills the gap: it samples `nvidia-smi` every
`--interval` seconds and writes both a CSV (for offline reports) and, if `--project` is given,
its own W&B run (so the data lives alongside the training runs). Meant to run in the background
during training; stop with SIGTERM/SIGINT (it finalizes the CSV and the W&B run).

    uv run python scripts/gpu_telemetry.py --interval 5 --project lexfr-embed --name gpu-tele-minilm

Robustness: probes throttle-reason field names (renamed `clocks_throttle_reasons.*` ->
`clocks_event_reasons.*` on driver R535+); drops any field the driver rejects; tolerates a
restricted dmesg (Xid capture then silently skipped). GeForce cards report no ECC (expected).
"""

from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import time
from pathlib import Path

CORE = [
    "temperature.gpu",
    "fan.speed",
    "power.draw",
    "power.limit",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "clocks.sm",
    "clocks.mem",
    "pstate",
]
THROTTLE = ["sw_power_cap", "hw_slowdown", "hw_thermal_slowdown", "sw_thermal_slowdown"]

_stop = False


def _handle(_sig, _frame):
    global _stop
    _stop = True


def _query_ok(fields: list[str]) -> bool:
    r = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def _pick_throttle_prefix() -> str | None:
    for prefix in ("clocks_event_reasons", "clocks_throttle_reasons"):
        if _query_ok([f"{prefix}.{THROTTLE[0]}"]):
            return prefix
    return None


def _num(v: str):
    v = v.strip()
    if v in ("[N/A]", "[Not Supported]", "N/A", ""):
        return None
    if v.startswith("P") and v[1:].isdigit():  # pstate P0..P12 -> int
        return int(v[1:])
    if v in ("Active", "Not Active"):
        return 1 if v == "Active" else 0
    try:
        return float(v)
    except ValueError:
        return v


def _xid_count() -> int | None:
    r = subprocess.run(["dmesg"], capture_output=True, text=True)
    if r.returncode != 0:  # dmesg_restrict=1 without sudo -> skip silently
        return None
    return sum(1 for line in r.stdout.splitlines() if "NVRM: Xid" in line)


def main() -> None:
    ap = argparse.ArgumentParser(description="Background nvidia-smi -> CSV (+ W&B) telemetry logger.")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--csv", default="results/gpu_telemetry.csv")
    ap.add_argument("--project", default=None, help="W&B project; omit to log CSV only")
    ap.add_argument("--name", default="gpu-telemetry")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    prefix = _pick_throttle_prefix()
    throttle_fields = [f"{prefix}.{t}" for t in THROTTLE] if prefix else []
    query = CORE + throttle_fields
    cols = ["ts", *[c.replace(f"{prefix}.", "throttle.") if prefix else c for c in query], "xid_count"]

    run = None
    if args.project:
        try:
            import wandb

            run = wandb.init(project=args.project, name=args.name, job_type="telemetry")
            run.define_metric("gpu/*", step_metric="gpu/_ts")
        except Exception as e:  # noqa: BLE001 - W&B is optional; never let it kill CSV logging
            print(f"[telemetry] W&B disabled ({e}); CSV only")
            run = None

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        print(
            f"[telemetry] logging {query} every {args.interval}s -> {args.csv}"
            + (f" + W&B {args.project}/{args.name}" if run else "")
        )
        while not _stop:
            r = subprocess.run(
                ["nvidia-smi", f"--query-gpu={','.join(query)}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            ts = time.time()
            if r.returncode == 0 and r.stdout.strip():
                raw = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
                vals = [_num(x) for x in raw]
                xid = _xid_count()
                writer.writerow([ts, *vals, xid])
                fh.flush()
                if run:
                    payload = {"gpu/_ts": ts, "gpu/xid_count": xid}
                    for col, val in zip(cols[1:-1], vals, strict=False):
                        if isinstance(val, (int, float)):
                            payload[f"gpu/{col}"] = val
                    run.log(payload)
            slept = 0.0
            while slept < args.interval and not _stop:  # responsive to SIGTERM
                time.sleep(0.5)
                slept += 0.5

    if run:
        run.finish()
    print(f"[telemetry] stopped; CSV at {args.csv}")


if __name__ == "__main__":
    main()
