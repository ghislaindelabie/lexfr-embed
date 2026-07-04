"""Unattended experiment-campaign orchestrator for lexfr-embed (local RTX 5060 Ti).

Runs an ordered queue (scripts/campaign_queue.json) of run_phase1.py experiments, one per fresh
SUBPROCESS (a try/except cannot catch a CUDA hard error / segfault / wedged-allocator OOM; the
process boundary also frees VRAM between runs). Each experiment: isolated results dir, GPU
telemetry logger alongside, per-run timeout, classify (ok/oom/crash/fail/timeout), scorecard.json
+ gpu.csv aggregated into an append-only ledger (jsonl + rebuilt csv) and a W&B summary run in a
DEDICATED project. Resumable (skip experiments already ok in the ledger), continue-on-failure,
checkpoint cleanup (disk guard), NF-42 reproduction HALT-gate, and a stop-starting clock so it
never launches a run that can't finish inside the wall-clock budget.

    uv run --no-sync python scripts/run_campaign.py            # run the queue
    uv run --no-sync python scripts/run_campaign.py --smoke    # 1 tiny CPU 'smoke' exp: prove the harness

Env required: WANDB_API_KEY (else runs log nothing). Everything else is set per-experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QUEUE = REPO / "scripts" / "campaign_queue.json"
CAMP_ROOT = REPO / "results" / "campaign"
LEDGER_JSONL = CAMP_ROOT / "ledger.jsonl"
LEDGER_CSV = CAMP_ROOT / "ledger.csv"
ENV_LOCK = CAMP_ROOT / "env_lock.json"

CAMPAIGN_ID = "sweep-20260705"
WANDB_PROJECT = "lexfr-embed-sweeps"  # DEDICATED project (not the main lexfr-embed)
WALLCLOCK_BUDGET_MIN = 1440  # 24h; never launch a run that can't finish inside this
LAUNCH_SLACK_MIN = 15  # stop-starting-clock margin
BASELINE_AFTER = 0.282  # NF-42 reproduction target
NF_GATE_TOL = 0.05  # HALT if NF-42 |after-0.282| exceeds this (gross breakage, not seed jitter)
MAX_ATTEMPTS = 2  # 1 retry on terminal failure

# Forced LEXFR_EMBED_* baseline (CachedMNRL is mandatory — plain MNRL OOMs on 16 GB). Everything
# else uses config.py defaults unless an arm overrides it.
BASE_LEXFR = {"LEXFR_EMBED_USE_CACHED_MNRL": "true"}
BASE_ENV = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "WANDB_PROJECT": WANDB_PROJECT,
    "WANDB_MODE": "online",
}

LEDGER_COLS = [
    "exp_id",
    "campaign_id",
    "attempt",
    "group",
    "arm",
    "status",
    "git_sha",
    "git_dirty",
    "config_hash",
    "base_model_key",
    "subset",
    "n_legal_pairs",
    "rehearsal_frac",
    "n_rehearsal",
    "epochs_s1",
    "epochs_s2",
    "batch_size",
    "mini_batch_size",
    "max_seq_len",
    "lr_lora",
    "lr_full_ft",
    "lora_r",
    "lora_alpha",
    "hard_neg_relative_margin",
    "num_negatives",
    "use_cached_mnrl",
    "matryoshka_dims",
    "seed",
    "ndcg_before",
    "ndcg_after",
    "delta",
    "ci_lo",
    "ci_hi",
    "excludes_zero",
    "mde",
    "within_noise",
    "n_queries",
    "retention_ran",
    "retention_pass",
    "retention_worst_delta",
    "duration_s",
    "peak_vram_mb",
    "avg_power_w",
    "peak_power_w",
    "avg_temp_c",
    "peak_temp_c",
    "throttle_thermal",
    "throttle_power_cap",
    "xid_errors",
    "partition_hash_gold",
    "partition_hash_corpus",
    "wandb_group",
    "error",
    "started_at",
    "ended_at",
]


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, cwd=REPO).stdout.strip()


def git_sha() -> str:
    return sh("git", "rev-parse", "HEAD")


def git_dirty() -> bool:
    return bool(sh("git", "status", "--porcelain"))


def config_hash(arm: dict) -> str:
    payload = {
        "lexfr": {**BASE_LEXFR, **arm.get("env", {})},
        "subset": arm.get("subset"),
        "skip_retention": arm.get("skip_retention", True),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def read_ledger() -> list[dict]:
    if not LEDGER_JSONL.exists():
        return []
    return [json.loads(line) for line in LEDGER_JSONL.read_text().splitlines() if line.strip()]


def append_ledger(row: dict) -> None:
    with open(LEDGER_JSONL, "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    rows = read_ledger()
    with open(LEDGER_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate_gpu(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    pw, tp, vram, thermal, pcap, xid = [], [], [], 0, 0, 0
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            for lst, key in ((pw, "power.draw"), (tp, "temperature.gpu"), (vram, "memory.used")):
                try:
                    lst.append(float(r[key]))
                except (KeyError, ValueError, TypeError):
                    pass
            if r.get("throttle.hw_thermal_slowdown") in ("1", "1.0"):
                thermal += 1
            if r.get("throttle.sw_power_cap") in ("1", "1.0"):
                pcap += 1
            try:
                xid = max(xid, int(float(r.get("xid_count") or 0)))
            except (ValueError, TypeError):
                pass
    return {
        "peak_vram_mb": max(vram) if vram else None,
        "avg_power_w": round(sum(pw) / len(pw), 1) if pw else None,
        "peak_power_w": max(pw) if pw else None,
        "avg_temp_c": round(sum(tp) / len(tp), 1) if tp else None,
        "peak_temp_c": max(tp) if tp else None,
        "throttle_thermal": thermal,
        "throttle_power_cap": pcap,
        "xid_errors": xid,
    }


def classify(rc: int | None, timed_out: bool, scorecard: Path, run_log: Path) -> str:
    if timed_out:
        return "timeout"
    log = run_log.read_text(errors="ignore") if run_log.exists() else ""
    if "out of memory" in log.lower() or "OutOfMemoryError" in log:
        return "oom"
    if rc == 0 and scorecard.exists():
        return "ok"
    if rc == 0:
        return "fail"  # exit 0 but no scorecard
    return "crash"


def resolved_settings(exp_dir: Path, arm: dict) -> dict:
    """Read the child's actual settings (env applied) so the ledger records the true config."""
    env = {**os.environ, **BASE_ENV, **BASE_LEXFR, **arm.get("env", {}), "LEXFR_EMBED_RESULTS_DIR": str(exp_dir)}
    out = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-c",
            "import json;from lexfr_embed.config import settings;print(json.dumps(settings.model_dump(),default=str))",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
    ).stdout.strip()
    try:
        return json.loads(out.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {}


def wandb_summary(exp_id: str, cfg: dict, arm: dict, metrics: dict) -> None:
    try:
        import wandb

        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"{exp_id}-summary",
            group=exp_id,
            job_type="summary",
            tags=[CAMPAIGN_ID, f"arm={arm['id']}", f"group={arm['group']}"],
            config={**cfg, "arm_id": arm["id"], "hypothesis": arm.get("hypothesis", "")},
            reinit=True,
        )
        run.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        run.summary.update({k: v for k, v in metrics.items() if v is not None})
        run.finish()
    except Exception as e:  # noqa: BLE001 - never let W&B kill the campaign
        print(f"[campaign] W&B summary failed for {exp_id}: {e}")


def cleanup_checkpoints(ledger: list[dict]) -> None:
    """Disk guard: keep phase1/final only for the current best ndcg_after; prune stage1 + others."""
    ok = [r for r in ledger if r.get("status") == "ok" and r.get("ndcg_after") is not None]
    best = max(ok, key=lambda r: r["ndcg_after"])["exp_id"] if ok else None
    keep_final = {best} | {r["exp_id"] for r in ledger if r.get("group") in ("noise-floor", "confirmatory")}
    for d in CAMP_ROOT.glob("*/phase1"):
        exp_id = d.parent.name
        shutil.rmtree(d / "stage1", ignore_errors=True)
        if exp_id not in keep_final:
            shutil.rmtree(d / "final", ignore_errors=True)
        # MTEB result dirs are tiny json; leave them.


def run_one(arm: dict, idx: int, ledger: list[dict], smoke: bool) -> dict:
    h = config_hash(arm)
    exp_id = f"{CAMPAIGN_ID}-{idx:02d}-{arm['id']}-{h}"
    exp_dir = CAMP_ROOT / exp_id
    subset = arm.get("subset")
    skip_ret = arm.get("skip_retention", True)
    est = arm.get("est_minutes", 60)

    prior = [r for r in ledger if r["exp_id"] == exp_id]
    if any(r.get("status") == "ok" for r in prior):
        print(f"[campaign] SKIP {exp_id} (already ok)")
        return next(r for r in prior if r.get("status") == "ok")
    attempt = len(prior) + 1
    if attempt > MAX_ATTEMPTS:
        print(f"[campaign] GIVE UP {exp_id} (retries exhausted)")
        return {
            "exp_id": exp_id,
            "campaign_id": CAMPAIGN_ID,
            "attempt": attempt,
            "group": arm["group"],
            "arm": arm["id"],
            "status": "skipped",
            "error": "exhausted retries",
            "wandb_group": exp_id,
        }

    if exp_dir.exists():
        shutil.rmtree(exp_dir, ignore_errors=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        **BASE_ENV,
        **BASE_LEXFR,
        **arm.get("env", {}),
        "LEXFR_EMBED_RESULTS_DIR": str(exp_dir),
        "WANDB_RUN_GROUP": exp_id,
        "WANDB_NAME": exp_id,
        "WANDB_TAGS": f"{CAMPAIGN_ID},arm={arm['id']},group={arm['group']}",
    }

    # telemetry alongside (same env -> inherits WANDB_RUN_GROUP)
    gpu_csv = exp_dir / "gpu.csv"
    tele = subprocess.Popen(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "scripts/gpu_telemetry.py",
            "--interval",
            "5",
            "--csv",
            str(gpu_csv),
            "--project",
            WANDB_PROJECT,
            "--name",
            f"{exp_id}-tele",
        ],
        cwd=REPO,
        env=env,
        stdout=open(exp_dir / "tele.log", "w"),
        stderr=subprocess.STDOUT,
    )

    cmd = ["uv", "run", "--no-sync", "python"]
    if smoke:
        cmd += [
            "-c",
            "from lexfr_embed.train import train_embedder;"
            "train_embedder(base_model_key='smoke', use_lora=False, max_steps=1,"
            "train_pairs=[{'anchor':f'q{i}','positive':f'a{i}','code':'civil'} for i in range(24)],"
            f"out_dir='{exp_dir}/phase1'); open('{exp_dir}/scorecard.json','w').write('{{}}')",
        ]
    else:
        cmd += ["scripts/run_phase1.py"]
        if subset:
            cmd += ["--subset", str(subset)]
        if skip_ret:
            cmd += ["--skip-retention"]

    timeout_s = max(90, int(2 * est)) * 60
    started = time.time()
    run_log = exp_dir / "run.log"
    timed_out = False
    with open(run_log, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(30)
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    rc = proc.returncode
    tele.send_signal(signal.SIGTERM)
    try:
        tele.wait(timeout=20)
    except subprocess.TimeoutExpired:
        tele.kill()
    duration = round(time.time() - started, 1)

    scorecard = exp_dir / "scorecard.json"
    status = classify(rc, timed_out, scorecard, run_log)

    row = {
        "exp_id": exp_id,
        "campaign_id": CAMPAIGN_ID,
        "attempt": attempt,
        "group": arm["group"],
        "arm": arm["id"],
        "status": status,
        "git_sha": GIT_SHA,
        "git_dirty": GIT_DIRTY,
        "config_hash": h,
        "subset": subset,
        "duration_s": duration,
        "wandb_group": exp_id,
        "started_at": started,
        "ended_at": time.time(),
        "error": "" if status == "ok" else status,
    }

    cfg = resolved_settings(exp_dir, arm) if not smoke else {}
    for k_led, k_cfg in {
        "base_model_key": "base_model_key",
        "rehearsal_frac": "rehearsal_frac",
        "epochs_s1": "epochs_stage1",
        "epochs_s2": "epochs_stage2",
        "batch_size": "batch_size",
        "mini_batch_size": "mini_batch_size",
        "max_seq_len": "max_seq_len",
        "lr_lora": "lr_lora",
        "lr_full_ft": "lr_full_ft",
        "lora_r": "lora_r",
        "lora_alpha": "lora_alpha",
        "hard_neg_relative_margin": "hard_neg_relative_margin",
        "num_negatives": "num_negatives",
        "use_cached_mnrl": "use_cached_mnrl",
        "matryoshka_dims": "matryoshka_dims",
        "seed": "seed",
    }.items():
        row[k_led] = cfg.get(k_cfg)

    if scorecard.exists() and status == "ok":
        try:
            sc = json.loads(scorecard.read_text())
            hl = sc.get("headline", {})
            row.update(
                {
                    "ndcg_before": hl.get("before"),
                    "ndcg_after": hl.get("after"),
                    "delta": hl.get("delta"),
                    "ci_lo": hl.get("ci_lo"),
                    "ci_hi": hl.get("ci_hi"),
                    "excludes_zero": hl.get("excludes_zero"),
                    "mde": hl.get("mde"),
                    "within_noise": hl.get("within_noise"),
                    "n_queries": hl.get("n"),
                }
            )
            ret = sc.get("retention") or []
            row["retention_ran"] = bool(ret)
            if ret:
                worst = min(r["delta"] for r in ret)
                row["retention_worst_delta"] = worst
                row["retention_pass"] = all(r["delta"] >= -r.get("mde", 0.02) for r in ret)
            ph = sc.get("partition_hashes", {})
            row["partition_hash_gold"] = ph.get("bsard_gold")
            row["partition_hash_corpus"] = ph.get("bsard_corpus")
        except Exception as e:  # noqa: BLE001
            row["error"] = f"scorecard parse: {e}"
    # n_legal_pairs / n_rehearsal from run.log
    for line in run_log.read_text(errors="ignore").splitlines():
        if line.startswith("[data]"):
            row["n_legal_pairs"] = int(line.split()[1])
        if line.startswith("[rehearsal] target"):
            try:
                row["n_rehearsal"] = int(line.split("loaded")[1].strip())
            except (IndexError, ValueError):
                pass
    row.update(aggregate_gpu(gpu_csv))

    print(
        f"[campaign] {exp_id}: {status} | after={row.get('ndcg_after')} | {duration / 60:.0f}min "
        f"| peakVRAM={row.get('peak_vram_mb')}MB peakT={row.get('peak_temp_c')}C"
    )
    wandb_summary(exp_id, cfg, arm, row)
    return row


GIT_SHA = ""
GIT_DIRTY = False


def main() -> None:
    global GIT_SHA, GIT_DIRTY
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run one tiny CPU smoke experiment and exit")
    args = ap.parse_args()

    CAMP_ROOT.mkdir(parents=True, exist_ok=True)
    GIT_SHA, GIT_DIRTY = git_sha(), git_dirty()

    if args.smoke:
        arm = {
            "id": "SMOKE",
            "group": "smoke",
            "env": {"LEXFR_EMBED_SEED": "42"},
            "subset": None,
            "skip_retention": True,
            "est_minutes": 5,
            "hypothesis": "harness isolation + ledger + scorecard.json",
        }
        row = run_one(arm, 0, read_ledger(), smoke=True)
        append_ledger(row)
        print(f"[campaign] SMOKE status={row['status']} — ledger at {LEDGER_CSV}")
        return

    if GIT_DIRTY:
        print("[campaign] REFUSING to start on a dirty tree — commit first (reproducibility).")
        sys.exit(1)

    queue = json.loads(QUEUE.read_text())
    ENV_LOCK.write_text(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "git_sha": GIT_SHA,
                "started_at": time.time(),
                "torch": sh("uv", "run", "--no-sync", "python", "-c", "import torch;print(torch.__version__)"),
                "queue_len": len(queue),
                "wandb_project": WANDB_PROJECT,
            },
            indent=2,
        )
    )
    print(f"[campaign] START {CAMPAIGN_ID} | {len(queue)} exps | budget {WALLCLOCK_BUDGET_MIN}min | sha {GIT_SHA[:8]}")

    campaign_start = time.time()
    for idx, arm in enumerate(queue):
        elapsed_min = (time.time() - campaign_start) / 60
        est = arm.get("est_minutes", 60)
        if elapsed_min + est + LAUNCH_SLACK_MIN > WALLCLOCK_BUDGET_MIN:
            print(
                f"[campaign] STOP-CLOCK: {elapsed_min:.0f}min elapsed + {est}min est would exceed budget; "
                f"skipping remaining {len(queue) - idx} experiments (droppable tail)."
            )
            break
        row = run_one(arm, idx, read_ledger(), smoke=False)
        append_ledger(row)
        cleanup_checkpoints(read_ledger())

        # NF-42 reproduction HALT-gate: protect the 24h from a broken rig.
        if arm["id"] == "NF-42":
            a = row.get("ndcg_after")
            ok = row.get("status") == "ok" and a is not None and abs(a - BASELINE_AFTER) <= NF_GATE_TOL
            if not ok:
                print(
                    f"[campaign] HALT: NF-42 did not reproduce ~{BASELINE_AFTER} "
                    f"(after={a}, status={row.get('status')}) — not burning the budget on a broken rig."
                )
                break

    print(f"[campaign] DONE {CAMPAIGN_ID} after {(time.time() - campaign_start) / 3600:.1f}h — ledger {LEDGER_CSV}")


if __name__ == "__main__":
    main()
