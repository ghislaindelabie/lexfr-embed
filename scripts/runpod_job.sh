#!/usr/bin/env bash
# RunPod fire-and-forget Phase-1 job for lexfr-embed (v2 — observable).
#   Captures ALL output to /workspace/job.log and uploads it to W&B on exit, so any failure is
#   diagnosable even though the pod self-terminates. STAGE A: GPU smoke gate. STAGE B: graded
#   run_phase1 --skip-retention (BSARD before->after + CI; retention is a separate run). Then
#   ship scorecard to W&B and SELF-TERMINATE. Env: RUNPOD_API_KEY, RUNPOD_POD_ID, WANDB_API_KEY.
set -uo pipefail
mkdir -p /workspace
exec > >(tee -a /workspace/job.log) 2>&1   # capture everything from here on

export WANDB_PROJECT="${WANDB_PROJECT:-lexfr-embed}" WANDB_MODE=online
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
# fit BGE-M3 on a 24GB card: CachedMNRL chunks the batch (mini_batch 16); reduce encode fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LEXFR_EMBED_USE_CACHED_MNRL=true
# RUN_RETENTION=1 -> also run the Axis-3 catastrophic-forgetting guard (installs mteb, drops --skip-retention)
export RUN_RETENTION="${RUN_RETENTION:-0}"
echo "[job] start $(date -u) | WANDB key present: ${WANDB_API_KEY:+yes} | RUNPOD_POD_ID=${RUNPOD_POD_ID:-?}"
nvidia-smi -L || echo "[job] WARN no nvidia-smi"

# wandb + runpod FIRST so logging + self-terminate are guaranteed
pip install -q wandb runpod 2>&1 | tail -2
wandb login "$WANDB_API_KEY" 2>/dev/null || echo "[job] WARN wandb login failed"

selfterminate() {
  echo "[job] ===== uploading job.log to W&B + self-terminating pod ${RUNPOD_POD_ID:-?} ====="
  python - <<'PYEOF' 2>/dev/null || true
import os, wandb
try:
    r = wandb.init(project=os.environ.get("WANDB_PROJECT", "lexfr-embed"), name="runpod-joblog", tags=["runpod", "log"])
    txt = open("/workspace/job.log").read()[-120000:]
    wandb.log({"joblog": wandb.Html("<pre>" + txt.replace("<", "&lt;") + "</pre>")})
    a = wandb.Artifact("joblog", type="log"); a.add_file("/workspace/job.log"); r.log_artifact(a); r.finish()
    print("[job] job.log uploaded to W&B")
except Exception as e:
    print("[job] log upload failed:", e)
PYEOF
  if command -v runpodctl >/dev/null 2>&1; then runpodctl remove pod "${RUNPOD_POD_ID:-}" && return 0; fi
  python - <<'PYEOF' 2>/dev/null || true
import os, runpod
runpod.api_key = os.environ["RUNPOD_API_KEY"]; runpod.terminate_pod(os.environ["RUNPOD_POD_ID"])
PYEOF
}
trap selfterminate EXIT

cd /workspace/lexfr || { echo "[job] repo missing"; exit 1; }
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# The base image ships torch 2.4; transformers requires torch>=2.6 to load BGE-M3's .bin
# weights (CVE-2025-32434). Upgrade within the same CUDA family (image is cuda12.4.1 -> cu124).
echo "[job] ===== upgrading torch>=2.6 (cu124) for BGE-M3 .bin loading ====="
pip install -q -U "torch>=2.6" --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
# torch 2.6 breaks the image's torch-2.4 torchvision (ABI: 'operator torchvision::nms does not exist'),
# which cascades into a transformers import failure. A text embedder needs neither -> remove them.
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true

# now pin the (upgraded) torch so later pip installs never swap it (Kaggle lesson)
python - <<'PY'
import pathlib, torch
pathlib.Path("/tmp/tc.txt").write_text(f"torch=={torch.__version__}\n")
print("[job] torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY

MTEB_PKG=""; [ "$RUN_RETENTION" = "1" ] && MTEB_PKG="mteb"   # retention guard needs mteb; headline run skips it
echo "[job] ===== installing training deps (retention=${RUN_RETENTION}) ====="
pip install -q -U sentence-transformers "datasets>=3" peft accelerate faiss-cpu \
    pydantic-settings rich $MTEB_PKG -c /tmp/tc.txt 2>&1 | tail -3
pip uninstall -y -q torchao 2>/dev/null || true

echo "[job] ===== STAGE A: GPU smoke on a small task ====="
timeout 900 python - <<'PY' || { echo "[job] GPU SMOKE FAILED — aborting before the paid run"; exit 1; }
import os.path, torch, wandb
assert torch.cuda.is_available(), "CUDA not available on the pod"
run = wandb.init(project=os.environ["WANDB_PROJECT"], name="runpod-gpu-smoke", tags=["smoke", "runpod"],
                 config={"gpu": torch.cuda.get_device_name(0)})
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); wandb.log({"matmul_mean": float((x @ x).float().mean())})
from lexfr_embed.train import train_embedder
pairs = [{"anchor": f"question {i} sur le bail et la caution",
          "positive": f"article {i}: le locataire peut resilier le bail ...", "code": "civil"} for i in range(24)]
train_embedder(base_model_key="smoke", use_lora=False, train_pairs=pairs, out_dir="/tmp/gpu_smoke", max_steps=1)
assert os.path.exists("/tmp/gpu_smoke/final")
wandb.log({"gpu_smoke_passed": 1}); run.finish(); print("[job] GPU SMOKE PASSED")
PY

if [ "$RUN_RETENTION" = "1" ]; then
  echo "[job] ===== STAGE B: graded run_phase1 WITH retention (BGE-M3 two-stage + BSARD + MTEB guard) ====="
  export WANDB_NAME="phase1-retention"
  timeout 12000 python scripts/run_phase1.py || echo "[job] run_phase1 non-zero — see log/scorecard"
else
  echo "[job] ===== STAGE B: graded run_phase1 --skip-retention (BGE-M3 two-stage + BSARD) ====="
  export WANDB_NAME="phase1-train"
  timeout 9000 python scripts/run_phase1.py --skip-retention || echo "[job] run_phase1 non-zero — see log/scorecard"
fi

echo "[job] ===== ship results to W&B ====="
python - <<'PY' || true
import os, wandb
r = wandb.init(project=os.environ["WANDB_PROJECT"], name="runpod-scorecard", tags=["runpod", "results"])
for f in ("results/scorecard.md", "results/partition_hashes.json"):
    if os.path.exists(f):
        a = wandb.Artifact(os.path.basename(f).split(".")[0], type="results"); a.add_file(f); r.log_artifact(a)
if os.path.exists("results/scorecard.md"):
    wandb.log({"scorecard": wandb.Html("<pre>" + open("results/scorecard.md").read() + "</pre>")})
r.finish(); print("[job] results shipped")
PY
echo "[job] ===== DONE $(date -u) ====="
