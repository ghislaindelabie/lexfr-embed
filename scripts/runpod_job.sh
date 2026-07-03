#!/usr/bin/env bash
# RunPod fire-and-forget Phase-1 job for lexfr-embed.
#   STAGE A: real GPU test on a small task (gate) -> STAGE B: graded run_phase1 -> ship results to W&B
#   -> SELF-TERMINATE (always, via trap). Everything timeout-bounded so cost is capped even on failure.
# Expects env: RUNPOD_API_KEY, RUNPOD_POD_ID (auto), WANDB_API_KEY, WANDB_PROJECT.
set -uo pipefail

selfterminate() {
  echo "[job] ===== self-terminating pod ${RUNPOD_POD_ID:-?} ====="
  if command -v runpodctl >/dev/null 2>&1; then runpodctl remove pod "${RUNPOD_POD_ID:-}" && return 0; fi
  python - <<'PY' 2>/dev/null || true
import os, runpod
runpod.api_key = os.environ["RUNPOD_API_KEY"]
runpod.terminate_pod(os.environ["RUNPOD_POD_ID"])
PY
}
trap selfterminate EXIT

export WANDB_PROJECT="${WANDB_PROJECT:-lexfr-embed}"
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
cd /workspace/lexfr || { echo "[job] repo missing"; exit 1; }
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "[job] ===== GPU ====="; nvidia-smi -L || { echo "[job] NO GPU"; exit 1; }

# Pin the image's torch so pip never swaps it (the Kaggle lesson); log the versions.
python - <<'PY'
import pathlib, torch
pathlib.Path("/tmp/tc.txt").write_text(f"torch=={torch.__version__}\n")
print("[job] torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY

echo "[job] ===== installing deps ====="
pip install -q -U sentence-transformers "datasets>=3" peft "mteb>=2" wandb ir-measures \
    faiss-cpu accelerate pydantic-settings rich runpod -c /tmp/tc.txt
pip uninstall -y -q torchao 2>/dev/null || true   # peft 0.19 rejects Kaggle/torchao <0.16; unused for LoRA

wandb login "$WANDB_API_KEY" 2>/dev/null || true

echo "[job] ===== STAGE A: GPU smoke on a small task ====="
timeout 900 python - <<'PY' || { echo "[job] GPU SMOKE FAILED — aborting before the paid run"; exit 1; }
import os.path, torch, wandb
assert torch.cuda.is_available(), "CUDA not available"
run = wandb.init(project=os.environ["WANDB_PROJECT"], name="runpod-gpu-smoke", tags=["smoke", "runpod"],
                 config={"gpu": torch.cuda.get_device_name(0)})
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
wandb.log({"matmul_mean": float((x @ x).float().mean().item())})
from lexfr_embed.train import train_embedder
pairs = [{"anchor": f"question {i} sur le bail et la caution",
          "positive": f"article {i}: le locataire peut resilier le bail ...", "code": "civil"} for i in range(24)]
train_embedder(base_model_key="smoke", use_lora=False, train_pairs=pairs, out_dir="/tmp/gpu_smoke", max_steps=1)
assert os.path.exists("/tmp/gpu_smoke/final"), "no checkpoint saved"
wandb.log({"gpu_smoke_passed": 1}); run.finish()
print("[job] GPU SMOKE PASSED")
PY

echo "[job] ===== STAGE B: graded run_phase1 (bge-m3 two-stage + BSARD + retention) ====="
export WANDB_NAME="phase1-train"
timeout 9000 python scripts/run_phase1.py || echo "[job] run_phase1 exited non-zero — see scorecard/log"

echo "[job] ===== ship results to W&B (survive termination) ====="
python - <<'PY' || true
import os, wandb
run = wandb.init(project=os.environ["WANDB_PROJECT"], name="runpod-scorecard", tags=["runpod", "results"])
for f in ("results/scorecard.md", "results/partition_hashes.json"):
    if os.path.exists(f):
        art = wandb.Artifact(os.path.basename(f).split(".")[0], type="results"); art.add_file(f); run.log_artifact(art)
if os.path.exists("results/scorecard.md"):
    wandb.log({"scorecard": wandb.Html("<pre>" + open("results/scorecard.md").read() + "</pre>")})
run.finish(); print("[job] results shipped to W&B")
PY
echo "[job] ===== DONE ====="
