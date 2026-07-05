# RUNBOOK — Bring up RTX 5060 Ti on P710 and reproduce lexfr-embed small+big runs

Target: headless Ubuntu 24.04.4, kernel 6.8.0-134, RTX 5060 Ti (Blackwell GB206, sm_120, PCI `10de:2d04`). Repo: `/home/gdelabie/code/lexfr-embed`.

_Produced by an ultracode hardening workflow (2026-07-04), validated against the live box: Secure Boot ENABLED, nouveau bound, headers+gcc present, dkms missing, apt offers `nvidia-driver-580-server-open 580.159.03-0ubuntu0.24.04.1`._

---

## ⚡ STATUS — YOU ARE HERE (2026-07-04). Chosen path: **Option A (keep Secure Boot)**

**Done remotely already (agent, no console needed) — do NOT re-run the install:**
- ✅ Prereqs: `dkms 3.0.11`, kernel headers `6.8.0-134`, `gcc 13.3`, `mokutil`.
- ✅ Driver installed + configured: `nvidia-driver-580-server-open 580.159.03` → `dkms status: installed`; module `/lib/modules/6.8.0-134-generic/updates/dkms/nvidia.ko.zst`.
- ✅ Module **signed** with MOK `CN=p710-ghislain Secure Boot Module Signature key`.
- ✅ nouveau blacklisted (`/lib/modprobe.d/nvidia-graphics-drivers.conf` + `/etc/modprobe.d/blacklist-nouveau.conf`) and baked into a rebuilt initramfs.
- ✅ MOK enrollment **queued** (`mokutil --import`), pending, password below.

**⏭ THE ONE MANUAL STEP — operator, physically at the P710:**
1. Reboot: `sudo reboot`.
2. On boot a **blue "Perform MOK management" screen** appears (~10s to press a key). Choose **Enroll MOK → Continue → Yes**, then enter the password:

   ### → password: `lexfrgpukey`

   then **Reboot**.
3. ⚠️ **Keyboard caveat:** the MOK screen uses a **US-QWERTY** scancode layout. `lexfrgpukey` was deliberately chosen from letters that sit in the **same position on QWERTY and AZERTY**, so it types identically on a French keyboard. (If it's ever rejected, suspect layout and retype slowly.)
4. If you miss the ~10s prompt and it boots normally, the module won't load (Secure Boot rejects the un-enrolled key — harmless on a headless box). Just `sudo reboot` and catch the blue screen. If the pending request was cleared, re-queue: `sudo mokutil --import /var/lib/shim-signed/mok/MOK.der` (sets a new password).

**⏭ THEN — agent, remote, no console:** run **GATE B** (nvidia-smi) → **§C** (torch `sm_120` + GATE C) → **§D** (W&B + GPU telemetry) → **§E** reproduction ladder (MiniLM ≈0.055→0.148, then BGE-M3 ≈0.240→0.292). Sections B1–B3 below are now **reference only** (already executed).

## Version decision (conflict resolved up front)

- **Driver vs torch-CUDA rule:** driver CUDA version must be ≥ torch's bundled CUDA runtime. R580 driver reports **CUDA 13.0** and is backward-compatible with the 12.x runtime, so it drives **both** cu128 and cu130 wheels.
- **Driver:** `nvidia-driver-580-server-open` (R580 LTSB, **open** module — mandatory on Blackwell; the proprietary module does not drive RTX 50-series). Archive candidate: `580.159.03-0ubuntu0.24.04.1`.
- **Torch:** **`torch==2.11.0` on the `cu128` index** — cu128 ships sm_120 kernels and is the battle-tested Blackwell reference. R580 runs it fine. (cu130/2.12.1 is a fallback only if you must match `uv.lock` exactly.)
- The earlier local "torch 2.12.1+cu130, cuda unavailable" was **NOT a wheel problem** — cu130 already contains sm_120. It was the **missing driver** (nouveau bound). Install the driver and torch works.

---

## A. Pre-flight & safety (read-only checks — done, all confirmed)
```bash
lspci -nn | grep -i nvidia        # [10de:2d04] RTX 5060 Ti  ✓
lsmod | grep -i nouveau           # nouveau loaded/bound  ✓
mokutil --sb-state                # "SecureBoot enabled"  ✓ (blocker)
dpkg -l | grep linux-headers-$(uname -r)   # present  ✓
gcc --version                     # 13.3.0  ✓   (dkms MISSING → install in B1)
```
**Physical-access reality:** P710 has **no BMC/IPMI/remote KVM** and Secure Boot is ON → **one** monitor+keyboard trip is unavoidable (enroll MOK, or disable Secure Boot). Decide **Option A** (keep SB, enroll MOK) or **Option B** (disable SB in UEFI — simplest for a compute-only box).

Rollback:
```bash
sudo apt-get purge -y '^nvidia-.*' 'libnvidia-.*'   # SUDO
sudo apt-get autoremove -y && sudo update-initramfs -u && sudo reboot   # SUDO — nouveau returns
```
Do NOT mix the `.run` installer with apt.

---

## B. Driver install (R580 open, server variant)
```bash
# B1. Prereqs
sudo apt update                                                              # SUDO
sudo apt install -y build-essential dkms linux-headers-generic \
                    linux-headers-$(uname -r) mokutil                        # SUDO
# B2. Confirm availability
apt-cache policy nvidia-driver-580-server-open      # candidate 580.159.03 ✓
ubuntu-drivers devices
# B3. Install the OPEN, SERVER driver (Blackwell REQUIRES -open)
sudo apt install -y nvidia-driver-580-server-open                            # SUDO
grep -r nouveau /etc/modprobe.d/                    # verify auto-blacklist
```
**Reboot is mandatory** (MOK screen is pre-OS; nouveau is bound to the console).
- **Option A (keep SB):** at boot, blue "Perform MOK management" → *Enroll MOK* → password → reboot. Headless forever after.
- **Option B (disable SB):** BIOS → Secure Boot off → reboot. Unsigned open module loads freely.

### GATE B — driver up
```bash
lsmod | grep -i nouveau                      # EMPTY
modinfo nvidia | grep license                # "Dual MIT/GPL" (proves OPEN module)
nvidia-smi                                   # Driver 580.x  CUDA 13.0  "RTX 5060 Ti"
```
"No devices found" → non-open pkg (reinstall `-server-open`). "Key rejected" → MOK not enrolled.

---

## C. PyTorch in the uv project + sm_120 gate
```bash
cd /home/gdelabie/code/lexfr-embed
uv sync --extra dev --extra eval --extra track
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
```
Never let bare-PyPI or cu126 win → "no kernel image is available". Only cu128/cu129/cu130 have sm_120.

### GATE C — sm_120 kernels execute (`uv run python verify_gpu.py`)
```python
import torch
assert torch.cuda.is_available(), "driver missing / nouveau bound (fix Gate B)"
assert "sm_120" in torch.cuda.get_arch_list(), "wheel lacks sm_120; reinstall cu128"
assert torch.cuda.get_device_capability(0) == (12, 0)
a = torch.randn(4096,4096,device="cuda"); b = torch.randn(4096,4096,device="cuda")
c = a @ b; torch.cuda.synchronize()   # forces the kernel to actually run
print("OK", torch.cuda.get_device_name(0), float(c.sum()))
```

---

## D. W&B + GPU telemetry
- **D1 auto system metrics:** `wandb.init(settings=wandb.Settings(x_stats_sampling_interval=5.0))` → `gpu.*` util/mem/temp/power/clocks. (GeForce 5060 Ti = **no ECC** → ECC keys N/A, expected.) Set `WANDB_API_KEY` in `.env` (only `RUNPOD_API_KEY` set now) else `report_to="none"`.
- **D2 nvidia-smi logger** (`scripts/gpu_telemetry.py`) fills gaps W&B misses — fan, pstate, throttle reasons, Xid. Fields: `temperature.gpu,fan.speed,power.draw,power.limit,utilization.gpu,memory.used,clocks.sm,pstate,clocks_event_reasons.*` (renamed from `clocks_throttle_reasons.*` on R535+; probe new→fallback old). Own x-axis via `define_metric(step_metric="gpu/_ts")`. Xid from `dmesg | grep "NVRM: Xid"` (needs `kernel.dmesg_restrict=0` or sudo; tolerate EPERM).

---

## E. Reproduction ladder (gate after each)
Env once: `export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false WANDB_PROJECT=lexfr-embed`
1. **CUDA dry-run** = Gate C.
2. **MiniLM CPU-vs-GPU smoke** — encode a few sentences on cpu+cuda, assert `max|Δ|<1e-2`.
3. **SMALL MiniLM full** — `scripts/phase0_kaggle.py`, but **edit its config block (~lines 42-57) to MiniLM** (`BASE_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `USE_LORA=False`); bf16 ok on Blackwell. **GATE: BSARD NDCG@10 ≈ 0.055 → 0.148** (±0.5 pt; investigate >1 pt).
4. **BIG BGE-M3 two-stage @512** — `LEXFR_EMBED_USE_CACHED_MNRL=true uv run --extra eval python scripts/run_phase1.py --skip-retention --subset 15000`. **GATE: ≈ 0.240 → 0.292** (0.292 needs branch `feat/rehearsal-floor` commit `5d22e77`; plain `main` gives ≈0.284–0.290). Keep `max_seq_len=512` before/after (the old 0.307 was a 512-vs-1024 splice). 5060 Ti ~16 GB: if OOM lower mini-batch/encode batch below 16.

---

## F. Reproducibility + realistic tolerance
Set before CUDA init: seeds (42), `torch.use_deterministic_algorithms(True)`, `cudnn.benchmark=False`, `cudnn.deterministic=True`, `allow_tf32=False`, `set_float32_matmul_precision("highest")`; pin batch/accum/shuffle; log torch/ST/transformers/driver versions + git SHA.
- **Re-eval a fixed checkpoint:** ~exact, **±0.1 pt**.
- **Re-train from same recipe on 5060 Ti (T4/RunPod → Blackwell, cu124→cu128):** bit-exact impossible; band **±0.3–1.0 pt**; accept **≈±0.5 pt**, investigate >1 pt as a config/version bug. Measure your own noise floor: same seed twice locally.

---

## G. MLflow decision + MLOps checklist
**Do NOT add MLflow.** W&B already covers tracking/sweeps/artifacts/registry/lineage + GPU system metrics; MLflow's only real delta is serving, better done with HF **TEI** or a thin FastAPI. Dual-tracking splits the source of truth. (Exception: if an evaluator explicitly wants a self-hosted OSS stack, add MLflow only as packaging/serving; W&B stays the hub.)

Checklist: pin env (uv.lock + torch/CUDA build + GPU/driver in run config) · log seeds + determinism status · data as immutable W&B Artifact + dataset card · config-as-code logged · frozen eval harness · always report base-model baseline · log training + system metrics · checkpoints as artifacts w/ Registry lineage · model card + aliases (candidate→staging→production) mirrored to HF · CI gate on metric regression · containerized TEI/FastAPI pinned to the evaluated revision · documented retrain trigger + drift check.

---

### Known failure modes
- `nvidia-smi` "No devices found" → non-`open` pkg; install `-server-open`.
- "Key rejected" / nouveau reappears → MOK not enrolled (§B A/B).
- torch "no kernel image" → cu126 wheel; reinstall cu128.
- `is_available()==False` → driver missing / nouveau bound (Gate B), not a wheel problem.
- Big run 0.290 not 0.292 → you're on `main`; 0.292 needs `feat/rehearsal-floor`.
- Metric gap >1 pt → config bug (max_seq_len/tokenizer/pooling/batch/TF32), not GPU noise.
