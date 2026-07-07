"""Portable GPU micro-benchmark — matched 5060 Ti vs T4 (Kaggle) comparison.

Replicates our training regime (BGE-M3 568M + LoRA r16/alpha32, batch 16, seq 512) and times
pure forward+backward+optimizer throughput, isolating raw GPU speed from data/eval noise. The
SAME script runs on the local 5060 Ti and on a Kaggle/RunPod T4 -> the steps/s ratio is the real
speedup (replacing the specs-only estimate in the bilan). Prints one machine-readable line.

    uv run --no-sync python scripts/gpu_microbench.py            # local 5060 Ti
    (Kaggle kernel pip-installs transformers/peft, then runs this verbatim on a T4)
"""

from __future__ import annotations

import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

MODEL = "BAAI/bge-m3"
BS, SEQ, STEPS, WARMUP = 16, 512, 60, 10  # mini_batch=16 & seq=512 = our CachedMNRL regime


def main() -> None:
    assert torch.cuda.is_available(), "no CUDA device"
    dev = torch.device("cuda")
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).train()
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "key", "value", "dense"], lora_dropout=0.05)
    model = get_peft_model(model, lora)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    # one fixed ~seq-length batch (throughput bench: content is irrelevant, size is what matters)
    filler = "Article L. 3141-1 du code du travail relatif aux congés payés du salarié. " * 40
    enc = tok([filler] * BS, padding="max_length", truncation=True, max_length=SEQ, return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()}
    labels = torch.arange(BS, device=dev)

    scaler = torch.cuda.amp.GradScaler()  # fp16 = our T4 training regime, and fits the 14.5 GiB T4
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for i in range(STEPS):
        if i == WARMUP:
            torch.cuda.synchronize()
            t0 = time.time()
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            emb = torch.nn.functional.normalize(model(**enc).last_hidden_state.mean(dim=1), dim=1)
            loss = torch.nn.functional.cross_entropy(emb @ emb.t() / 0.05, labels)  # in-batch InfoNCE
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.time() - t0
    n = STEPS - WARMUP
    name = torch.cuda.get_device_name(0)
    print(
        f"BENCH gpu={name!r} torch={torch.__version__} bs={BS} seq={SEQ} steps={n} "
        f"wall_s={dt:.2f} steps_per_s={n / dt:.4f} samples_per_s={n * BS / dt:.2f} "
        f"peak_vram_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}"
    )


if __name__ == "__main__":
    main()
