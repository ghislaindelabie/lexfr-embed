"""Build the A1-bis distillation teacher cache -> results/distill_cache/ (OFFLINE, inference-only).

Two decoupled sub-steps, run in SEQUENCE — both fit 16 GB because neither builds a training graph:

  Step A  mine top-K candidates with the STAGE-1 embedder ONLY (no reranker, no filter) via
          `hard_negatives.mine_teacher_candidates` -> (anchor, positive, negative_1..K) n-tuples.
  Step B  score every (query, candidate) with the TEACHER reranker (`bge-reranker-v2-m3`), taking RAW
          logits (`activation_fn=Identity`) then applying the teacher activation EXPLICITLY in Python
          via `distill.teacher_label_row` — never relying on CrossEncoder's implicit per-model default
          (SILENT-BUG #2: raw logits ±10 vs cosine margins ±2 fit to nothing without crashing).

Writes a `datasets` Arrow dataset (anchor, positive, negative_1..K, label[len K+1], raw_scores) with
`Dataset.save_to_disk`, PLUS a `meta.json` provenance sidecar (miner ckpt + content hash, reranker id,
activation, K, LegalKit subset size+seed, loader-dedup identity, pair/row counts, rows dropped, ST
version) that `train.distill_embedder` asserts against `settings` before training (cache/training
alignment). The miner checkpoint is loaded with a HARD existence check — NO silent base-bge-m3 fallback
(that would mine OFF-POLICY negatives and understate the gain).

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/build_distill_cache.py
    # device-agnostic: export CUDA_VISIBLE_DEVICES="" to force a (slow) CPU build.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # mirror build_trackb_v2; setdefault keeps a CPU override ("")

RERANK_BATCH = 32  # bge-reranker-v2-m3 ~42 pairs/s @150W fp32 on the 5060 Ti (PROJECT_LOG)


def _ckpt_content_hash(path: str) -> str:
    """Lightweight content fingerprint of a checkpoint dir: (relpath, size) for every file plus the
    bytes of small text/JSON config files. Avoids reading multi-GB safetensors while still changing if
    the model, adapter config, or weight sizes change (detects a wrong/stale miner checkpoint)."""
    root = Path(path)
    h = hashlib.sha256()
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(root).as_posix()
        h.update(rel.encode())
        h.update(str(f.stat().st_size).encode())
        if f.suffix in {".json", ".txt"} and f.stat().st_size < 1_000_000:
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _load_miner():
    """Load the stage-1 miner embedder with a HARD existence check (no silent base fallback)."""
    from sentence_transformers import SentenceTransformer

    from lexfr_embed.config import settings

    ckpt = settings.distill_miner_ckpt
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"miner checkpoint {ckpt!r} missing — train stage-1 first (results/phase1/stage1). "
            "Refusing to fall back to base bge-m3: that would mine OFF-POLICY negatives (PROJECT_LOG risk)."
        )
    m = SentenceTransformer(ckpt)
    m.max_seq_length = settings.max_seq_len
    return m


def main() -> None:
    import sentence_transformers as st
    import torch
    from sentence_transformers import CrossEncoder

    from lexfr_embed.config import settings
    from lexfr_embed.data.hard_negatives import mine_teacher_candidates
    from lexfr_embed.data.legalkit import load_legalkit
    from lexfr_embed.distill import build_distill_dataset, teacher_label_row

    t0 = time.time()
    k = settings.distill_num_negatives
    activation = settings.distill_teacher_activation

    # Training pairs = the SAME LegalKit subset (size+seed) the distill trainer will consume. The
    # loader-dedup identity is hashed so a mismatched cache is caught at train-load time.
    pairs = load_legalkit(settings.phase0_subset, seed=settings.seed)
    dedup_identity = hashlib.sha256(
        "\n".join(sorted(f"{p['anchor']}\x1f{p['positive']}" for p in pairs)).encode()
    ).hexdigest()[:16]
    print(f"[cache] {len(pairs)} LegalKit pairs (subset={settings.phase0_subset}, seed={settings.seed})")

    # --- Step A: embedder-only candidate mining (no reranker, no filter) ---
    miner = _load_miner()
    mined = mine_teacher_candidates(pairs, miner, num_negatives=k)
    print(f"[cache] mined {len(mined)} n-tuples (K={k}) with {settings.distill_miner_ckpt}")
    del miner
    if torch.cuda.is_available():  # free the embedder before the reranker loads — sequential, 16 GB-safe
        torch.cuda.empty_cache()

    # --- Step B: teacher reranker scoring (raw logits -> explicit activation) ---
    reranker = CrossEncoder(settings.distill_reranker_id, max_length=settings.max_seq_len)
    neg_cols = [f"negative_{i}" for i in range(1, k + 1)]
    flat_pairs: list[tuple[str, str]] = []
    for row in mined:
        q = row["anchor"]
        flat_pairs.append((q, row["positive"]))
        flat_pairs.extend((q, row[c]) for c in neg_cols)
    raw = reranker.predict(
        flat_pairs, activation_fn=torch.nn.Identity(), batch_size=RERANK_BATCH, convert_to_numpy=True
    )
    raw = [float(x) for x in raw]

    records = []
    stride = k + 1
    for i, row in enumerate(mined):
        block = raw[i * stride : (i + 1) * stride]  # [pos_logit, neg_1_logit, ..., neg_K_logit]
        pos_raw, neg_raws = block[0], block[1:]
        records.append(
            {
                "anchor": row["anchor"],
                "positive": row["positive"],
                "negatives": [row[c] for c in neg_cols],
                "label": teacher_label_row(pos_raw, neg_raws, activation),
                "raw_scores": block,
            }
        )

    ds = build_distill_dataset(records)
    raw_kept = [records[i]["raw_scores"] for i in range(len(records)) if len(records[i]["negatives"]) >= k]
    ds = ds.add_column("raw_scores", raw_kept)  # pre-activation logits -> a different transform can be re-derived
    n_dropped = len(records) - len(ds)

    out_dir = str(settings.distill_cache_dir)
    ds.save_to_disk(out_dir)
    meta = {
        "miner_ckpt": settings.distill_miner_ckpt,
        "miner_ckpt_hash": _ckpt_content_hash(settings.distill_miner_ckpt),
        "reranker_id": settings.distill_reranker_id,
        "activation": activation,
        "num_negatives": k,
        "subset_size": settings.phase0_subset,
        "subset_seed": settings.seed,
        "loader_dedup_identity": dedup_identity,
        "n_pairs": len(pairs),
        "n_rows": len(ds),
        "n_dropped_incomplete": n_dropped,
        "max_seq_len": settings.max_seq_len,  # 512 truncates long articles -> notes the teacher-score bias
        "sentence_transformers_version": st.__version__,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (Path(out_dir) / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[cache] wrote {out_dir} ({len(ds)} rows, dropped {n_dropped}) + meta.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
