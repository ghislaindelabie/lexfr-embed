"""Eval-expansion for lexfr-embed — the campaign proved BSARD (n=222) can't distinguish configs,
so this gathers MORE and MORE-DISCRIMINATIVE evidence on models/checkpoints we already have,
without training. Inference-only. Writes JSON + prints tables. Reuses evaluate/metrics/quantize.

Modes:
  powered    NDCG@10 + Recall@10 + bootstrap CI + MDE on BSARD test(222) AND traintest(1108)
             -> honest held-out numbers + a smaller MDE (T1/T8).
  matryoshka encode once, truncate to {1024,512,256,128,64} + renorm -> NDCG per dim (T3).
  rerank     dense top-100 -> cross-encoder (bge-reranker-v2-m3) re-rank -> NDCG before/after,
             paired bootstrap CI (T2, the "is fine-tuning worth it" counter-evidence).

    uv run --no-sync python scripts/eval_extra.py --mode powered   --model bge-m3
    uv run --no-sync python scripts/eval_extra.py --mode matryoshka --model <ckpt_dir>
    uv run --no-sync python scripts/eval_extra.py --mode rerank     --model bge-m3 --split test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MATRYOSHKA = [1024, 512, 256, 128, 64]


def load_split(split: str):
    """BSARD split; 'traintest' = train(886)+test(222) merged (bigger n -> smaller MDE);
    'trackb' = large leak-free held-out LegalKit eval (in-distribution, powered — see data.trackb)."""
    if split == "trackb":
        from lexfr_embed.data.trackb import load_trackb

        return load_trackb()
    if split == "trackb2":
        # BM25-hard v2 artifact (see data/trackb_v2.py) — build once with scripts/build_trackb_v2.py
        import gzip

        with gzip.open("results/trackb_v2.json.gz", "rt", encoding="utf-8") as f:
            d = json.load(f)
        return d["queries"], d["corpus"], {q: set(r) for q, r in d["relevant"].items()}
    if split == "tax":
        # External FR professional (tax-law) retrieval eval — louisbrulenaudet/tax-retrieval-benchmark
        # (query, positive) pairs; NB Lemone-embed is a tax specialist -> home-turf advantage on this split.
        from datasets import load_dataset

        from lexfr_embed.data.trackb import build_trackb_eval

        ds = load_dataset("louisbrulenaudet/tax-retrieval-benchmark", "default", split="train")
        pairs = [{"anchor": r["query"], "positive": r["positive"], "code": "Fiscal"} for r in ds]
        return build_trackb_eval(pairs, n_queries=None)
    from lexfr_embed.evaluate import load_bsard

    if split != "traintest":
        return load_bsard(split)
    qs, corpus, rel = {}, None, {}
    for s in ("train", "test"):
        q, c, r = load_bsard(s)
        corpus = c
        for k, v in q.items():
            qs[f"{s}:{k}"] = v
        for k, v in r.items():
            rel[f"{s}:{k}"] = v
    return qs, corpus, rel


def load_model(spec: str):
    """Frozen base ('bge-m3') or a saved LoRA checkpoint dir. Returns a SentenceTransformer."""
    from sentence_transformers import SentenceTransformer

    if spec == "bge-m3":
        m = SentenceTransformer("BAAI/bge-m3")
    else:
        try:  # ST can reload a saved dir directly; trust_remote_code for custom-arch models (e.g. Lemone/GTE)
            m = SentenceTransformer(spec, trust_remote_code=True)
        except Exception:  # fallback: base + PEFT adapter
            m = SentenceTransformer("BAAI/bge-m3")
            m.load_adapter(spec) if hasattr(m, "load_adapter") else m[0].auto_model.load_adapter(spec)
    from lexfr_embed.config import settings

    m.max_seq_length = settings.max_seq_len
    return m


def _bootstrap(values, n_boot=5000, seed=42):
    import numpy as np

    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _recall_at_k(model, queries, corpus, relevant, k=10, batch_size=32):
    from sentence_transformers import util

    cids, qids = list(corpus), list(queries)
    ce = model.encode(
        [corpus[c] for c in cids],
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    qe = model.encode(
        [queries[q] for q in qids],
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    hits = util.semantic_search(qe, ce, top_k=k)
    out = []
    for i, q in enumerate(qids):
        gold = relevant.get(q, set())
        got = {cids[h["corpus_id"]] for h in hits[i]}
        out.append(len(got & gold) / max(1, len(gold)))
    return out


def mode_powered(model_spec, splits):
    from lexfr_embed.evaluate import per_query_ndcg_at_k
    from lexfr_embed.metrics import min_detectable_effect

    m = load_model(model_spec)
    res = {}
    for split in splits:
        q, c, r = load_split(split)
        ndcg = per_query_ndcg_at_k(m, q, c, r, k=10, batch_size=16)
        rec = _recall_at_k(m, q, c, r, k=10, batch_size=16)
        import numpy as np

        mean, lo, hi = _bootstrap(ndcg)
        mde = min_detectable_effect(len(ndcg), float(np.std(ndcg)))
        res[split] = {
            "n": len(ndcg),
            "ndcg10": mean,
            "ndcg_ci": [lo, hi],
            "mde": mde,
            "recall10": float(np.mean(rec)),
            "per_query": [round(float(x), 6) for x in ndcg],  # enables PAIRED cross-model deltas offline
        }
        print(
            f"[powered] {model_spec} {split:9} n={len(ndcg):4d} NDCG@10={mean:.4f} "
            f"CI[{lo:.4f},{hi:.4f}] MDE={mde:.4f} R@10={np.mean(rec):.4f}"
        )
    return res


def mode_matryoshka(model_spec, split):
    from sentence_transformers import util

    from lexfr_embed.metrics import ndcg_at_k
    from lexfr_embed.quantize import truncate_matryoshka

    m = load_model(model_spec)
    q, c, r = load_split(split)
    cids, qids = list(c), list(q)
    ce = m.encode(
        [c[x] for x in cids], batch_size=16, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
    )
    qe = m.encode(
        [q[x] for x in qids], batch_size=16, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
    )
    import numpy as np
    import torch

    res = {}
    full = ce.shape[1]
    for dim in [d for d in MATRYOSHKA if d <= full]:
        cet = torch.nn.functional.normalize(ce[:, :dim], p=2, dim=1)
        qet = torch.nn.functional.normalize(qe[:, :dim], p=2, dim=1)
        hits = util.semantic_search(qet, cet, top_k=10)
        ndcg = [ndcg_at_k([cids[h["corpus_id"]] for h in hits[i]], r.get(x, set()), 10) for i, x in enumerate(qids)]
        res[dim] = float(np.mean(ndcg))
        print(f"[matryoshka] {model_spec} dim={dim:4d} NDCG@10={np.mean(ndcg):.4f}")
    _ = truncate_matryoshka  # (documents the reusable helper; inline renorm above for the tensor path)
    return res


def mode_rerank(model_spec, split):
    from sentence_transformers import CrossEncoder, util

    from lexfr_embed.metrics import ndcg_at_k

    m = load_model(model_spec)
    q, c, r = load_split(split)
    cids, qids = list(c), list(q)
    ce = m.encode(
        [c[x] for x in cids], batch_size=16, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
    )
    qe = m.encode(
        [q[x] for x in qids], batch_size=16, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
    )
    hits = util.semantic_search(qe, ce, top_k=100)
    dense = [ndcg_at_k([cids[h["corpus_id"]] for h in hits[i]], r.get(x, set()), 10) for i, x in enumerate(qids)]

    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    reranked = []
    for i, x in enumerate(qids):
        cand = [h["corpus_id"] for h in hits[i]]
        scores = reranker.predict([[q[x], c[cids[j]]] for j in cand], batch_size=32, show_progress_bar=False)
        order = [cand[j] for j in sorted(range(len(cand)), key=lambda t: -scores[t])][:10]
        reranked.append(ndcg_at_k([cids[j] for j in order], r.get(x, set()), 10))

    import numpy as np

    from lexfr_embed.metrics import paired_delta_ci

    d, lo, hi = paired_delta_ci(dense, reranked, seed=42)
    print(
        f"[rerank] {model_spec} {split} dense NDCG@10={np.mean(dense):.4f} -> +rerank={np.mean(reranked):.4f} "
        f"(Δ {d:+.4f}, CI[{lo:+.4f},{hi:+.4f}])"
    )
    return {
        "split": split,
        "n": len(dense),
        "dense": float(np.mean(dense)),
        "reranked": float(np.mean(reranked)),
        "delta": d,
        "ci": [lo, hi],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["powered", "matryoshka", "rerank"])
    ap.add_argument("--model", default="bge-m3", help="'bge-m3' or a checkpoint dir")
    ap.add_argument("--split", default="test", help="test | train | traintest")
    ap.add_argument("--splits", default="test,traintest", help="comma list (powered mode)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    if args.mode == "powered":
        out = mode_powered(args.model, args.splits.split(","))
    elif args.mode == "matryoshka":
        out = mode_matryoshka(args.model, args.split)
    else:
        out = mode_rerank(args.model, args.split)

    payload = {"mode": args.mode, "model": args.model, "result": out, "seconds": round(time.time() - t0, 1)}
    dest = Path(args.out) if args.out else Path("results/eval_extra") / f"{args.mode}_{Path(args.model).name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[eval_extra] wrote {dest}")


if __name__ == "__main__":
    main()
