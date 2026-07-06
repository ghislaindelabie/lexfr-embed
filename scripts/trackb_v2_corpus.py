"""Track-B v2 sizing: measure the lexical fail-rate when the corpus is enlarged to ALL
deduped LegalKit articles (trained ones as distractors), and the trained-article
near-duplicate leakage risk. Appends to results/day/trackb_forensics.json.

CPU-only. CUDA_VISIBLE_DEVICES="" uv run --no-sync python scripts/trackb_v2_corpus.py
"""

from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

REPO = Path(__file__).resolve().parent.parent
CACHE = Path("/tmp/claude-1000/-home-gdelabie-code/516a4024-4d6c-4036-a0a2-0a3a96ab7f29/scratchpad")
OUT = REPO / "results" / "day" / "trackb_forensics.json"
SEED = 42
K = 10


def z(p):
    from statistics import NormalDist

    return NormalDist().inv_cdf(p)


def mde(n, sd, alpha=0.05, power=0.8):
    if n <= 0:
        return float("inf")
    return (z(1 - alpha / 2) + z(power)) * sd / math.sqrt(n)


def build_full():
    cache = CACHE / "v2_full.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    from lexfr_embed.data.legalkit import load_legalkit
    from lexfr_embed.data.trackb import build_holdout, trained_ids, pair_id

    all_pairs = load_legalkit(None)
    trained = trained_ids()
    held_ids = {pair_id(p) for p in all_pairs if pair_id(p) not in trained}

    # full deduped article corpus (by exact text) + which texts are held-out golds
    text2cid: dict[str, str] = {}
    corpus_texts: list[str] = []
    is_heldout_gold = []  # per corpus row: text appears as a held-out pair's positive
    heldout_gold_texts: set[str] = set()
    trained_texts: set[str] = set()
    for p in all_pairs:
        if pair_id(p) in held_ids:
            heldout_gold_texts.add(p["positive"])
        else:
            trained_texts.add(p["positive"])
    for p in all_pairs:
        t = p["positive"]
        if t not in text2cid:
            text2cid[t] = len(corpus_texts)
            corpus_texts.append(t)
            is_heldout_gold.append(t in heldout_gold_texts)

    # all held-out pairs as queries (the v2 candidate query pool)
    q_anchor, q_gold_idx = [], []
    seen_pairs = set()
    for p in all_pairs:
        if pair_id(p) in held_ids:
            key = (p["anchor"], p["positive"])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            q_anchor.append(p["anchor"])
            q_gold_idx.append(text2cid[p["positive"]])

    # texts that are held-out golds AND also a trained positive (pure leakage if used as answer)
    overlap_gold_trained = len(heldout_gold_texts & trained_texts)

    data = {
        "corpus_texts": corpus_texts,
        "is_heldout_gold": is_heldout_gold,
        "q_anchor": q_anchor,
        "q_gold_idx": q_gold_idx,
        "n_full_corpus": len(corpus_texts),
        "n_heldout_gold_articles": len(heldout_gold_texts),
        "n_trained_articles": len(trained_texts),
        "n_gold_also_trained_text": overlap_gold_trained,
        "n_query_pool": len(q_anchor),
    }
    cache.write_bytes(pickle.dumps(data))
    return data


def main():
    print("building full corpus ...")
    d = build_full()
    print(f"  full corpus={d['n_full_corpus']}  query_pool={d['n_query_pool']}  "
          f"heldout_gold_articles={d['n_heldout_gold_articles']}  gold_also_trained_text={d['n_gold_also_trained_text']}")

    vec = TfidfVectorizer(sublinear_tf=True, lowercase=True, dtype=np.float32)
    Xc = vec.fit_transform(d["corpus_texts"])
    Xq = vec.transform(d["q_anchor"])
    gold = np.asarray(d["q_gold_idx"])
    is_gold = np.asarray(d["is_heldout_gold"], dtype=bool)
    trained_cols = np.where(~is_gold)[0]  # corpus rows that are NOT held-out golds (distractors incl. trained)

    n_q = Xq.shape[0]
    top1 = np.zeros(n_q, dtype=bool)
    recallk = np.zeros(n_q, dtype=bool)
    # false-negative risk: top-1 retrieved is a DIFFERENT article at cos>0.8 (plausible unlabeled positive)
    fn_top1_neardup = np.zeros(n_q, dtype=bool)
    batch = 512
    for s in range(0, n_q, batch):
        e = min(s + batch, n_q)
        sims = (Xq[s:e] @ Xc.T).toarray()
        part = np.argpartition(-sims, kth=K, axis=1)[:, :K]
        for i in range(e - s):
            row = sims[i]
            topk = part[i][np.argsort(-row[part[i]])]
            g = gold[s + i]
            top1[s + i] = topk[0] == g
            recallk[s + i] = g in topk
            if topk[0] != g and row[topk[0]] > 0.8:
                fn_top1_neardup[s + i] = True
        print(f"  retrieved {e}/{n_q}", end="\r")
    print()

    fail1 = float(1 - top1.mean())
    fail10 = float(1 - recallk.mean())

    # trained-article near-dup leakage: for held-out gold articles, is there a NON-gold
    # (trained/distractor) article at TF-IDF cos>0.8? -> would be an unlabeled positive in v2.
    g_rows = np.where(is_gold)[0]
    Xg = Xc[g_rows]
    Xd = Xc[trained_cols]  # distractor (non-held-out-gold) articles
    has_trained_dup = np.zeros(len(g_rows), dtype=bool)
    b = 256
    for s in range(0, len(g_rows), b):
        e = min(s + b, len(g_rows))
        sims = (Xg[s:e] @ Xd.T).toarray()
        has_trained_dup[s : e] = sims.max(axis=1) > 0.8
    frac_gold_trained_dup = float(has_trained_dup.mean())

    n_pool = d["n_query_pool"]
    res = {
        "n_full_corpus": d["n_full_corpus"],
        "n_heldout_gold_articles": d["n_heldout_gold_articles"],
        "n_trained_distractor_articles": int(len(trained_cols)),
        "n_gold_also_trained_text_PURE_LEAK": d["n_gold_also_trained_text"],
        "query_pool": n_pool,
        "tfidf_full_corpus": {
            "top1_acc": float(top1.mean()),
            "recall_at_10": float(recallk.mean()),
            "fail_at_1": fail1,
            "fail_at_10": fail10,
        },
        "false_negative_risk": {
            "frac_queries_top1_is_neardup_gt0.8": float(fn_top1_neardup.mean()),
            "frac_heldout_gold_with_trained_neardup_gt0.8": frac_gold_trained_dup,
        },
        "surviving_hard_queries": {
            "filter_fail_at_10_fullcorpus": int(round(n_pool * fail10)),
            "filter_fail_at_1_fullcorpus": int(round(n_pool * fail1)),
        },
        "mde": {},
    }
    for scen, n in (
        ("fail@10_fullcorpus", res["surviving_hard_queries"]["filter_fail_at_10_fullcorpus"]),
        ("fail@1_fullcorpus", res["surviving_hard_queries"]["filter_fail_at_1_fullcorpus"]),
        ("all_pool_no_filter", n_pool),
    ):
        res["mde"][scen] = {f"sd{sd}": mde(n, sd) for sd in (0.30, 0.35, 0.40)}

    out = json.loads(OUT.read_text())
    out["v2_full_corpus"] = res
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nappended to {OUT}")


if __name__ == "__main__":
    main()
