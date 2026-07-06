"""Track-B / tax forensics — QUANTIFY why the LLM-generated legal-retrieval evals are near-ceiling.

CPU-only. Run: CUDA_VISIBLE_DEVICES="" uv run --no-sync python scripts/trackb_forensics.py
Writes results/day/trackb_forensics.json and prints a compact summary.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import re
import statistics
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

REPO = Path(__file__).resolve().parent.parent
CACHE = Path("/tmp/claude-1000/-home-gdelabie-code/516a4024-4d6c-4036-a0a2-0a3a96ab7f29/scratchpad")
CACHE.mkdir(parents=True, exist_ok=True)
OUT = REPO / "results" / "day" / "trackb_forensics.json"

SEED = 42
N_SAMPLE = 500
K = 10

# Comprehensive-enough French stopword set (content-word filter also requires len>3).
FR_STOP = set(
    """
au aux avec ce ces dans de des du elle en et eux il je la le les leur lui ma mais me meme
mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi
ton tu un une vos votre vous cet cette ceux dont donc quel quels quelle quelles quoi comment
lequel laquelle lesquels lesquelles auquel duquel plus moins tres tout tous toute toutes
etre avoir fait faire cela ceci celui celle entre sans sous chaque aussi alors ainsi entre
notamment lorsque lorsqu selon apres avant pendant depuis contre vers chez leurs certain
certaine certains certaines autre autres meme memes doit peut sont est ete etait avait
quelle quels quelles cas lors afin dune duno leur etc via
""".split()
)

_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^\w ]+", " ", text)
    return _WS.sub(" ", text).strip()


def content_words(text: str) -> set[str]:
    return {t for t in norm(text).split() if len(t) > 3 and t not in FR_STOP}


CITE = re.compile(r"\bart(?:icle|\.)\s*\d|\bl\.?\s*\d{2,}|\br\.?\s*\d{2,}|\bd\.?\s*\d{2,}", re.I)


# --------------------------------------------------------------------------- data


def build_trackb():
    """Reproduce load_trackb() but keep parallel metadata (code, positive text, cite flag)."""
    cache = CACHE / "trackb_built.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    from lexfr_embed.data.legalkit import stratify_by_code
    from lexfr_embed.data.trackb import build_holdout, trained_ids
    from lexfr_embed.data.legalkit import load_legalkit

    all_pairs = load_legalkit(None)
    held = build_holdout(all_pairs, trained_ids())

    text2cid: dict[str, str] = {}
    corpus_texts: list[str] = []
    for p in held:
        t = p["positive"]
        if t not in text2cid:
            text2cid[t] = f"a{len(text2cid)}"
            corpus_texts.append(t)

    sampled = stratify_by_code(held, 5000, seed=SEED)
    q_anchor, q_gold_idx, q_code = [], [], []
    for p in sampled:
        q_anchor.append(p["anchor"])
        q_gold_idx.append(int(text2cid[p["positive"]][1:]))
        q_code.append(p.get("code", "unknown"))

    data = {
        "n_heldout_pairs": len(held),
        "corpus_texts": corpus_texts,
        "q_anchor": q_anchor,
        "q_gold_idx": q_gold_idx,
        "q_code": q_code,
    }
    cache.write_bytes(pickle.dumps(data))
    return data


def build_tax():
    cache = CACHE / "tax_built.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    from datasets import load_dataset

    ds = load_dataset("louisbrulenaudet/tax-retrieval-benchmark", "default", split="train")
    pairs = [{"anchor": r["query"], "positive": r["positive"]} for r in ds]
    text2cid: dict[str, str] = {}
    corpus_texts: list[str] = []
    for p in pairs:
        t = p["positive"]
        if t not in text2cid:
            text2cid[t] = f"a{len(text2cid)}"
            corpus_texts.append(t)
    q_anchor = [p["anchor"] for p in pairs]
    q_gold_idx = [int(text2cid[p["positive"]][1:]) for p in pairs]
    data = {
        "n_pairs": len(pairs),
        "corpus_texts": corpus_texts,
        "q_anchor": q_anchor,
        "q_gold_idx": q_gold_idx,
        "q_code": ["Fiscal"] * len(pairs),
    }
    cache.write_bytes(pickle.dumps(data))
    return data


# --------------------------------------------------------------------------- analyses


def quartiles(vals):
    a = sorted(vals)
    return {
        "mean": float(statistics.mean(a)),
        "median": float(statistics.median(a)),
        "q1": float(np.percentile(a, 25)),
        "q3": float(np.percentile(a, 75)),
        "min": float(a[0]),
        "max": float(a[-1]),
    }


def analysis_overlap(data, rng, n_sample):
    """(1) content-word containment: query CWs found in gold vs 5 random articles."""
    corpus_texts = data["corpus_texts"]
    n_q = len(data["q_anchor"])
    idx = rng.sample(range(n_q), min(n_sample, n_q))
    corpus_cw = None  # lazy: only need random targets

    gold_cont, rand_cont = [], []
    N = len(corpus_texts)
    for qi in idx:
        qcw = content_words(data["q_anchor"][qi])
        if not qcw:
            continue
        gcw = content_words(corpus_texts[data["q_gold_idx"][qi]])
        gold_cont.append(len(qcw & gcw) / len(qcw))
        # 5 random distractor articles
        for _ in range(5):
            r = rng.randrange(N)
            while r == data["q_gold_idx"][qi]:
                r = rng.randrange(N)
            rcw = content_words(corpus_texts[r])
            rand_cont.append(len(qcw & rcw) / len(qcw))
    return {
        "n_sampled": len(idx),
        "gold_containment": quartiles(gold_cont),
        "random_containment": quartiles(rand_cont),
        "gold_minus_random_mean": float(statistics.mean(gold_cont) - statistics.mean(rand_cont)),
        "_sample_idx": idx,
        "_gold_cont_by_sample": {qi: gc for qi, gc in zip(idx, gold_cont)} if len(idx) == len(gold_cont) else None,
    }


def build_tfidf(corpus_texts):
    vec = TfidfVectorizer(sublinear_tf=True, lowercase=True, dtype=np.float32)
    Xc = vec.transform(corpus_texts) if hasattr(vec, "vocabulary_") else vec.fit_transform(corpus_texts)
    return vec, Xc


def tfidf_retrieve(data, k=K, batch=512):
    """(2) TF-IDF word-level, sublinear_tf, over FULL corpus. Return per-query rank-of-gold."""
    corpus_texts = data["corpus_texts"]
    vec = TfidfVectorizer(sublinear_tf=True, lowercase=True, dtype=np.float32)
    Xc = vec.fit_transform(corpus_texts)  # already L2-normalised by TfidfVectorizer
    Xq = vec.transform(data["q_anchor"])
    gold = np.asarray(data["q_gold_idx"])
    n_q = Xq.shape[0]
    top1 = np.zeros(n_q, dtype=bool)
    recallk = np.zeros(n_q, dtype=bool)
    rank_of_gold = np.full(n_q, -1, dtype=np.int32)  # -1 = not in top-k
    for s in range(0, n_q, batch):
        e = min(s + batch, n_q)
        sims = (Xq[s:e] @ Xc.T).toarray()  # (b, N) cosine (both L2-normed)
        # top-k indices per row
        part = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        for i in range(e - s):
            row = sims[i]
            topk = part[i][np.argsort(-row[part[i]])]
            g = gold[s + i]
            top1[s + i] = topk[0] == g
            hit = np.where(topk == g)[0]
            if hit.size:
                recallk[s + i] = True
                rank_of_gold[s + i] = int(hit[0])
    return {
        "vec": vec,
        "Xc": Xc,
        "top1": top1,
        "recallk": recallk,
        "rank_of_gold": rank_of_gold,
    }


def analysis_neardup(data, vec, Xc, thresh=0.8, batch=256):
    """(4) fraction of GOLD articles with a non-self corpus article at TF-IDF cosine > thresh."""
    gold_idx = sorted(set(data["q_gold_idx"]))
    g = np.asarray(gold_idx)
    Xg = Xc[g]
    N = Xc.shape[0]
    has_dup = np.zeros(len(g), dtype=bool)
    max_nb = np.zeros(len(g), dtype=np.float32)
    for s in range(0, len(g), batch):
        e = min(s + batch, len(g))
        sims = (Xg[s:e] @ Xc.T).toarray()  # (b, N)
        for i in range(e - s):
            row = sims[i]
            row[g[s + i]] = -1.0  # exclude self
            m = float(row.max())
            max_nb[s + i] = m
            has_dup[s + i] = m > thresh
    return {
        "n_gold_articles": int(len(g)),
        "frac_gold_with_neardup_gt0.8": float(has_dup.mean()),
        "frac_gold_with_neardup_gt0.9": float((max_nb > 0.9).mean()),
        "frac_gold_with_neardup_gt0.95": float((max_nb > 0.95).mean()),
        "mean_max_neighbor_cos": float(max_nb.mean()),
        "median_max_neighbor_cos": float(np.median(max_nb)),
    }


def analysis_difficulty(data, tf, overlap):
    """(5) characterize TF-IDF failures (gold not in top-k)."""
    rank = tf["rank_of_gold"]
    fail = rank < 0
    q_len = np.array([len(norm(a).split()) for a in data["q_anchor"]])
    codes = np.array(data["q_code"])
    cited = np.array([bool(CITE.search(a)) for a in data["q_anchor"]])
    corpus_texts = data["corpus_texts"]
    cont = np.array(
        [
            (lambda qcw: (len(qcw & content_words(corpus_texts[gi])) / len(qcw)) if qcw else 0.0)(
                content_words(a)
            )
            for a, gi in zip(data["q_anchor"], data["q_gold_idx"])
        ]
    )

    def stats_mask(mask):
        return {
            "count": int(mask.sum()),
            "frac_of_all": float(mask.mean()),
            "mean_query_len": float(q_len[mask].mean()) if mask.any() else None,
            "mean_gold_containment": float(cont[mask].mean()) if mask.any() else None,
            "frac_query_cites_article": float(cited[mask].mean()) if mask.any() else None,
        }

    # code distribution among failures vs overall
    from collections import Counter

    fail_codes = Counter(codes[fail].tolist())
    all_codes = Counter(codes.tolist())
    code_fail_rate = {
        c: {"fail": fail_codes.get(c, 0), "total": all_codes[c], "rate": fail_codes.get(c, 0) / all_codes[c]}
        for c in sorted(all_codes, key=lambda c: -all_codes[c])
    }
    return {
        "tfidf_fail_at_k": stats_mask(fail),
        "tfidf_success_at_k": stats_mask(~fail),
        "overall_mean_query_len": float(q_len.mean()),
        "overall_frac_cites_article": float(cited.mean()),
        "code_fail_rate": code_fail_rate,
    }


def z(p):
    from statistics import NormalDist

    return NormalDist().inv_cdf(p)


def mde(n, sd, alpha=0.05, power=0.8):
    if n <= 0:
        return float("inf")
    return (z(1 - alpha / 2) + z(power)) * sd / math.sqrt(n)


def run_split(name, data, rng):
    res = {"n_queries": len(data["q_anchor"]), "n_corpus": len(data["corpus_texts"])}
    if "n_heldout_pairs" in data:
        res["n_heldout_pairs"] = data["n_heldout_pairs"]

    ov = analysis_overlap(data, rng, N_SAMPLE)
    sample_idx = ov.pop("_sample_idx")
    gold_cont_by_sample = ov.pop("_gold_cont_by_sample")
    res["overlap"] = ov

    tf = tfidf_retrieve(data)
    res["tfidf_ceiling"] = {
        "full_set": {
            "n": int(len(tf["top1"])),
            "top1_acc": float(tf["top1"].mean()),
            "recall_at_10": float(tf["recallk"].mean()),
            "fail_rate_at_10": float((tf["rank_of_gold"] < 0).mean()),
        },
        "sampled_500": {
            "n": len(sample_idx),
            "top1_acc": float(tf["top1"][sample_idx].mean()),
            "recall_at_10": float(tf["recallk"][sample_idx].mean()),
        },
    }
    res["neardup"] = analysis_neardup(data, tf["vec"], tf["Xc"])
    res["difficulty"] = analysis_difficulty(data, tf, ov)

    # false-negative trap check: among TF-IDF failures, how often is the top-1 retrieved
    # article a >0.8 near-dup of the gold (i.e. a plausible UNLABELED positive)?
    return res, tf


def main():
    random.seed(SEED)
    out = {"meta": {"seed": SEED, "n_sample": N_SAMPLE, "k": K}}

    print("building Track-B ...")
    tb = build_trackb()
    print("building tax ...")
    tx = build_tax()

    rng = random.Random(SEED)
    print("Track-B analyses ...")
    out["trackb"], tf_tb = run_split("trackb", tb, rng)
    rng = random.Random(SEED)
    print("tax analyses ...")
    out["tax"], _ = run_split("tax", tx, rng)

    # ---- v2 sizing projection (Track-B) --------------------------------------
    heldout = tb["n_heldout_pairs"]
    fail_rate = out["trackb"]["tfidf_ceiling"]["full_set"]["fail_rate_at_10"]
    # If v2 uses ALL held-out pairs as queries then keeps only BM25/TF-IDF failures at k:
    surviving = int(round(heldout * fail_rate))
    proj = {"heldout_pairs_total": heldout, "tfidf_fail_rate_at_10": fail_rate, "surviving_hard_queries": surviving}
    for sd in (0.30, 0.35, 0.40, 0.45):
        proj[f"mde_at_survivor_sd{sd}"] = mde(surviving, sd)
    # For reference, MDE if we DON'T filter (use all held-out as queries):
    for sd in (0.30, 0.40):
        proj[f"mde_all_heldout_sd{sd}"] = mde(heldout, sd)
    out["v2_projection"] = proj

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")

    # compact console summary
    def line(s):
        print(s)

    for split in ("trackb", "tax"):
        r = out[split]
        line(f"\n===== {split.upper()} (n_q={r['n_queries']}, corpus={r['n_corpus']}) =====")
        o = r["overlap"]
        line(f"  content-word containment  gold: mean={o['gold_containment']['mean']:.3f} "
             f"median={o['gold_containment']['median']:.3f} "
             f"[q1={o['gold_containment']['q1']:.3f} q3={o['gold_containment']['q3']:.3f}]")
        line(f"  content-word containment  rand: mean={o['random_containment']['mean']:.3f} "
             f"median={o['random_containment']['median']:.3f}")
        line(f"  gold - random (mean)          : {o['gold_minus_random_mean']:.3f}")
        t = r["tfidf_ceiling"]["full_set"]
        line(f"  TF-IDF (zero-ML) full-set     : top1={t['top1_acc']:.3f}  R@10={t['recall_at_10']:.3f}  "
             f"fail@10={t['fail_rate_at_10']:.3f}")
        nd = r["neardup"]
        line(f"  near-dup gold (cos>0.8)       : {nd['frac_gold_with_neardup_gt0.8']:.3f}  "
             f"(>0.9={nd['frac_gold_with_neardup_gt0.9']:.3f})  mean_maxnb={nd['mean_max_neighbor_cos']:.3f}")
        d = r["difficulty"]["tfidf_fail_at_k"]
        line(f"  TF-IDF failures @10           : {d['count']} ({d['frac_of_all']:.3f})  "
             f"mean_qlen={d['mean_query_len']}  cites_art={d['frac_query_cites_article']}")

    p = out["v2_projection"]
    line(f"\n===== V2 PROJECTION (Track-B) =====")
    line(f"  held-out pairs total          : {p['heldout_pairs_total']}")
    line(f"  TF-IDF fail-rate@10           : {p['tfidf_fail_rate_at_10']:.3f}")
    line(f"  surviving hard queries        : {p['surviving_hard_queries']}")
    line(f"  MDE @survivor sd=0.35         : {p['mde_at_survivor_sd0.35']:.4f}")
    line(f"  MDE @survivor sd=0.40         : {p['mde_at_survivor_sd0.4']:.4f}")


if __name__ == "__main__":
    main()
