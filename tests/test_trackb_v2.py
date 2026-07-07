"""Hermetic tests for the Track-B v2 builder (BM25/TF-IDF-hard, full-corpus, qrels-repaired).

v1 lesson (forensics 2026-07-06): a zero-ML TF-IDF retriever solves 90% of Track-B v1 at
rank 1 -> v1 measures lexical paraphrase matching. v2 keeps only queries the lexical
retriever FAILS at rank 1 on the full 52k corpus, drops pure-leak golds (byte-identical to
a trained positive), and multi-labels near-duplicate articles. Tiny fakes, CPU-only.
"""

from lexfr_embed.data.trackb_v2 import (
    build_v2_eval,
    drop_pure_leaks,
    lexical_top1_fail_mask,
    neardup_extra_golds,
)


def _p(anchor, positive, code="Code civil"):
    return {"anchor": anchor, "positive": positive, "code": code}


def test_drop_pure_leaks_removes_goldtext_seen_in_training():
    held = [_p("q1", "ARTICLE A"), _p("q2", "ARTICLE B")]
    kept = drop_pure_leaks(held, trained_texts={"ARTICLE A"})
    assert [p["anchor"] for p in kept] == ["q2"]


def test_lexical_top1_fail_mask_flags_only_hard_queries():
    corpus = [
        "le contrat de travail est rompu par démission du salarié",
        "la vente est parfaite dès accord sur la chose et le prix",
        "le juge peut prononcer la résolution du bail commercial",
    ]
    # q0 quotes doc0 almost verbatim -> lexical top-1 SUCCEEDS -> not hard (mask False)
    # q1 shares no vocabulary with its gold doc1 -> lexical FAILS -> hard (mask True)
    queries = ["rupture du contrat de travail par démission du salarié", "quand l'achat devient-il définitif ?"]
    mask = lexical_top1_fail_mask(queries, gold_idx=[0, 1], corpus_texts=corpus)
    assert mask.tolist() == [False, True]


def test_neardup_extra_golds_adds_only_similar_articles():
    corpus = [
        "le salarié dispose d'un délai de deux mois pour agir",
        "le salarié dispose d'un délai de deux mois pour agir devant le juge",  # near-dup of 0
        "les baux ruraux sont régis par le code rural",  # unrelated
    ]
    extra = neardup_extra_golds(gold_idx=[0], corpus_texts=corpus, threshold=0.8)
    assert 1 in extra[0]  # near-dup labelled as extra gold
    assert 2 not in extra[0]  # unrelated article NOT added


def test_build_v2_eval_shape_and_multilabel_qrels():
    held = [
        # hard: shares 1 content word with its gold but 5 with a decoy distractor -> lexical fail@1
        _p("comment le salarié peut contester la rupture ?", "le salarié dispose d'un délai de deux mois pour agir"),
        # easy: quotes its gold verbatim -> lexical success -> filtered out
        _p("rupture du contrat de travail par démission", "le contrat de travail est rompu par démission"),
    ]
    distractors = [
        "comment contester la rupture : le salarié peut contester la rupture devant le conseil",  # lexical decoy
        "le salarié dispose d'un délai de deux mois pour agir devant le juge",  # near-dup of the hard gold
        "les baux ruraux sont régis par le code rural",  # unrelated
    ]
    queries, corpus, relevant = build_v2_eval(held, distractor_texts=distractors, trained_texts=set())
    # corpus = golds + distractors, deduped
    assert len(corpus) == 5
    # only the hard query survives the fail@1 filter
    assert len(queries) == 1
    (qid,) = queries
    assert "contester" in queries[qid]
    # its near-dup article is multi-labelled as an extra gold; the decoy is NOT (below 0.8)
    gold_texts = {corpus[cid] for cid in relevant[qid]}
    assert gold_texts == {
        "le salarié dispose d'un délai de deux mois pour agir",
        "le salarié dispose d'un délai de deux mois pour agir devant le juge",
    }
