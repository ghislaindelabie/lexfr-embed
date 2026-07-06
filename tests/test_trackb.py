"""Hermetic tests for the Track-B held-out eval builder (LegalKit) — no network/torch.

Track-B is a large, powered, IN-DISTRIBUTION FR-legal retrieval eval built from LegalKit
pairs held out (leak-free) from every trained subset. It has the SAME (queries, corpus,
relevant) shape as evaluate.load_bsard, so eval_extra's powered/rerank/matryoshka modes
consume it directly. These tests pin the pure selection/dedup/qrels logic with fakes.
"""

from lexfr_embed.data.trackb import build_holdout, build_trackb_eval, pair_id


def _p(anchor, positive, code="Code civil"):
    return {"anchor": anchor, "positive": positive, "code": code}


def test_pair_id_stable_and_text_based():
    assert pair_id(_p("q1", "art A")) == pair_id(_p("q1", "art A"))
    assert pair_id(_p("q1", "art A")) != pair_id(_p("q1", "art B"))


def test_pair_id_case_and_whitespace_insensitive():
    # must match the loader's dedup key so held-out is genuinely leak-free
    assert pair_id(_p("Q1", "art A")) == pair_id(_p("  q1 ", "art   a"))


def test_build_holdout_excludes_trained_pairs():
    all_pairs = [_p("q1", "A"), _p("q2", "B"), _p("q3", "C")]
    trained = {pair_id(_p("q1", "A")), pair_id(_p("q2", "B"))}
    held = build_holdout(all_pairs, trained)
    ids = {pair_id(p) for p in held}
    assert ids == {pair_id(_p("q3", "C"))}
    assert ids.isdisjoint(trained)


def test_trackb_eval_shape_and_every_relevant_in_corpus():
    held = [_p("q1", "A"), _p("q2", "B"), _p("q3", "C")]
    queries, corpus, relevant = build_trackb_eval(held, n_queries=3, seed=42)
    assert set(queries) == set(relevant)
    for rel in relevant.values():
        assert len(rel) >= 1
        assert rel <= set(corpus)  # gold cid must exist in the corpus


def test_trackb_dedup_articles_and_multilabel():
    # two queries share the SAME article text -> one corpus entry, both relevant to it
    held = [_p("q1", "SAME"), _p("q2", "SAME"), _p("q3", "OTHER")]
    _, corpus, relevant = build_trackb_eval(held, n_queries=3, seed=42)
    assert len(corpus) == 2
    same_cid = next(c for c, t in corpus.items() if t == "SAME")
    pointing = [qid for qid, rel in relevant.items() if same_cid in rel]
    assert len(pointing) == 2


def test_trackb_corpus_keeps_all_heldout_articles_as_distractors():
    # sampling fewer queries must NOT shrink the corpus (realistic distractors)
    held = [_p(f"q{i}", f"A{i}") for i in range(20)]
    queries, corpus, _ = build_trackb_eval(held, n_queries=5, seed=1)
    assert len(queries) == 5
    assert len(corpus) == 20


def test_trackb_deterministic_with_seed():
    held = [_p(f"q{i}", f"A{i}") for i in range(50)]
    qa, _, _ = build_trackb_eval(held, n_queries=10, seed=7)
    qb, _, _ = build_trackb_eval(held, n_queries=10, seed=7)
    assert set(qa.values()) == set(qb.values())


def test_trackb_none_uses_all_heldout_queries():
    held = [_p(f"q{i}", f"A{i}") for i in range(12)]
    queries, corpus, _ = build_trackb_eval(held, n_queries=None, seed=3)
    assert len(queries) == 12
    assert len(corpus) == 12
