"""Hermetic tests for the pure hard-negative helper (no network/torch).

The mining call itself (sentence_transformers.util.mine_hard_negatives) is a thin
integration layer exercised by the marked smoke run.
"""

import pytest

from lexfr_embed.data.hard_negatives import pairs_to_anchor_positive_dict


def test_maps_pairs_to_two_columns():
    pairs = [
        {"anchor": "q1", "positive": "a1", "code": "civil"},
        {"anchor": "q2", "positive": "a2", "code": "travail"},
    ]
    d = pairs_to_anchor_positive_dict(pairs)
    assert d == {"anchor": ["q1", "q2"], "positive": ["a1", "a2"]}
    assert "code" not in d  # 'code' is dropped — MNRL wants only anchor/positive


def test_raises_on_empty():
    with pytest.raises(ValueError):
        pairs_to_anchor_positive_dict([])


def test_raises_on_missing_key():
    with pytest.raises((KeyError, ValueError)):
        pairs_to_anchor_positive_dict([{"anchor": "q1"}])  # missing 'positive'


def test_mine_forwards_cross_encoder_and_max_score_for_denoising(monkeypatch):
    """A1: given a cross-encoder + threshold, mine() forwards them so the false-negative
    filter runs on the RERANKER's scores (RocketQA-style denoising), not just embedder sim."""
    import sentence_transformers.util as st_util

    from lexfr_embed.data import hard_negatives as hn

    captured = {}

    def fake_mine(dataset, model, **kwargs):
        captured.update(kwargs)
        return "TRIPLETS"

    monkeypatch.setattr(st_util, "mine_hard_negatives", fake_mine)
    sentinel_ce = object()
    out = hn.mine([{"anchor": "q", "positive": "p"}], model=object(), cross_encoder=sentinel_ce, max_score=0.7)
    assert out == "TRIPLETS"
    assert captured["cross_encoder"] is sentinel_ce
    assert captured["max_score"] == 0.7


def test_mine_without_cross_encoder_stays_embedder_only(monkeypatch):
    """Backward-compat: no denoising params leak when cross_encoder is None (campaign baseline)."""
    import sentence_transformers.util as st_util

    from lexfr_embed.data import hard_negatives as hn

    captured = {}
    monkeypatch.setattr(st_util, "mine_hard_negatives", lambda dataset, model, **kw: captured.update(kw) or "T")
    hn.mine([{"anchor": "q", "positive": "p"}], model=object())
    assert "cross_encoder" not in captured
    assert "max_score" not in captured
