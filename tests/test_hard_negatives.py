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
