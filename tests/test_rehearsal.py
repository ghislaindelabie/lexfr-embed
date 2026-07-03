"""Rehearsal-floor mixing (anti-forgetting): pure helpers only (no network).

The retention guard caught an EN-financial-QA regression because training saw legal pairs
ONLY. The fix is to mix a small fraction of general (query, positive) pairs into training —
the `rehearsal_frac` floor that lived in config.py but was never wired in. These tests pin
the two pure pieces: how many rehearsal pairs to add, and how they are merged.
"""

from lexfr_embed.data.rehearsal import mix_rehearsal, rehearsal_count


def test_rehearsal_count_hits_the_target_fraction():
    # 7 rehearsal pairs added to 93 legal -> 7/100 = exactly the 0.07 floor
    assert rehearsal_count(93, 0.07) == 7


def test_rehearsal_count_zero_when_frac_is_zero_or_no_legal():
    assert rehearsal_count(10_000, 0.0) == 0
    assert rehearsal_count(0, 0.07) == 0


def test_mix_adds_all_rehearsal_pairs_and_tags_them():
    legal = [{"anchor": f"q{i}", "positive": f"a{i}", "code": "civil"} for i in range(3)]
    rehearsal = [{"anchor": "en q", "positive": "en passage"}, {"anchor": "fr q", "positive": "fr passage"}]

    mixed = mix_rehearsal(legal, rehearsal, seed=42)

    assert len(mixed) == 5  # nothing dropped
    # every legal pair survives unchanged
    for p in legal:
        assert p in mixed
    # rehearsal pairs are tagged so they are auditable / excludable downstream
    tagged = [p for p in mixed if p.get("code") == "rehearsal"]
    assert len(tagged) == 2
    assert {p["anchor"] for p in tagged} == {"en q", "fr q"}


def test_mix_is_deterministic_for_a_seed():
    legal = [{"anchor": f"q{i}", "positive": f"a{i}", "code": "civil"} for i in range(5)]
    rehearsal = [{"anchor": f"g{i}", "positive": f"p{i}"} for i in range(3)]

    first = mix_rehearsal(legal, rehearsal, seed=7)
    second = mix_rehearsal(legal, rehearsal, seed=7)

    assert [p["anchor"] for p in first] == [p["anchor"] for p in second]


def test_mix_with_no_rehearsal_returns_legal_unchanged():
    legal = [{"anchor": "q", "positive": "a", "code": "civil"}]
    assert mix_rehearsal(legal, [], seed=1) == legal
