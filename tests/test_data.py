"""Hermetic unit tests for the pure data helpers (no network, no torch).

These are real, passing tests — the TDD anchor for the data layer. Extend them as you
implement synthetic generation / hard-negative wrapping.
"""

from lexfr_embed.data.legalkit import _norm, dedup_pairs, stratify_by_code


def _p(anchor, positive, code="civil"):
    return {"anchor": anchor, "positive": positive, "code": code}


def test_norm_strips_accents_case_punct():
    assert _norm("Délai de PRESCRIPTION, art. 2224 !") == "delai de prescription art 2224"


def test_dedup_removes_normalised_duplicates():
    pairs = [
        _p("Délai de prescription ?", "Article 2224"),
        _p("delai de prescription", "article 2224"),  # same after _norm -> dropped
        _p("Résiliation du bail ?", "Article 1224"),
    ]
    out = dedup_pairs(pairs)
    assert len(out) == 2
    assert {p["positive"] for p in out} == {"Article 2224", "Article 1224"}


def test_dedup_keeps_distinct_pairs():
    pairs = [_p("q1", "a1"), _p("q2", "a2"), _p("q1", "a2")]
    assert len(dedup_pairs(pairs)) == 3


def test_stratify_balances_across_codes():
    pairs = [_p(f"q{i}", f"a{i}", "travail") for i in range(100)]
    pairs += [_p(f"c{i}", f"b{i}", "civil") for i in range(10)]
    out = stratify_by_code(pairs, target_n=20, seed=1)
    codes = [p["code"] for p in out]
    # despite travail being 10x larger, both codes should be represented near-evenly
    assert codes.count("civil") >= 8
    assert codes.count("travail") >= 8
    assert len(out) <= 20


def test_stratify_is_deterministic_with_seed():
    pairs = [_p(f"q{i}", f"a{i}", "civil") for i in range(50)]
    assert stratify_by_code(pairs, 10, seed=7) == stratify_by_code(pairs, 10, seed=7)


def test_stratify_empty_input():
    assert stratify_by_code([], target_n=10) == []
