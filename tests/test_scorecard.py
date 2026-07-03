"""Hermetic tests for the scorecard renderer (pure string formatting).

Enforces the trust rules: report the CI + whether it excludes zero, flag sub-MDE deltas
as 'within noise', mark retention regressions, and print the partition hashes.
"""

from lexfr_embed.scorecard import format_scorecard

_HEADLINE_REAL = {
    "metric": "NDCG@10",
    "before": 0.242,
    "after": 0.307,
    "delta": 0.065,
    "ci": (0.02, 0.11),
    "mde": 0.035,
    "n": 222,
}


def test_renders_metric_delta_and_hashes():
    md = format_scorecard(_HEADLINE_REAL, retention=[], partition_hashes={"bsard_gold": "abc123"})
    assert "NDCG@10" in md
    assert "abc123" in md
    assert "0.307" in md


def test_flags_within_noise_when_delta_below_mde():
    hl = {**_HEADLINE_REAL, "after": 0.247, "delta": 0.005, "ci": (-0.02, 0.03), "mde": 0.035}
    md = format_scorecard(hl, retention=[], partition_hashes={})
    assert "within noise" in md.lower()


def test_flags_ci_excludes_zero_when_it_does():
    md = format_scorecard(_HEADLINE_REAL, retention=[], partition_hashes={})
    assert "excludes zero" in md.lower()


def test_marks_retention_regression():
    ret = [{"task": "SciFact", "before": 0.50, "after": 0.40, "delta": -0.10, "mde": 0.02}]
    md = format_scorecard(_HEADLINE_REAL, retention=ret, partition_hashes={})
    assert "regress" in md.lower() or "FAIL" in md


def test_retention_within_tolerance_is_not_flagged_regressed():
    ret = [{"task": "AlloprofRetrieval", "before": 0.50, "after": 0.49, "delta": -0.01, "mde": 0.03}]
    md = format_scorecard(_HEADLINE_REAL, retention=ret, partition_hashes={})
    assert "no regression" in md.lower()
