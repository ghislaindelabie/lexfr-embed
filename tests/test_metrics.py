"""Hermetic unit tests for the statistics helpers (numpy only — no network/torch).

These back the Axis-1 bootstrap CI + the per-task MDE note in the scorecard.
"""

from lexfr_embed.metrics import bootstrap_ci, min_detectable_effect, paired_delta_ci


def test_bootstrap_ci_zero_width_on_constant():
    low, high = bootstrap_ci([0.3] * 50, n_boot=200, seed=1)
    assert abs(high - low) < 1e-9
    assert abs(low - 0.3) < 1e-9


def test_bootstrap_ci_wider_for_more_spread():
    tight = bootstrap_ci([0.5, 0.5, 0.5, 0.51, 0.49] * 10, n_boot=500, seed=1)
    wide = bootstrap_ci([0.0, 1.0, 0.0, 1.0, 0.5] * 10, n_boot=500, seed=1)
    assert (wide[1] - wide[0]) > (tight[1] - tight[0])


def test_bootstrap_ci_seed_reproducible():
    a = bootstrap_ci([0.1, 0.9, 0.5, 0.3, 0.7], n_boot=500, seed=7)
    b = bootstrap_ci([0.1, 0.9, 0.5, 0.3, 0.7], n_boot=500, seed=7)
    assert a == b


def test_paired_delta_ci_straddles_zero_when_no_change():
    mean, low, high = paired_delta_ci([0.3, 0.5, 0.4], [0.3, 0.5, 0.4], n_boot=300, seed=2)
    assert abs(mean) < 1e-9
    assert low <= 0.0 <= high


def test_paired_delta_ci_excludes_zero_when_clearly_improved():
    before = [0.2, 0.3, 0.25, 0.28, 0.22] * 6
    after = [0.4, 0.5, 0.45, 0.48, 0.42] * 6
    mean, low, high = paired_delta_ci(before, after, n_boot=500, seed=3)
    assert mean > 0
    assert low > 0  # CI excludes zero -> a real improvement


def test_mde_shrinks_as_n_grows():
    assert min_detectable_effect(50, 0.2) > min_detectable_effect(500, 0.2)
    assert min_detectable_effect(200, 0.2) > 0
