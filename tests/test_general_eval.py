"""Hermetic unit tests for the general-capability retention logic (no network, no torch).

Covers the pure helpers only; `run_mteb` is an integration layer exercised manually
via `scripts/eval_general.py`.
"""

from lexfr_embed.general_eval import (
    GENERAL_TASKS,
    build_report,
    format_markdown,
    missing_tasks,
    task_names,
)


def test_suite_is_unique_and_strictly_nonlegal():
    names = task_names()
    assert len(names) == len(set(names))  # no duplicates
    assert "AlloprofRetrieval" in names
    # The retention suite must be legal-free — BSARD / the FR legal set live in evaluate.py.
    assert not any("bsard" in n.lower() or "legal" in n.lower() or "lleqa" in n.lower() for n in names)
    # Retrieval is the deployed capability → it must dominate the suite.
    assert sum(t.family == "retrieval" for t in GENERAL_TASKS) >= 4


def test_pass_when_within_tolerance():
    before = {"AlloprofRetrieval": 0.50, "SICKFr": 0.70}
    after = {"AlloprofRetrieval": 0.49, "SICKFr": 0.71}  # -0.01 within ±0.02
    rows, passed = build_report(before, after)
    assert passed
    assert all(not r.regressed for r in rows)


def test_flags_regression_beyond_tolerance():
    rows, passed = build_report({"AlloprofRetrieval": 0.50}, {"AlloprofRetrieval": 0.40})  # -0.10
    assert not passed
    assert rows[0].regressed
    assert rows[0].delta < 0


def test_gains_are_never_flagged():
    rows, passed = build_report({"SciFact": 0.40}, {"SciFact": 0.55})
    assert passed
    assert not rows[0].regressed


def test_only_tasks_present_in_both_are_scored():
    before = {"AlloprofRetrieval": 0.5, "SciFact": 0.4}
    after = {"AlloprofRetrieval": 0.5}  # SciFact missing from the after-run
    rows, _ = build_report(before, after)
    assert {r.name for r in rows} == {"AlloprofRetrieval"}


def test_missing_tasks_reported():
    miss = missing_tasks({"AlloprofRetrieval": 0.5}, {"AlloprofRetrieval": 0.5})
    assert "SciFact" in miss
    assert "AlloprofRetrieval" not in miss


def test_tolerance_is_configurable():
    before = {"SICKFr": 0.70}
    after = {"SICKFr": 0.66}  # -0.04
    _, passed_loose = build_report(before, after, tolerance=0.05)
    _, passed_strict = build_report(before, after, tolerance=0.02)
    assert passed_loose
    assert not passed_strict


def test_format_markdown_renders_verdict_and_flag():
    rows, passed = build_report({"AlloprofRetrieval": 0.5}, {"AlloprofRetrieval": 0.4})
    md = format_markdown(rows, passed)
    assert "REGRESSED" in md
    assert "FAIL" in md
    assert "| AlloprofRetrieval |" in md
