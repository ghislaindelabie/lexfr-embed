"""General-capability retention check — guard against catastrophic forgetting.

Fine-tuning an embedder contrastively on a narrow *legal* distribution can degrade
its general French / English behaviour (catastrophic forgetting / representation
collapse toward legalese). This module runs a SMALL, fixed, strictly **non-legal**
MTEB(fr) + BEIR subset BEFORE and AFTER training and reports the per-task deltas with
a PASS/FAIL verdict, so a Phase-1 run can prove the model kept its general capacities.

Design: the pure helpers (`task_names`, `build_report`, `format_markdown`) import
nothing heavy and are unit-tested. `run_mteb()` is a thin integration layer that
imports `mteb` lazily (optional `eval` extra). BSARD / the French legal eval set are
handled in `evaluate.py`; the retention suite here is deliberately legal-free.
"""

from __future__ import annotations

from dataclasses import dataclass

# A general score may dip by at most this (absolute: NDCG@10 / Spearman / V-measure
# points) before we call it a regression. Gains are never flagged. Tune per appetite.
DEFAULT_TOLERANCE = 0.02


@dataclass(frozen=True)
class RetentionTask:
    name: str  # MTEB task name
    family: str  # "retrieval" | "sts" | "clustering"
    language: str  # "fr" | "en"
    main_metric: str  # the score MTEB reports as main (informational)
    note: str = ""


# Small, non-legal suite. Retrieval-first (the capability the product actually uses);
# a little EN to confirm BGE-M3's cross-lingual retention; STS as a cheap geometry
# sanity check; one clustering task as an optional extra signal.
GENERAL_TASKS: list[RetentionTask] = [
    RetentionTask("AlloprofRetrieval", "retrieval", "fr", "ndcg_at_10", "FR QA retrieval (education)"),
    # NB: SyntecRetrieval was dropped — MTEB tags it domains=['Legal'] (Syntec collective agreement),
    # so it is NOT valid in a strictly NON-legal retention guard (a legal fine-tune could *improve* it).
    RetentionTask("MintakaRetrieval", "retrieval", "fr", "ndcg_at_10", "FR open-domain QA retrieval"),
    RetentionTask("SciFact", "retrieval", "en", "ndcg_at_10", "EN scientific-claim retrieval (BEIR)"),
    RetentionTask("FiQA2018", "retrieval", "en", "ndcg_at_10", "EN financial QA retrieval (BEIR)"),
    RetentionTask("STSBenchmarkMultilingualSTS", "sts", "fr", "cosine_spearman", "FR STS (semantic geometry)"),
    RetentionTask("SICKFr", "sts", "fr", "cosine_spearman", "FR STS"),
    RetentionTask("AlloProfClusteringS2S", "clustering", "fr", "v_measure", "FR clustering (optional)"),
]


@dataclass(frozen=True)
class RetentionRow:
    name: str
    family: str
    language: str
    before: float
    after: float
    delta: float
    regressed: bool


def task_names(tasks: list[RetentionTask] | None = None) -> list[str]:
    return [t.name for t in (tasks or GENERAL_TASKS)]


def build_report(
    before: dict[str, float],
    after: dict[str, float],
    tolerance: float = DEFAULT_TOLERANCE,
    tasks: list[RetentionTask] | None = None,
) -> tuple[list[RetentionRow], bool]:
    """Pair before/after scores into rows + an overall PASS/FAIL verdict.

    `before` / `after` map task-name -> main score. Only tasks present in BOTH are
    scored (a task MTEB failed to run is skipped, not counted against the model). A
    task `regressed` if its score dropped by more than `tolerance`. `passed` is True
    iff no scored task regressed.
    """
    by_name = {t.name: t for t in (tasks or GENERAL_TASKS)}
    rows: list[RetentionRow] = []
    for name, task in by_name.items():
        if name in before and name in after:
            b, a = float(before[name]), float(after[name])
            delta = a - b
            rows.append(RetentionRow(name, task.family, task.language, b, a, delta, delta < -tolerance))
    rows.sort(key=lambda r: (r.family, r.language, r.name))
    passed = not any(r.regressed for r in rows)
    return rows, passed


def missing_tasks(
    before: dict[str, float],
    after: dict[str, float],
    tasks: list[RetentionTask] | None = None,
) -> list[str]:
    """Tasks in the expected suite that are absent from either run (for honest reporting)."""
    return [n for n in task_names(tasks) if n not in before or n not in after]


def format_markdown(rows: list[RetentionRow], passed: bool, tolerance: float = DEFAULT_TOLERANCE) -> str:
    lines = [
        f"### General-capability retention (tolerance ±{tolerance:.3f})",
        "",
        "| Task | Family | Lang | Before | After | Δ | Status |",
        "|---|---|---|---:|---:|---:|:--:|",
    ]
    for r in rows:
        status = "⚠️ REGRESSED" if r.regressed else "ok"
        lines.append(
            f"| {r.name} | {r.family} | {r.language} | {r.before:.4f} | {r.after:.4f} | {r.delta:+.4f} | {status} |"
        )
    verdict = (
        "✅ PASS — general capacities retained"
        if passed
        else "❌ FAIL — over-specialisation / catastrophic forgetting detected"
    )
    lines += ["", f"**Verdict: {verdict}**"]
    return "\n".join(lines)


def extract_main_score(task_result) -> float | None:
    """Best-effort main score for one MTEB TaskResult (the API has churned across versions)."""
    getter = getattr(task_result, "get_score", None)
    if callable(getter):
        try:
            return float(getter())
        except Exception:  # noqa: BLE001 - any failure falls through to the dict probe
            pass
    scores = getattr(task_result, "scores", None)
    if isinstance(scores, dict):  # {split: [{"main_score": x, ...}, ...]}
        for split_results in scores.values():
            if isinstance(split_results, list):
                for entry in split_results:
                    if isinstance(entry, dict) and entry.get("main_score") is not None:
                        try:
                            return float(entry["main_score"])
                        except (TypeError, ValueError):
                            pass
    return None


def run_mteb(
    model,
    tasks: list[str] | None = None,
    output_folder: str | None = None,
    batch_size: int = 32,
) -> dict[str, float]:
    """Run the retention subset on a loaded model -> {task_name: main_score}.

    `model` is anything MTEB accepts (a SentenceTransformer works). Imports `mteb`
    lazily so the pure helpers stay dependency-free; install via the `eval` extra.
    """
    try:
        import mteb
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise SystemExit("mteb not installed — run:  uv pip install -e '.[eval]'") from e

    names = tasks or task_names()
    evaluation = mteb.MTEB(tasks=mteb.get_tasks(tasks=names))
    results = evaluation.run(model, output_folder=output_folder, encode_kwargs={"batch_size": batch_size})
    out: dict[str, float] = {}
    for tr in results:
        name = getattr(tr, "task_name", None) or getattr(getattr(tr, "task", None), "metadata", None)
        score = extract_main_score(tr)
        if isinstance(name, str) and score is not None:
            out[name] = score
    return out
