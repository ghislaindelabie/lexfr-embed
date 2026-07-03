"""Phase-1 scorecard renderer — pure string formatting, hermetic.

Enforces the trust rules in prose: report the paired CI + whether it excludes zero, flag
any sub-MDE delta as 'within noise', mark retention regressions (delta < -MDE), and surface
the frozen partition hashes. `scripts/run_phase1.py` calls this with measured numbers.
"""

from __future__ import annotations


def format_scorecard(headline: dict, retention: list[dict], partition_hashes: dict) -> str:
    hl = headline
    ci = hl.get("ci", (float("nan"), float("nan")))
    excludes_zero = (ci[0] > 0) or (ci[1] < 0)
    within_noise = abs(hl.get("delta", 0.0)) < hl.get("mde", 0.0)

    lines = [
        "# Phase-1 scorecard",
        "",
        f"## Axis 1 — headline: {hl.get('metric')} (BSARD, within-config paired before→after, n={hl.get('n')})",
        "",
        f"- {hl.get('metric')}: **{hl.get('before'):.3f} → {hl.get('after'):.3f}** (Δ {hl.get('delta'):+.3f}); "
        f"95% paired-bootstrap CI [{ci[0]:+.3f}, {ci[1]:+.3f}] — "
        + ("**excludes zero**" if excludes_zero else "includes zero (not significant)"),
        f"- MDE (n={hl.get('n')}): ±{hl.get('mde'):.3f}"
        + ("  → ⚠️ **within noise — not evidence of improvement**" if within_noise else ""),
        "",
        "## Axis 3 — general-language retention guard",
        "",
    ]

    regressed = [r for r in retention if r.get("delta", 0.0) < -r.get("mde", 0.0)]
    if retention:
        lines += ["| Task | before | after | Δ | ±MDE | status |", "|---|---:|---:|---:|---:|:--:|"]
        for r in retention:
            reg = r.get("delta", 0.0) < -r.get("mde", 0.0)
            lines.append(
                f"| {r['task']} | {r['before']:.3f} | {r['after']:.3f} | {r['delta']:+.3f} | "
                f"±{r['mde']:.3f} | {'⚠️ REGRESSED' if reg else 'ok'} |"
            )
        verdict = (
            "❌ **FAIL — regression detected** beyond ±MDE on: " + ", ".join(r["task"] for r in regressed)
            if regressed
            else "✅ **no regression detectable above ±MDE**"
        )
        lines += ["", f"**Verdict:** {verdict}", ""]
    else:
        lines += ["_(retention not run)_", ""]

    lines += ["## Partition hashes (frozen before mining)", ""]
    if partition_hashes:
        lines += [f"- `{name}`: `{h}`" for name, h in partition_hashes.items()]
    else:
        lines.append("_(none recorded)_")

    lines += [
        "",
        "## Limitations",
        "",
        "- BSARD is a **Belgian + lay + LLM-synthetic-train transfer proxy** — not a French-professional claim.",
        "- Any Δ below its MDE is within noise; every inferred/external-literature number is labelled as such.",
    ]
    return "\n".join(lines)
