"""Statistics helpers for the Phase-1 scorecard — bootstrap CIs + minimum detectable effect.

Pure numpy + stdlib; hermetic. Feeds the Axis-1 paired before->after CI and the per-task
MDE note (a delta below its MDE is flagged 'within noise', never claimed as improvement).
"""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist

import numpy as np


def bootstrap_ci(scores, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the MEAN of per-query scores. Returns (low, high)."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_delta_ci(
    before, after, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Paired bootstrap of the per-query delta (after - before). Returns (mean_delta, low, high)."""
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    if b.shape != a.shape or b.size == 0:
        raise ValueError("before/after must be the same non-empty length")
    deltas = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(n_boot, deltas.size))
    boot_means = deltas[idx].mean(axis=1)
    low = float(np.quantile(boot_means, alpha / 2))
    high = float(np.quantile(boot_means, 1 - alpha / 2))
    return (float(deltas.mean()), low, high)


def min_detectable_effect(n: int, sd: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Approx two-sided MDE for a paired mean: (z_{1-alpha/2} + z_power) * sd / sqrt(n).

    z-quantiles from stdlib NormalDist (exact, no scipy). At n=222 / sd~0.3 this lands the
    ~0.03-0.04 floor the blueprint cites; deltas below it are 'within noise'.
    """
    if n <= 0:
        return float("inf")
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    return (z_alpha + z_power) * float(sd) / sqrt(n)
