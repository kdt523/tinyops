"""Bootstrap CIs and paired significance tests. All paired tests need matching scenario ids."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats as sps


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 1234
) -> tuple[float, float, float]:
    """(mean, lo, hi), percentile method. For rates pass 0/1 values."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def _check_ids(ids_a: Sequence[str] | None, ids_b: Sequence[str] | None, n_a: int, n_b: int) -> None:
    if n_a != n_b:
        raise ValueError(f"paired samples differ in length: {n_a} vs {n_b}")
    if (ids_a is None) != (ids_b is None):
        raise ValueError("pass scenario ids for both samples or neither")
    if ids_a is not None and list(ids_a) != list(ids_b):
        raise ValueError("scenario ids do not match between the two samples (same order required)")


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 1000, alpha: float = 0.05,
    seed: int = 1234, ids_a: Sequence[str] | None = None, ids_b: Sequence[str] | None = None,
) -> tuple[float, float, float, float]:
    """(diff, lo, hi, p_two_sided) with diff = mean(b) - mean(a) over the same scenarios."""
    _check_ids(ids_a, ids_b, len(a), len(b))
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.size == 0:
        raise ValueError("paired_bootstrap_diff needs at least one pair")
    d = y - x
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    diffs = d[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_le = (np.sum(diffs <= 0) + 1) / (n_resamples + 1)
    p_ge = (np.sum(diffs >= 0) + 1) / (n_resamples + 1)
    return float(d.mean()), float(lo), float(hi), float(min(1.0, 2 * min(p_le, p_ge)))


def mcnemar(
    a_success: Sequence[bool], b_success: Sequence[bool],
    ids_a: Sequence[str] | None = None, ids_b: Sequence[str] | None = None,
) -> tuple[int, int, float, float]:
    """(b01, b10, statistic, p). b01 = a fails & b succeeds, b10 = a succeeds & b fails.

    Exact binomial test when b01 + b10 < 25 (statistic = min(b01, b10)), else chi-square with
    continuity correction.
    """
    _check_ids(ids_a, ids_b, len(a_success), len(b_success))
    a, b = np.asarray(a_success, dtype=bool), np.asarray(b_success, dtype=bool)
    b01 = int(np.sum(~a & b))
    b10 = int(np.sum(a & ~b))
    n = b01 + b10
    if n == 0:
        return b01, b10, 0.0, 1.0
    if n < 25:
        return b01, b10, float(min(b01, b10)), float(sps.binomtest(min(b01, b10), n, 0.5).pvalue)
    stat = (abs(b01 - b10) - 1) ** 2 / n
    return b01, b10, float(stat), float(sps.chi2.sf(stat, 1))
