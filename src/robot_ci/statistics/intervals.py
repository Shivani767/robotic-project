"""Descriptive and inferential statistics for experiment results.

NumPy-only so the notebook backend stays dependency-light; SciPy is used
when available for exact intervals, with a normal approximation fallback.
Every interval reports its method and sample count alongside the number,
so a descriptive mean can never be mistaken for statistical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ProportionInterval",
    "bootstrap_mean_ci",
    "cohens_h",
    "mean_ci",
    "proportion_ci",
    "welch_t",
]


@dataclass(frozen=True)
class ProportionInterval:
    """Success-rate estimate with an uncertainty interval."""

    rate: float
    lower: float
    upper: float
    n: int
    method: str


def proportion_ci(successes: int, n: int, *, confidence: float = 0.95) -> ProportionInterval:
    """Wilson score interval for a success rate.

    The Wilson interval (rather than the Wald ``p +/- z*sqrt``) because it
    stays valid at the 0/1 boundaries this project routinely hits (a policy
    that never succeeds still gets an honest, non-degenerate interval).
    """
    if n <= 0:
        return ProportionInterval(rate=0.0, lower=0.0, upper=1.0, n=0, method="wilson-empty")
    try:
        from scipy.stats import norm as _norm

        z = float(_norm.ppf(0.5 + confidence / 2.0))
    except ImportError:
        z = 1.96
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return ProportionInterval(
        rate=p,
        lower=max(0.0, centre - half),
        upper=min(1.0, centre + half),
        n=n,
        method="wilson",
    )


@dataclass(frozen=True)
class MeanInterval:
    """Mean estimate with an uncertainty interval."""

    mean: float
    lower: float
    upper: float
    n: int
    method: str


def mean_ci(values: list[float] | np.ndarray, *, confidence: float = 0.95) -> MeanInterval:
    """Normal-approximation interval for a mean (descriptive, not a test)."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return MeanInterval(mean=float("nan"), lower=float("nan"), upper=float("nan"), n=0, method="empty")
    mean = float(np.mean(arr))
    if n == 1:
        return MeanInterval(mean=mean, lower=mean, upper=mean, n=1, method="single-sample")
    try:
        from scipy.stats import t as _t

        critical = float(_t.ppf(0.5 + confidence / 2.0, n - 1))
    except ImportError:
        critical = 1.96
    half = critical * float(np.std(arr, ddof=1)) / float(np.sqrt(n))
    return MeanInterval(mean=mean, lower=mean - half, upper=mean + half, n=n, method="t")


def bootstrap_mean_ci(
    values: list[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> MeanInterval:
    """Percentile bootstrap interval for a mean."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return MeanInterval(mean=float("nan"), lower=float("nan"), upper=float("nan"), n=0, method="bootstrap-empty")
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_resamples, n), replace=True)
    means = np.mean(draws, axis=1)
    alpha = 1.0 - confidence
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return MeanInterval(mean=float(np.mean(arr)), lower=lower, upper=upper, n=n, method="bootstrap")


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size between two proportions (0 = no effect)."""
    return float(2 * np.arcsin(np.sqrt(np.clip(p1, 0.0, 1.0))) - 2 * np.arcsin(np.sqrt(np.clip(p2, 0.0, 1.0))))


@dataclass(frozen=True)
class WelchResult:
    """Welch's t comparison of two independent samples."""

    mean_a: float
    mean_b: float
    delta: float
    t_stat: float
    dof: float
    p_value: float
    n_a: int
    n_b: int
    method: str


def welch_t(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> WelchResult:
    """Welch's unequal-variance t comparison (descriptive p-value, not a gate)."""
    aa = np.asarray(list(a), dtype=np.float64)
    bb = np.asarray(list(b), dtype=np.float64)
    aa = aa[np.isfinite(aa)]
    bb = bb[np.isfinite(bb)]
    na, nb = int(aa.size), int(bb.size)
    ma = float(np.mean(aa)) if na else float("nan")
    mb = float(np.mean(bb)) if nb else float("nan")
    if na < 2 or nb < 2:
        return WelchResult(ma, mb, mb - ma, float("nan"), 0.0, 1.0, na, nb, "welch-insufficient")
    va, vb = float(np.var(aa, ddof=1)), float(np.var(bb, ddof=1))
    denom = va / na + vb / nb
    if denom <= 0:
        return WelchResult(ma, mb, mb - ma, 0.0, 1.0, 1.0, na, nb, "welch-degenerate")
    t_stat = (mb - ma) / float(np.sqrt(denom))
    dof = denom**2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    try:
        from scipy.stats import t as _t

        p_value = float(2 * _t.sf(abs(t_stat), dof))
    except ImportError:
        p_value = float(2 * min(0.5, 1.0 / (1.0 + abs(t_stat))))
    return WelchResult(ma, mb, mb - ma, float(t_stat), float(dof), p_value, na, nb, "welch")
