"""Confidence interval helpers for the paper's asymptotic approximations."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

__all__ = [
    "z_value",
    "asymptotic_variance",
    "mean_ci",
    "ratio_delta_ci",
    "linear_coefficient_ci",
    "interval_width",
    "empirical_coverage",
]


_DENOMINATOR_TOLERANCE = 1e-12


def _validate_confidence_level(confidence_level: float) -> float:
    level = float(confidence_level)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be in the interval (0, 1)")
    return level


def _as_1d(name: str, values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _sample_size(n: int | None, default: int) -> int:
    if n is None:
        return default
    if isinstance(n, bool) or int(n) != n:
        raise ValueError("n must be a positive integer")
    size = int(n)
    if size <= 0:
        raise ValueError("n must be a positive integer")
    return size


def z_value(confidence_level: float = 0.90) -> float:
    """Return the two-sided standard-normal critical value."""

    level = _validate_confidence_level(confidence_level)
    return float(NormalDist().inv_cdf(0.5 + level / 2.0))


def asymptotic_variance(values, ddof: int = 1) -> float:
    """Estimate the variance of influence values used by the local CLTs."""

    array = _as_1d("values", values)
    if isinstance(ddof, bool) or int(ddof) != ddof:
        raise ValueError("ddof must be a nonnegative integer smaller than len(values)")
    ddof_int = int(ddof)
    if ddof_int < 0 or ddof_int >= array.size:
        raise ValueError("ddof must be a nonnegative integer smaller than len(values)")
    return float(np.var(array, ddof=ddof_int))


def mean_ci(
    point: float,
    influence_values,
    n: int | None = None,
    confidence_level: float = 0.90,
) -> tuple[float, float]:
    """Construct ``point +/- z * sqrt(var(influence_values) / n)``."""

    point_float = float(point)
    if not np.isfinite(point_float):
        raise ValueError("point must be finite")
    influence = _as_1d("influence_values", influence_values)
    size = _sample_size(n, influence.size)
    standard_error = np.sqrt(asymptotic_variance(influence) / size)
    radius = z_value(confidence_level) * standard_error
    return (float(point_float - radius), float(point_float + radius))


def ratio_delta_ci(
    numer_point: float,
    denom_point: float,
    influence_numer,
    influence_denom,
    confidence_level: float = 0.90,
) -> tuple[float, float]:
    """Construct a delta-method CI for ``numer_point / denom_point``."""

    numerator = float(numer_point)
    denominator = float(denom_point)
    if not np.isfinite(numerator):
        raise ValueError("numer_point must be finite")
    if not np.isfinite(denominator):
        raise ValueError("denom_point must be finite")
    if abs(denominator) <= _DENOMINATOR_TOLERANCE:
        raise ValueError("denom_point is too close to zero")

    numer_influence = _as_1d("influence_numer", influence_numer)
    denom_influence = _as_1d("influence_denom", influence_denom)
    if numer_influence.size != denom_influence.size:
        raise ValueError("influence_numer and influence_denom must have the same length")

    ratio_point = numerator / denominator
    ratio_influence = numer_influence / denominator - numerator * denom_influence / denominator**2
    standard_error = np.sqrt(asymptotic_variance(ratio_influence) / ratio_influence.size)
    radius = z_value(confidence_level) * standard_error
    return (float(ratio_point - radius), float(ratio_point + radius))


def linear_coefficient_ci(
    beta,
    covariance,
    n: int,
    index: int,
    confidence_level: float = 0.90,
) -> tuple[float, float]:
    """Construct the coefficient CI from ``sqrt(covariance[index, index] / n)``."""

    beta_array = _as_1d("beta", beta)
    covariance_array = np.asarray(covariance, dtype=float)
    if covariance_array.ndim != 2 or covariance_array.shape[0] != covariance_array.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if covariance_array.shape[0] != beta_array.size:
        raise ValueError("covariance dimension must match beta length")
    if not np.all(np.isfinite(covariance_array)):
        raise ValueError("covariance must contain only finite values")
    if isinstance(index, bool) or int(index) != index:
        raise ValueError("index must be an integer")
    idx = int(index)
    if idx < 0 or idx >= beta_array.size:
        raise IndexError("index is out of bounds for beta")

    size = _sample_size(n, beta_array.size)
    variance = float(covariance_array[idx, idx])
    if variance < 0.0:
        raise ValueError("covariance[index, index] must be nonnegative")
    standard_error = np.sqrt(variance / size)
    radius = z_value(confidence_level) * standard_error
    return (float(beta_array[idx] - radius), float(beta_array[idx] + radius))


def interval_width(ci) -> float:
    """Return the upper endpoint minus the lower endpoint."""

    interval = _as_1d("ci", ci)
    if interval.size != 2:
        raise ValueError("ci must contain exactly two endpoints")
    lower, upper = interval
    if lower > upper:
        raise ValueError("ci lower endpoint must be less than or equal to upper endpoint")
    return float(upper - lower)


def empirical_coverage(intervals, truth) -> float:
    """Return the fraction of intervals covering ``truth``, including boundaries."""

    interval_array = np.asarray(intervals, dtype=float)
    if interval_array.ndim != 2 or interval_array.shape[1] != 2 or interval_array.shape[0] == 0:
        raise ValueError("intervals must be a nonempty array with shape (n, 2)")
    if not np.all(np.isfinite(interval_array)):
        raise ValueError("intervals must contain only finite values")
    lower = interval_array[:, 0]
    upper = interval_array[:, 1]
    if np.any(lower > upper):
        raise ValueError("each interval lower endpoint must be less than or equal to upper endpoint")

    truth_array = np.asarray(truth, dtype=float)
    if truth_array.ndim == 0:
        truth_values = float(truth_array)
        if not np.isfinite(truth_values):
            raise ValueError("truth must be finite")
        covered = (lower <= truth_values) & (truth_values <= upper)
    elif truth_array.ndim == 1 and truth_array.size == interval_array.shape[0]:
        if not np.all(np.isfinite(truth_array)):
            raise ValueError("truth must contain only finite values")
        covered = (lower <= truth_array) & (truth_array <= upper)
    else:
        raise ValueError("truth must be scalar or have one value per interval")
    return float(np.mean(covered))
