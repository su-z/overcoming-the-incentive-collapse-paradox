"""Estimators for the incentive-aware active inference experiments.

The formulas here implement the estimators from Yin, Su, and Li,
"Overcoming the Incentive Collapse Paradox" (ICML 2026):

* Eq. ``eq:active-mean`` for the incentive-robust active mean estimator.
* The appendix baseline formulas for active, uniform, and classical symmetric
  label-noise corrections.
* The M-estimation sandwich form specialized to linear least squares.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "mean_incentive_robust",
    "mean_active_debiased",
    "mean_uniform_debiased",
    "mean_classical_symmetric_noise",
    "ratio_estimator",
    "protein_idr_phosphorylation_ratio",
    "weighted_least_squares_coef",
    "sandwich_linear_covariance",
]


def _as_1d(name: str, values: np.ndarray | float | list[float]) -> np.ndarray:
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


def _as_compatible_1d(
    name: str, values: np.ndarray | float | list[float], n: int
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(n, float(array))
    elif array.ndim != 1:
        raise ValueError(f"{name} must be scalar or one-dimensional")
    elif array.size != n:
        raise ValueError(f"{name} must have length {n}, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_same_length(reference_name: str, reference: np.ndarray, **arrays: np.ndarray) -> None:
    for name, array in arrays.items():
        if array.size != reference.size:
            raise ValueError(
                f"{name} must have the same length as {reference_name}: "
                f"{array.size} != {reference.size}"
            )


def _validate_probability(name: str, values: np.ndarray, *, strict_half: bool = False) -> None:
    lower_ok = values > 0.5 if strict_half else values > 0.0
    if not np.all(lower_ok & (values <= 1.0)):
        if strict_half:
            raise ValueError(f"{name} must be in the interval (0.5, 1]")
        raise ValueError(f"{name} must be in the interval (0, 1]")


def _validate_nonnegative_weights(weights: np.ndarray) -> None:
    if np.any(weights < 0.0):
        raise ValueError("weights must be nonnegative")
    if np.sum(weights) <= 0.0:
        raise ValueError("weights must have positive total mass")


def mean_incentive_robust(
    y_reported,
    f,
    xi,
    zeta,
    pi,
    rho: float,
    q_effort,
) -> float:
    """Estimate ``E[Y_true]`` using the sentinel-audited active estimator.

    Implements Eq. ``eq:active-mean``:

    ``mean(f_i + (Y_i - f_i) * (xi_i * zeta_i / (1 - rho)) / (pi_i * q_i))``.

    Assumptions made explicit by validation: sampling probabilities ``pi_i`` and
    correction probabilities ``q_i`` are in ``(0, 1]``; ``rho`` leaves a positive
    regular-task probability. The paper assumes ``rho in (0, 1)``; this function
    also permits the limiting no-sentinel case ``rho = 0``.
    """

    y = _as_1d("y_reported", y_reported)
    n = y.size
    f_array = _as_compatible_1d("f", f, n)
    xi_array = _as_compatible_1d("xi", xi, n)
    zeta_array = _as_compatible_1d("zeta", zeta, n)
    pi_array = _as_compatible_1d("pi", pi, n)
    q_array = _as_compatible_1d("q_effort", q_effort, n)
    if not np.isfinite(rho) or rho < 0.0 or rho >= 1.0:
        raise ValueError("rho must be in the interval [0, 1)")
    _validate_probability("pi", pi_array)
    _validate_probability("q_effort", q_array)

    correction = (xi_array * zeta_array / (1.0 - rho)) / (pi_array * q_array)
    return float(np.mean(f_array + (y - f_array) * correction))


def mean_active_debiased(y_reported, f, xi, pi, q_effort) -> float:
    """Estimate ``E[Y_true]`` with the active baseline debiasing correction.

    Implements the appendix baseline formula:

    ``mean(f_i + (Y_i - f_i) * xi_i / (pi_i * q_i))``.
    """

    y = _as_1d("y_reported", y_reported)
    n = y.size
    f_array = _as_compatible_1d("f", f, n)
    xi_array = _as_compatible_1d("xi", xi, n)
    pi_array = _as_compatible_1d("pi", pi, n)
    q_array = _as_compatible_1d("q_effort", q_effort, n)
    _validate_probability("pi", pi_array)
    _validate_probability("q_effort", q_array)

    return float(np.mean(f_array + (y - f_array) * xi_array / (pi_array * q_array)))


def mean_uniform_debiased(y_reported, f, xi, pi_uniform, q_effort) -> float:
    """Estimate ``E[Y_true]`` with the uniform-sampling debiasing correction.

    This is the active baseline formula with ``pi_i`` fixed to
    ``pi_uniform`` for each observation.
    """

    y = _as_1d("y_reported", y_reported)
    n = y.size
    pi_array = _as_compatible_1d("pi_uniform", pi_uniform, n)
    return mean_active_debiased(y, f, xi, pi_array, q_effort)


def mean_classical_symmetric_noise(y_reported, xi, pi_uniform, q_effort) -> float:
    """Estimate a binary mean under the classical symmetric-noise correction.

    Implements the appendix formula:

    ``mean(xi_i / pi_uniform * (Y_i + q_i - 1) / (2 * q_i - 1))``.

    The symmetric-noise correction is identifiable only when each
    ``q_i > 0.5``; the function rejects lower values explicitly.
    """

    y = _as_1d("y_reported", y_reported)
    n = y.size
    xi_array = _as_compatible_1d("xi", xi, n)
    pi_array = _as_compatible_1d("pi_uniform", pi_uniform, n)
    q_array = _as_compatible_1d("q_effort", q_effort, n)
    _validate_probability("pi_uniform", pi_array)
    _validate_probability("q_effort", q_array, strict_half=True)

    corrected = (y + q_array - 1.0) / (2.0 * q_array - 1.0)
    return float(np.mean(xi_array / pi_array * corrected))


def ratio_estimator(numerator_values, denominator_values, weights=None) -> float:
    """Return a ratio of means, optionally using the same weights in both means."""

    numerator = _as_1d("numerator_values", numerator_values)
    denominator = _as_1d("denominator_values", denominator_values)
    _validate_same_length("numerator_values", numerator, denominator_values=denominator)

    if weights is None:
        numerator_mean = np.mean(numerator)
        denominator_mean = np.mean(denominator)
    else:
        weight_array = _as_compatible_1d("weights", weights, numerator.size)
        _validate_nonnegative_weights(weight_array)
        numerator_mean = np.average(numerator, weights=weight_array)
        denominator_mean = np.average(denominator, weights=weight_array)

    if denominator_mean == 0.0:
        raise ValueError("denominator weighted mean is zero")
    return float(numerator_mean / denominator_mean)


def protein_idr_phosphorylation_ratio(y_idr, phosphorylated, correction_weights=None) -> float:
    """Estimate the protein IDR/phosphorylation binary odds ratio.

    The returned value is
    ``odds(IDR = 1 | phosphorylated = 1) / odds(IDR = 1 | phosphorylated = 0)``.
    Optional correction weights are applied consistently to both subgroup means.
    """

    y = _as_1d("y_idr", y_idr)
    phosphorylated_array = _as_compatible_1d("phosphorylated", phosphorylated, y.size)
    phosphorylated_mask = phosphorylated_array.astype(bool)
    if not np.any(phosphorylated_mask):
        raise ValueError("phosphorylated must include at least one treated observation")
    if np.all(phosphorylated_mask):
        raise ValueError("phosphorylated must include at least one untreated observation")

    if correction_weights is None:
        weights = np.ones(y.size)
    else:
        weights = _as_compatible_1d("correction_weights", correction_weights, y.size)
        _validate_nonnegative_weights(weights)

    treated_weight = weights[phosphorylated_mask]
    untreated_weight = weights[~phosphorylated_mask]
    _validate_nonnegative_weights(treated_weight)
    _validate_nonnegative_weights(untreated_weight)

    treated_mean = np.average(y[phosphorylated_mask], weights=treated_weight)
    untreated_mean = np.average(y[~phosphorylated_mask], weights=untreated_weight)
    if treated_mean <= 0.0 or treated_mean >= 1.0:
        raise ValueError("treated IDR probability must lie strictly between 0 and 1")
    if untreated_mean <= 0.0 or untreated_mean >= 1.0:
        raise ValueError("untreated IDR probability must lie strictly between 0 and 1")
    return float((treated_mean / (1.0 - treated_mean)) / (untreated_mean / (1.0 - untreated_mean)))


def _as_design_matrix(X) -> np.ndarray:
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("X must have at least one row and one column")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("X must contain only finite values")
    return matrix


def weighted_least_squares_coef(X, y, weights=None) -> np.ndarray:
    """Compute weighted least-squares coefficients.

    No intercept column is added. Callers must include an intercept in ``X`` if
    the target regression requires one.
    """

    matrix = _as_design_matrix(X)
    y_array = _as_compatible_1d("y", y, matrix.shape[0])
    if weights is None:
        weight_array = np.ones(matrix.shape[0])
    else:
        weight_array = _as_compatible_1d("weights", weights, matrix.shape[0])
        _validate_nonnegative_weights(weight_array)

    xtw = matrix.T * weight_array
    normal_matrix = xtw @ matrix
    normal_rhs = xtw @ y_array
    try:
        return np.linalg.solve(normal_matrix, normal_rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("weighted least-squares normal matrix is singular") from exc


def sandwich_linear_covariance(X, residuals, weights=None) -> np.ndarray:
    """Return the HC0 sandwich covariance for linear least-squares coefficients.

    This is the linear least-squares specialization of the M-estimation sandwich
    form, using score terms ``w_i x_i residual_i``:

    ``(X' W X)^(-1) X' diag(w_i^2 residual_i^2) X (X' W X)^(-1)``.

    As with ``weighted_least_squares_coef``, no intercept column is added.
    """

    matrix = _as_design_matrix(X)
    residual_array = _as_compatible_1d("residuals", residuals, matrix.shape[0])
    if weights is None:
        weight_array = np.ones(matrix.shape[0])
    else:
        weight_array = _as_compatible_1d("weights", weights, matrix.shape[0])
        _validate_nonnegative_weights(weight_array)

    xtw = matrix.T * weight_array
    bread = xtw @ matrix
    score_scale = (weight_array * residual_array) ** 2
    meat = matrix.T @ (matrix * score_scale[:, None])
    try:
        bread_inv = np.linalg.solve(bread, np.eye(bread.shape[0]))
    except np.linalg.LinAlgError as exc:
        raise ValueError("linear sandwich covariance bread matrix is singular") from exc
    return bread_inv @ meat @ bread_inv
