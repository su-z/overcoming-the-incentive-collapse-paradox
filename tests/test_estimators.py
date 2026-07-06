from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from inference.estimators import (  # noqa: E402
    mean_active_debiased,
    mean_classical_symmetric_noise,
    mean_incentive_robust,
    mean_uniform_debiased,
    protein_idr_phosphorylation_ratio,
    ratio_estimator,
    sandwich_linear_covariance,
    weighted_least_squares_coef,
)


def test_mean_estimators_match_hand_computed_values():
    y = np.array([1.0, 0.0, 1.0])
    f = np.array([0.2, 0.3, 0.4])
    xi = np.array([1.0, 0.0, 1.0])
    zeta = np.array([1.0, 1.0, 0.0])
    pi = np.array([0.5, 0.25, 1.0])
    q = np.array([1.0, 0.8, 0.5])

    assert mean_incentive_robust(y, f, xi, zeta, pi, rho=0.2, q_effort=q) == pytest.approx(
        2.9 / 3.0
    )
    assert mean_active_debiased(y, f, xi, pi, q) == pytest.approx(3.7 / 3.0)
    assert mean_uniform_debiased(y, f, xi, pi_uniform=0.5, q_effort=q) == pytest.approx(
        4.9 / 3.0
    )


def test_active_and_incentive_estimators_are_unbiased_when_q_is_one():
    y_true = np.array([1.0, 0.0, 1.0])
    f = np.array([0.1, 0.4, 0.7])
    pi = np.array([0.25, 0.5, 0.75])
    rho = 0.2
    q = np.ones_like(y_true)
    target = np.mean(y_true)

    active_expectation = 0.0
    incentive_expectation = 0.0
    for xi_tuple in itertools.product([0.0, 1.0], repeat=y_true.size):
        xi = np.array(xi_tuple)
        xi_prob = np.prod(np.where(xi == 1.0, pi, 1.0 - pi))
        active_expectation += xi_prob * mean_active_debiased(y_true, f, xi, pi, q)
        for zeta_tuple in itertools.product([0.0, 1.0], repeat=y_true.size):
            zeta = np.array(zeta_tuple)
            zeta_prob = np.prod(np.where(zeta == 1.0, 1.0 - rho, rho))
            incentive_expectation += (
                xi_prob
                * zeta_prob
                * mean_incentive_robust(y_true, f, xi, zeta, pi, rho, q)
            )

    assert active_expectation == pytest.approx(target)
    assert incentive_expectation == pytest.approx(target)


def test_active_and_incentive_estimators_are_unbiased_with_imperfect_detection():
    y_true = np.array([1.0, 0.0, 1.0])
    f = np.array([0.2, 0.4, 0.7])
    pi = np.array([0.25, 0.5, 0.75])
    q = np.array([0.6, 0.8, 0.9])
    rho = 0.3
    target = np.mean(y_true)

    active_expectation = 0.0
    incentive_expectation = 0.0
    for detected_tuple in itertools.product([0.0, 1.0], repeat=y_true.size):
        detected = np.array(detected_tuple)
        detection_prob = np.prod(np.where(detected == 1.0, q, 1.0 - q))
        y_reported = f + (y_true - f) * detected
        for xi_tuple in itertools.product([0.0, 1.0], repeat=y_true.size):
            xi = np.array(xi_tuple)
            xi_prob = np.prod(np.where(xi == 1.0, pi, 1.0 - pi))
            active_expectation += (
                detection_prob
                * xi_prob
                * mean_active_debiased(y_reported, f, xi, pi, q)
            )
            for zeta_tuple in itertools.product([0.0, 1.0], repeat=y_true.size):
                zeta = np.array(zeta_tuple)
                zeta_prob = np.prod(np.where(zeta == 1.0, 1.0 - rho, rho))
                incentive_expectation += (
                    detection_prob
                    * xi_prob
                    * zeta_prob
                    * mean_incentive_robust(y_reported, f, xi, zeta, pi, rho, q)
                )

    assert active_expectation == pytest.approx(target)
    assert incentive_expectation == pytest.approx(target)


def test_classical_symmetric_noise_correction_recovers_binary_mean():
    y_true = np.array([0.0, 1.0, 1.0, 0.0])
    q = np.array([0.6, 0.7, 0.8, 0.9])
    expected_reported_label = q * y_true + (1.0 - q) * (1.0 - y_true)

    corrected = mean_classical_symmetric_noise(
        expected_reported_label,
        xi=np.ones_like(y_true),
        pi_uniform=1.0,
        q_effort=q,
    )

    assert corrected == pytest.approx(np.mean(y_true))
    with pytest.raises(ValueError, match=r"\(0\.5, 1\]"):
        mean_classical_symmetric_noise(y_true, xi=np.ones_like(y_true), pi_uniform=1.0, q_effort=0.5)


def test_ratio_estimators_on_toy_vectors():
    assert ratio_estimator([2.0, 4.0], [1.0, 3.0]) == pytest.approx(1.5)
    assert ratio_estimator([2.0, 4.0], [1.0, 3.0], weights=[3.0, 1.0]) == pytest.approx(
        10.0 / 6.0
    )

    y_idr = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    phosphorylated = np.array([1, 1, 0, 0, 0], dtype=bool)
    assert protein_idr_phosphorylation_ratio(y_idr, phosphorylated) == pytest.approx(0.5)
    assert protein_idr_phosphorylation_ratio(
        y_idr,
        phosphorylated,
        correction_weights=np.array([1.0, 3.0, 2.0, 1.0, 1.0]),
    ) == pytest.approx(1.0 / 9.0)


def test_weighted_least_squares_recovers_coefficients_without_adding_intercept():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    X = np.column_stack([np.ones_like(x), x])
    y = 2.0 + 3.0 * x

    coef = weighted_least_squares_coef(X, y, weights=np.array([1.0, 2.0, 1.0, 4.0]))

    np.testing.assert_allclose(coef, np.array([2.0, 3.0]), atol=1e-12)


def test_linear_helpers_reject_singular_designs():
    X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    with pytest.raises(ValueError, match="singular"):
        weighted_least_squares_coef(X, np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="singular"):
        sandwich_linear_covariance(X, np.array([0.1, -0.2, 0.3]))


def test_sandwich_linear_covariance_matches_hand_computed_matrix():
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    residuals = np.array([1.0, -1.0, 2.0])
    weights = np.array([1.0, 2.0, 1.0])

    covariance = sandwich_linear_covariance(X, residuals, weights)

    expected = np.array([[17.0 / 16.0, -7.0 / 8.0], [-7.0 / 8.0, 5.0 / 4.0]])
    np.testing.assert_allclose(covariance, expected)
