from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference.sampling import (  # noqa: E402
    active_probabilities,
    clip_probabilities,
    expected_budget,
    incentive_aware_probabilities,
    mixed_active_uniform_probabilities,
    scale_probabilities_to_budget,
    uniform_probabilities,
)


def test_uniform_probabilities_sum_to_expected_labels():
    pi = uniform_probabilities(4, expected_labels=2.0)

    np.testing.assert_allclose(pi, np.full(4, 0.5))
    assert pi.sum() == pytest.approx(2.0)

    from_budget = uniform_probabilities(4, budget=3.0, per_sample_cost=1.5)
    np.testing.assert_allclose(from_budget, np.full(4, 0.5))

    capped = uniform_probabilities(3, expected_labels=10.0)
    np.testing.assert_allclose(capped, np.ones(3))


def test_active_probabilities_follow_sqrt_tau_ordering():
    tau = np.array([1.0, 4.0, 9.0])

    pi = active_probabilities(tau, expected_labels=1.2)
    expected = 1.2 * np.sqrt(tau) / np.sqrt(tau).sum()

    np.testing.assert_allclose(pi, expected)
    assert pi[0] < pi[1] < pi[2]
    assert pi.sum() == pytest.approx(1.2)


def test_mixed_active_uniform_endpoints_and_budget_conversion():
    tau = np.array([1.0, 4.0, 9.0, 16.0])
    expected_labels = 2.0

    active = active_probabilities(tau, expected_labels=expected_labels)
    uniform = uniform_probabilities(tau.size, expected_labels=expected_labels)

    np.testing.assert_allclose(
        mixed_active_uniform_probabilities(
            tau, mix_tau=0.0, budget=4.0, per_sample_cost=2.0
        ),
        active,
    )
    np.testing.assert_allclose(
        mixed_active_uniform_probabilities(
            tau, mix_tau=1.0, budget=4.0, per_sample_cost=2.0
        ),
        uniform,
    )
    np.testing.assert_allclose(
        mixed_active_uniform_probabilities(
            tau, mix_tau=0.5, budget=4.0, per_sample_cost=2.0
        ),
        0.5 * active + 0.5 * uniform,
    )


def test_scaling_clips_probabilities_and_respects_budget():
    weights = np.array([100.0, 1.0, 0.0])
    cost = np.array([1.0, 1.0, 5.0])

    pi = scale_probabilities_to_budget(weights, budget=1.5, per_item_cost=cost)

    np.testing.assert_allclose(pi, [1.0, 0.5, 0.0])
    assert expected_budget(pi, cost) == pytest.approx(1.5)
    assert np.all((pi >= 0.0) & (pi <= 1.0))

    np.testing.assert_allclose(clip_probabilities([-0.2, 0.4, 1.8]), [0.0, 0.4, 1.0])


def test_active_probabilities_fall_back_to_uniform_when_tau_is_all_zero():
    pi = active_probabilities(np.zeros(3), expected_labels=1.5)

    np.testing.assert_allclose(pi, np.full(3, 0.5))


def test_incentive_aware_matches_unclipped_closed_form_default():
    tau = np.array([1.0, 4.0, 9.0, 16.0])
    budget = 1.4
    rho = 0.2
    w0 = 1.0
    k = 1.0

    pi = incentive_aware_probabilities(tau, budget=budget, rho=rho, w0=w0, k=k)
    expected = ((budget - rho * k) / (2.0 * w0)) * np.sqrt(tau) / np.sqrt(tau).sum()

    np.testing.assert_allclose(pi, expected)
    assert np.all(pi < 1.0)
    assert expected_budget(pi, per_item_cost=2.0 * w0, fixed_cost=rho * k) == pytest.approx(
        budget
    )


def test_incentive_aware_returns_zero_when_budget_cannot_cover_fixed_cost():
    tau = np.array([1.0, 4.0, 9.0])

    pi = incentive_aware_probabilities(tau, budget=0.5, rho=0.25, w0=1.0, k=2.0)

    np.testing.assert_array_equal(pi, np.zeros_like(tau))
