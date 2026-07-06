from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incentive.design import (  # noqa: E402
    effort_from_bonus,
    expected_cost_per_sample,
    misspecified_effort,
    optimized_bonus,
    q_linear,
    risk_neutral_fixed_rho_design,
    theory_budget,
)


def test_theory_budget_matches_main_text_formula_exactly():
    pi = np.array([0.2, 0.5, 0.1])
    bonus = np.array([1.5, 2.0, 0.75])
    effort = np.array([0.4, 0.6, 0.8])
    rho = 0.25
    w0 = 1.2
    k = 3.0

    expected = np.sum(((rho * bonus * q_linear(effort) + w0) * pi)) + rho * k

    assert theory_budget(pi, rho, bonus, effort, w0, k) == pytest.approx(expected)


def test_optimized_bonus_and_effort_algebra():
    rho = 0.2
    w0 = 0.25

    bonus = optimized_bonus(w0, rho)
    effort = effort_from_bonus(rho, bonus)

    assert bonus == pytest.approx(np.sqrt(w0) / rho)
    assert effort == pytest.approx(np.sqrt(w0))
    assert expected_cost_per_sample(rho, bonus, effort, w0) == pytest.approx(2 * w0)


def test_fixed_rho_design_uses_sqrt_tau_allocation_before_clipping():
    tau = np.array([1.0, 4.0, 9.0])
    budget = 2.25
    rho = 0.25
    w0 = 2.0
    k = 1.0

    pi = risk_neutral_fixed_rho_design(tau, budget, rho, w0, k)
    expected = ((budget - rho * k) / (2 * w0)) * np.sqrt(tau) / np.sqrt(tau).sum()

    np.testing.assert_allclose(pi, expected)
    assert theory_budget(pi, rho, optimized_bonus(w0, rho), np.sqrt(w0), w0, k) == pytest.approx(
        budget
    )


def test_posterior_effort_scales_effective_audit_probability():
    effort = effort_from_bonus(rho=0.5, bonus=1.2, posterior_scale=0.8)

    assert effort == pytest.approx(0.8 * 0.5 * 1.2)


def test_misspecified_effort_scales_by_kappa_and_clips():
    effort = np.array([0.4, 0.9])

    np.testing.assert_allclose(misspecified_effort(effort, kappa=0.5), [0.2, 0.45])
    np.testing.assert_allclose(misspecified_effort(effort, kappa=2.0), [0.8, 1.0])
    np.testing.assert_allclose(
        misspecified_effort(effort, kappa=2.0, clip=False), [0.8, 1.8]
    )


def test_design_returns_zero_probabilities_when_budget_cannot_cover_sentinels():
    tau = np.array([1.0, 4.0, 9.0])
    rho = 0.4
    k = 5.0

    pi = risk_neutral_fixed_rho_design(tau, budget=rho * k, rho=rho, w0=1.0, k=k)

    np.testing.assert_array_equal(pi, np.zeros_like(tau))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tau": [1.0], "budget": 1.0, "rho": 0.0, "w0": 1.0, "k": 0.0},
        {"tau": [1.0], "budget": 1.0, "rho": 1.0, "w0": 1.0, "k": 0.0},
        {"tau": [1.0], "budget": 1.0, "rho": 0.5, "w0": 0.0, "k": 0.0},
        {"tau": [1.0], "budget": -1.0, "rho": 0.5, "w0": 1.0, "k": 0.0},
        {"tau": [-1.0], "budget": 1.0, "rho": 0.5, "w0": 1.0, "k": 0.0},
    ],
)
def test_design_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        risk_neutral_fixed_rho_design(**kwargs)
