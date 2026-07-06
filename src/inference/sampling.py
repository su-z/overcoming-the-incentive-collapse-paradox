"""Sampling policies for budget-constrained active inference."""

from __future__ import annotations

from typing import Any

import numpy as np


ArrayLike = Any


def _validate_nonnegative_scalar(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return value


def _validate_positive_scalar(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return value


def _validate_nonnegative_array(name: str, value: ArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain only finite nonnegative values")
    return array


def _validate_probability_array(name: str, value: ArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return array


def _validate_n(n: int) -> int:
    if isinstance(n, bool):
        raise ValueError("n must be a nonnegative integer")
    n_int = int(n)
    if n_int != n or n_int < 0:
        raise ValueError("n must be a nonnegative integer")
    return n_int


def _broadcast_cost(per_item_cost: ArrayLike, shape: tuple[int, ...]) -> np.ndarray:
    cost = _validate_nonnegative_array("per_item_cost", per_item_cost)
    try:
        return np.broadcast_to(cost, shape).astype(float, copy=False)
    except ValueError as exc:
        raise ValueError("per_item_cost must be scalar or broadcast to weights") from exc


def _expected_label_budget(
    n: int,
    expected_labels: float | None,
    budget: float | None,
    per_sample_cost: float | None,
) -> float:
    if expected_labels is not None and budget is not None:
        raise ValueError("provide either expected_labels or budget, not both")
    if expected_labels is not None:
        return _validate_nonnegative_scalar("expected_labels", expected_labels)
    if budget is None:
        return 1.0 if n > 0 else 0.0

    budget = _validate_nonnegative_scalar("budget", budget)
    cost = 1.0 if per_sample_cost is None else _validate_positive_scalar(
        "per_sample_cost", per_sample_cost
    )
    return budget / cost


def clip_probabilities(pi: ArrayLike) -> np.ndarray:
    """Clip finite numeric values into the probability interval [0, 1]."""

    array = np.asarray(pi, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError("pi must contain only finite values")
    return np.clip(array, 0.0, 1.0)


def expected_budget(
    pi: ArrayLike, per_item_cost: ArrayLike, fixed_cost: float = 0.0
) -> float:
    """Return sum_i pi_i * per_item_cost_i + fixed_cost."""

    pi_array = _validate_probability_array("pi", pi)
    fixed_cost = _validate_nonnegative_scalar("fixed_cost", fixed_cost)
    cost = _broadcast_cost(per_item_cost, pi_array.shape)
    return float(np.sum(pi_array * cost) + fixed_cost)


def scale_probabilities_to_budget(
    weights: ArrayLike,
    budget: float,
    per_item_cost: ArrayLike,
    fixed_cost: float = 0.0,
) -> np.ndarray:
    """Scale nonnegative weights into capped probabilities under a budget.

    The returned probabilities have the form ``min(1, lambda * weights)`` for
    positive-cost items, with ``lambda`` found by monotone bisection. The fixed
    cost is treated as an unavoidable overhead; if it exhausts the budget, no
    positive-cost item is sampled.
    """

    weight_array = _validate_nonnegative_array("weights", weights)
    budget = _validate_nonnegative_scalar("budget", budget)
    fixed_cost = _validate_nonnegative_scalar("fixed_cost", fixed_cost)
    cost = _broadcast_cost(per_item_cost, weight_array.shape)

    probabilities = np.zeros_like(weight_array, dtype=float)
    if weight_array.size == 0 or budget <= fixed_cost:
        return probabilities

    positive_weight = weight_array > 0.0
    if not np.any(positive_weight):
        return probabilities

    remaining_budget = budget - fixed_cost
    free_items = positive_weight & (cost == 0.0)
    paid_items = positive_weight & (cost > 0.0)
    probabilities[free_items] = 1.0
    if not np.any(paid_items):
        return probabilities

    max_paid_cost = float(np.sum(cost[paid_items]))
    if remaining_budget >= max_paid_cost:
        probabilities[paid_items] = 1.0
        return probabilities

    def variable_cost(scale: float) -> float:
        paid_probabilities = np.minimum(1.0, scale * weight_array[paid_items])
        return float(np.sum(paid_probabilities * cost[paid_items]))

    low = 0.0
    high = 1.0
    for _ in range(256):
        if variable_cost(high) >= remaining_budget:
            break
        high *= 2.0
    else:
        raise RuntimeError("failed to bracket budget scaling factor")

    for _ in range(128):
        mid = 0.5 * (low + high)
        if variable_cost(mid) <= remaining_budget:
            low = mid
        else:
            high = mid

    probabilities[paid_items] = np.minimum(1.0, low * weight_array[paid_items])
    while expected_budget(probabilities, cost, fixed_cost) > budget:
        next_low = np.nextafter(low, 0.0)
        if next_low == low:
            break
        low = next_low
        probabilities[paid_items] = np.minimum(1.0, low * weight_array[paid_items])
    return probabilities


def uniform_probabilities(
    n: int,
    expected_labels: float | None = None,
    budget: float | None = None,
    per_sample_cost: float | None = None,
) -> np.ndarray:
    """Uniform sampling probabilities with a target expected label count."""

    n = _validate_n(n)
    if n == 0:
        return np.empty(0, dtype=float)

    label_budget = _expected_label_budget(n, expected_labels, budget, per_sample_cost)
    return scale_probabilities_to_budget(np.ones(n), label_budget, 1.0)


def active_probabilities(
    tau: ArrayLike,
    expected_labels: float | None = None,
    budget: float | None = None,
    per_sample_cost: float | None = None,
) -> np.ndarray:
    """Active sampling probabilities proportional to sqrt(tau_i)."""

    tau_array = _validate_nonnegative_array("tau", tau)
    if tau_array.size == 0:
        return np.empty_like(tau_array, dtype=float)

    label_budget = _expected_label_budget(
        tau_array.size, expected_labels, budget, per_sample_cost
    )
    weights = np.sqrt(tau_array)
    if not np.any(weights > 0.0):
        weights = np.ones_like(tau_array, dtype=float)
    return scale_probabilities_to_budget(weights, label_budget, 1.0)


def mixed_active_uniform_probabilities(
    tau: ArrayLike,
    mix_tau: float,
    expected_labels: float | None = None,
    budget: float | None = None,
    per_sample_cost: float | None = None,
) -> np.ndarray:
    """Appendix baseline: (1-mix_tau)*active + mix_tau*uniform.

    When a monetary budget is supplied, it is converted to the paper's
    expected-label scale by dividing by ``per_sample_cost`` before applying the
    ``budget / n`` uniform component.
    """

    tau_array = _validate_nonnegative_array("tau", tau)
    mix_tau = float(mix_tau)
    if not np.isfinite(mix_tau) or mix_tau < 0.0 or mix_tau > 1.0:
        raise ValueError("mix_tau must be a finite value in [0, 1]")
    if tau_array.size == 0:
        return np.empty_like(tau_array, dtype=float)

    label_budget = _expected_label_budget(
        tau_array.size, expected_labels, budget, per_sample_cost
    )
    active = active_probabilities(tau_array, expected_labels=label_budget)
    uniform_level = min(1.0, label_budget / tau_array.size)
    uniform = np.full(tau_array.shape, uniform_level, dtype=float)
    return clip_probabilities((1.0 - mix_tau) * active + mix_tau * uniform)


def incentive_aware_probabilities(
    tau: ArrayLike,
    budget: float,
    rho: float,
    w0: float,
    k: float,
    bonus: ArrayLike | None = None,
    effort: ArrayLike | None = None,
) -> np.ndarray:
    """Incentive-aware active sampling under the main-text budget.

    Defaults use the risk-neutral optimum ``bonus=sqrt(w0)/rho`` and
    ``effort=sqrt(w0)`` with ``q(e)=e``. The effective per-item cost is
    ``rho*bonus*q(effort)+w0`` and the fixed sentinel cost is ``rho*k``.
    """

    tau_array = _validate_nonnegative_array("tau", tau)
    budget = _validate_nonnegative_scalar("budget", budget)
    rho = float(rho)
    if not np.isfinite(rho) or rho <= 0.0 or rho >= 1.0:
        raise ValueError("rho must lie strictly between 0 and 1")
    w0 = _validate_nonnegative_scalar("w0", w0)
    k = _validate_nonnegative_scalar("k", k)

    fixed_cost = rho * k
    if tau_array.size == 0:
        return np.empty_like(tau_array, dtype=float)
    if budget <= fixed_cost:
        return np.zeros_like(tau_array, dtype=float)

    default_bonus = np.sqrt(w0) / rho
    default_effort = np.sqrt(w0)
    bonus_array = _validate_nonnegative_array(
        "bonus", default_bonus if bonus is None else bonus
    )
    effort_array = _validate_nonnegative_array(
        "effort", default_effort if effort is None else effort
    )
    bonus_b, effort_b, _ = np.broadcast_arrays(bonus_array, effort_array, tau_array)
    per_item_cost = rho * bonus_b * effort_b + w0

    weights = np.sqrt(tau_array)
    if not np.any(weights > 0.0):
        weights = np.ones_like(tau_array, dtype=float)
    return scale_probabilities_to_budget(weights, budget, per_item_cost, fixed_cost)


__all__ = [
    "uniform_probabilities",
    "active_probabilities",
    "mixed_active_uniform_probabilities",
    "incentive_aware_probabilities",
    "scale_probabilities_to_budget",
    "clip_probabilities",
    "expected_budget",
]
