"""Incentive design utilities derived from the ICML paper."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


ArrayLike = Any


def _maybe_scalar(value: np.ndarray) -> float | np.ndarray:
    if value.ndim == 0:
        return float(value)
    return value


def _call(func: Callable[[ArrayLike], ArrayLike], value: ArrayLike) -> np.ndarray:
    try:
        result = func(value)
    except (TypeError, ValueError):
        result = np.vectorize(func, otypes=[float])(value)
    return np.asarray(result, dtype=float)


def _validate_rho(rho: float) -> float:
    rho = float(rho)
    if not np.isfinite(rho) or rho <= 0.0 or rho >= 1.0:
        raise ValueError("rho must lie strictly between 0 and 1")
    return rho


def _validate_positive_scalar(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_nonnegative_scalar(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _validate_nonnegative_array(name: str, value: ArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain only finite nonnegative values")
    return array


def q_linear(e: ArrayLike) -> float | np.ndarray:
    """Default correction probability q(e)=e."""

    effort = np.asarray(e, dtype=float)
    return _maybe_scalar(effort)


def cost_quadratic(e: ArrayLike) -> float | np.ndarray:
    """Default effort cost c(e)=0.5*e**2."""

    effort = np.asarray(e, dtype=float)
    cost = 0.5 * effort**2
    return _maybe_scalar(cost)


def risk_neutral_utility(b: ArrayLike) -> float | np.ndarray:
    """Risk-neutral utility mu(b)=b."""

    bonus = np.asarray(b, dtype=float)
    return _maybe_scalar(bonus)


def effort_from_bonus(
    rho: float,
    bonus: ArrayLike,
    utility: Callable[[ArrayLike], ArrayLike] | None = None,
    cost_derivative_inverse: Callable[[ArrayLike], ArrayLike] | None = None,
    q_prime: Callable[[ArrayLike], ArrayLike] | None = None,
    posterior_scale: float = 1.0,
    clip: bool = True,
) -> float | np.ndarray:
    """Solve rho*mu(b)*q'(e)=c'(e) for effort.

    Defaults match q(e)=e, c(e)=0.5*e**2, and mu(b)=b, yielding e=rho*b.
    ``posterior_scale`` implements imperfect sentinel beliefs by replacing rho
    with rho_posterior = posterior_scale*rho.
    """

    rho = _validate_rho(rho)
    posterior_scale = _validate_nonnegative_scalar(
        "posterior_scale", posterior_scale
    )
    bonus_array = _validate_nonnegative_array("bonus", bonus)
    utility = risk_neutral_utility if utility is None else utility
    cost_derivative_inverse = (
        (lambda x: x)
        if cost_derivative_inverse is None
        else cost_derivative_inverse
    )

    marginal = rho * posterior_scale * _call(utility, bonus_array)
    if np.any(~np.isfinite(marginal)) or np.any(marginal < 0.0):
        raise ValueError("utility must return finite nonnegative values")

    if q_prime is None:
        effort = _call(cost_derivative_inverse, marginal)
    else:
        effort = _call(cost_derivative_inverse, marginal)
        if clip:
            effort = np.clip(effort, 0.0, 1.0)
        for _ in range(100):
            q_prime_values = _call(q_prime, effort)
            if np.any(~np.isfinite(q_prime_values)) or np.any(q_prime_values < 0.0):
                raise ValueError("q_prime must return finite nonnegative values")
            next_effort = _call(cost_derivative_inverse, marginal * q_prime_values)
            if clip:
                next_effort = np.clip(next_effort, 0.0, 1.0)
            if np.allclose(next_effort, effort, rtol=1e-12, atol=1e-12):
                effort = next_effort
                break
            effort = next_effort

    if np.any(~np.isfinite(effort)) or np.any(effort < 0.0):
        raise ValueError("cost_derivative_inverse must return finite nonnegative values")
    if clip:
        effort = np.clip(effort, 0.0, 1.0)
    return _maybe_scalar(np.asarray(effort, dtype=float))


def optimized_bonus(w0: float, rho: float) -> float:
    """Fixed-rho optimum b*=sqrt(w0)/rho for the risk-neutral model."""

    w0 = _validate_positive_scalar("w0", w0)
    rho = _validate_rho(rho)
    return float(np.sqrt(w0) / rho)


def _capped_proportional_allocation(
    weights: np.ndarray, target_mass: float
) -> np.ndarray:
    allocation = np.zeros_like(weights, dtype=float)
    active = weights > 0.0
    if target_mass <= 0.0 or not np.any(active):
        return allocation

    remaining = min(float(target_mass), float(np.count_nonzero(active)))
    while remaining > 0.0 and np.any(active):
        active_indices = np.flatnonzero(active)
        active_weights = weights[active]
        total_weight = float(active_weights.sum())
        if total_weight <= 0.0:
            break

        candidate = remaining * active_weights / total_weight
        saturated = candidate >= 1.0
        if not np.any(saturated):
            allocation[active] = candidate
            return allocation

        saturated_indices = active_indices[saturated]
        allocation[saturated_indices] = 1.0
        active[saturated_indices] = False
        remaining -= float(len(saturated_indices))

    return allocation


def risk_neutral_fixed_rho_design(
    tau: ArrayLike,
    budget: float,
    rho: float,
    w0: float,
    k: float,
) -> np.ndarray:
    """Return fixed-rho sampling probabilities for the risk-neutral optimum."""

    tau_array = _validate_nonnegative_array("tau", tau)
    budget = _validate_nonnegative_scalar("budget", budget)
    rho = _validate_rho(rho)
    w0 = _validate_positive_scalar("w0", w0)
    k = _validate_nonnegative_scalar("k", k)

    if budget <= rho * k:
        return np.zeros_like(tau_array, dtype=float)

    weights = np.sqrt(tau_array)
    total_weight = float(weights.sum())
    if total_weight == 0.0:
        return np.zeros_like(tau_array, dtype=float)

    target_mass = (budget - rho * k) / (2.0 * w0)
    raw_pi = target_mass * weights / total_weight
    if np.all(raw_pi <= 1.0):
        return raw_pi.astype(float, copy=False)
    return _capped_proportional_allocation(weights, target_mass)


def theory_budget(
    pi: ArrayLike,
    rho: float,
    bonus: ArrayLike,
    effort: ArrayLike,
    w0: float,
    k: float,
    q_values: ArrayLike | None = None,
) -> float:
    """Main-text budget: sum_i ((rho*b_i*q(e_i)+w0)*pi_i)+rho*k."""

    pi_array = _validate_nonnegative_array("pi", pi)
    rho = _validate_rho(rho)
    bonus_array = _validate_nonnegative_array("bonus", bonus)
    _ = _validate_nonnegative_array("effort", effort)
    w0 = _validate_positive_scalar("w0", w0)
    k = _validate_nonnegative_scalar("k", k)
    q_array = (
        _validate_nonnegative_array("q_values", q_values)
        if q_values is not None
        else np.asarray(q_linear(effort), dtype=float)
    )

    pi_b, bonus_b, q_b = np.broadcast_arrays(pi_array, bonus_array, q_array)
    return float(np.sum((rho * bonus_b * q_b + w0) * pi_b) + rho * k)


def expected_cost_per_sample(
    rho: float,
    bonus: ArrayLike,
    effort: ArrayLike,
    w0: float,
    q_value: ArrayLike | None = None,
) -> float | np.ndarray:
    """Expected labeling cost per sampled task, excluding rho*k setup cost."""

    rho = _validate_rho(rho)
    bonus_array = _validate_nonnegative_array("bonus", bonus)
    effort_array = _validate_nonnegative_array("effort", effort)
    w0 = _validate_positive_scalar("w0", w0)
    q_array = (
        _validate_nonnegative_array("q_value", q_value)
        if q_value is not None
        else np.asarray(q_linear(effort_array), dtype=float)
    )

    bonus_b, q_b = np.broadcast_arrays(bonus_array, q_array)
    cost = rho * bonus_b * q_b + w0
    return _maybe_scalar(np.asarray(cost, dtype=float))


def misspecified_effort(
    effort: ArrayLike, kappa: float, clip: bool = True
) -> float | np.ndarray:
    """Scale designed effort by the actual utility multiplier kappa."""

    effort_array = _validate_nonnegative_array("effort", effort)
    kappa = _validate_nonnegative_scalar("kappa", kappa)
    actual_effort = kappa * effort_array
    if clip:
        actual_effort = np.clip(actual_effort, 0.0, 1.0)
    return _maybe_scalar(np.asarray(actual_effort, dtype=float))


__all__ = [
    "q_linear",
    "cost_quadratic",
    "risk_neutral_utility",
    "effort_from_bonus",
    "optimized_bonus",
    "risk_neutral_fixed_rho_design",
    "theory_budget",
    "expected_cost_per_sample",
    "misspecified_effort",
]
