"""Human label simulation utilities for AI-assisted tasks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


ArrayLike = Any


def _maybe_scalar(value: np.ndarray) -> float | np.ndarray:
    if value.ndim == 0:
        return float(value)
    return value


def _call_q(
    q_fn: Callable[[ArrayLike], ArrayLike] | None, effort: np.ndarray
) -> np.ndarray:
    if q_fn is None:
        q_values = effort
    else:
        try:
            q_values = q_fn(effort)
        except (TypeError, ValueError):
            q_values = np.vectorize(q_fn, otypes=[float])(effort)
    q_array = np.asarray(q_values, dtype=float)
    if q_array.shape == ():
        q_array = np.full(effort.shape, float(q_array))
    elif q_array.shape != effort.shape:
        raise ValueError("q_fn must return a scalar or an array matching effort")
    if np.any(~np.isfinite(q_array)) or np.any((q_array < 0.0) | (q_array > 1.0)):
        raise ValueError("q(e) must contain probabilities in [0, 1]")
    return q_array


def _validate_effort(effort: ArrayLike, shape: tuple[int, ...]) -> np.ndarray:
    effort_array = np.asarray(effort, dtype=float)
    if effort_array.shape == ():
        effort_array = np.full(shape, float(effort_array))
    elif effort_array.shape != shape:
        raise ValueError("effort must be a scalar or match y_true shape")
    if np.any(~np.isfinite(effort_array)) or np.any(
        (effort_array < 0.0) | (effort_array > 1.0)
    ):
        raise ValueError("effort must contain finite values in [0, 1]")
    return effort_array


def _validate_probability(name: str, value: ArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return array


def _validate_same_shape(
    y_true: ArrayLike, ai_prediction: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    y_array = np.asarray(y_true)
    ai_array = np.asarray(ai_prediction)
    if y_array.shape != ai_array.shape:
        raise ValueError("y_true and ai_prediction must have the same shape")
    return y_array, ai_array


def _validate_optional_item_array(
    name: str, value: ArrayLike | None, shape: tuple[int, ...]
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape == ():
        return array
    if array.shape != shape:
        raise ValueError(f"{name} must be a scalar or match y_true shape")
    return array


def _select_optional(
    value: ArrayLike | None, mask: np.ndarray, shape: tuple[int, ...], name: str
) -> ArrayLike | None:
    array = _validate_optional_item_array(name, value, shape)
    if array is None or array.shape == ():
        return array
    return array[mask]


def _is_binary_labels(y_true: np.ndarray) -> bool:
    return bool(np.all((y_true == 0) | (y_true == 1)))


def _nonzero_offset(
    offset: ArrayLike | None, shape: tuple[int, ...]
) -> np.ndarray | None:
    if offset is None:
        return None
    offset_array = np.asarray(offset, dtype=float)
    if offset_array.shape == ():
        offset_array = np.full(shape, float(offset_array))
    elif offset_array.shape != shape:
        raise ValueError(
            "continuous_sentinel_offset must be a scalar or match y_true shape"
        )
    if np.any(~np.isfinite(offset_array)) or np.any(offset_array == 0.0):
        raise ValueError("continuous_sentinel_offset must be finite and nonzero")
    return offset_array


def _rng_random(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    try:
        return rng.random(shape)
    except AttributeError as exc:
        raise TypeError("rng must support numpy Generator-style random draws") from exc


def label_accuracy_probability(
    ai_error_probability: ArrayLike,
    effort: ArrayLike,
    q_fn: Callable[[ArrayLike], ArrayLike] | None = None,
) -> float | np.ndarray:
    """Return 1 - p*(1-q(e)) for the paper's label-generation model."""

    p_error = _validate_probability("ai_error_probability", ai_error_probability)
    raw_effort = np.asarray(effort, dtype=float)
    if np.any(~np.isfinite(raw_effort)) or np.any(
        (raw_effort < 0.0) | (raw_effort > 1.0)
    ):
        raise ValueError("effort must contain finite values in [0, 1]")
    try:
        p_error, effort_array = np.broadcast_arrays(p_error, raw_effort)
    except ValueError as exc:
        raise ValueError(
            "ai_error_probability and effort must be broadcastable"
        ) from exc
    q_values = _call_q(q_fn, effort_array)
    accuracy = 1.0 - p_error * (1.0 - q_values)
    return _maybe_scalar(np.asarray(accuracy, dtype=float))


def force_ai_error(
    y_true: ArrayLike,
    ai_prediction: ArrayLike,
    rng: np.random.Generator,
    false_label: ArrayLike | None = None,
    continuous_sentinel_offset: ArrayLike | None = None,
) -> np.ndarray:
    """Construct displayed AI labels that are wrong for every item."""

    del rng
    y_array, _ = _validate_same_shape(y_true, ai_prediction)

    if false_label is not None:
        forced = _validate_optional_item_array("false_label", false_label, y_array.shape)
        forced = np.full(y_array.shape, forced.item()) if forced.shape == () else forced
    elif _is_binary_labels(y_array):
        forced = np.where(y_array == 1, 0, 1)
    else:
        offset = _nonzero_offset(continuous_sentinel_offset, y_array.shape)
        if offset is None:
            raise ValueError(
                "continuous sentinel tasks require false_label or nonzero "
                "continuous_sentinel_offset"
            )
        forced = y_array.astype(float, copy=False) + offset

    forced_array = np.asarray(forced)
    if forced_array.shape != y_array.shape:
        raise ValueError("forced labels must match y_true shape")
    if np.any(forced_array == y_array):
        raise ValueError("forced AI labels must differ from y_true")
    return forced_array.copy()


def simulate_human_labels(
    y_true: ArrayLike,
    ai_prediction: ArrayLike,
    effort: ArrayLike,
    rng: np.random.Generator,
    q_fn: Callable[[ArrayLike], ArrayLike] | None = None,
    sentinel: ArrayLike | None = None,
    false_label: ArrayLike | None = None,
    continuous_sentinel_offset: ArrayLike | None = None,
) -> dict[str, np.ndarray]:
    """Simulate AI-assisted human reports under effort-dependent correction."""

    y_array, ai_array = _validate_same_shape(y_true, ai_prediction)
    effort_array = _validate_effort(effort, y_array.shape)
    q_values = _call_q(q_fn, effort_array)

    if sentinel is None:
        sentinel_array = np.zeros(y_array.shape, dtype=bool)
    else:
        raw_sentinel = np.asarray(sentinel, dtype=bool)
        if raw_sentinel.shape == ():
            sentinel_array = np.full(y_array.shape, bool(raw_sentinel))
        elif raw_sentinel.shape == y_array.shape:
            sentinel_array = raw_sentinel
        else:
            raise ValueError("sentinel must be a scalar or match y_true shape")

    displayed_ai = ai_array.copy()
    if np.any(sentinel_array):
        forced = force_ai_error(
            y_array[sentinel_array],
            ai_array[sentinel_array],
            rng,
            false_label=_select_optional(
                false_label, sentinel_array, y_array.shape, "false_label"
            ),
            continuous_sentinel_offset=_select_optional(
                continuous_sentinel_offset,
                sentinel_array,
                y_array.shape,
                "continuous_sentinel_offset",
            ),
        )
        displayed_ai = displayed_ai.astype(np.result_type(displayed_ai, forced), copy=False)
        displayed_ai[sentinel_array] = forced

    ai_error = displayed_ai != y_array
    draws = _rng_random(rng, y_array.shape)
    detected = ai_error & (draws < q_values)
    reported = np.where(detected, y_array, displayed_ai)
    correct = reported == y_array

    return {
        "reported": np.asarray(reported).copy(),
        "displayed_ai": np.asarray(displayed_ai).copy(),
        "ai_error": np.asarray(ai_error, dtype=bool),
        "detected": np.asarray(detected, dtype=bool),
        "sentinel": np.asarray(sentinel_array, dtype=bool),
        "correct": np.asarray(correct, dtype=bool),
    }


__all__ = [
    "force_ai_error",
    "label_accuracy_probability",
    "simulate_human_labels",
]
