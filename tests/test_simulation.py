from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incentive.simulation import (  # noqa: E402
    force_ai_error,
    label_accuracy_probability,
    simulate_human_labels,
)


def test_empirical_accuracy_matches_theory_over_many_bernoulli_trials():
    rng = np.random.default_rng(123)
    n = 60_000
    p_error = 0.35
    effort = 0.6
    y_true = rng.integers(0, 2, size=n)
    ai_wrong = rng.random(n) < p_error
    ai_prediction = np.where(ai_wrong, 1 - y_true, y_true)

    result = simulate_human_labels(y_true, ai_prediction, effort, rng)
    expected = label_accuracy_probability(p_error, effort)

    assert np.mean(result["correct"]) == pytest.approx(expected, abs=0.01)


def test_sentinel_binary_tasks_force_errors_before_correction():
    rng = np.random.default_rng(7)
    y_true = np.array([0, 1, 1, 0])
    ai_prediction = y_true.copy()
    sentinel = np.array([True, False, True, False])

    result = simulate_human_labels(
        y_true,
        ai_prediction,
        effort=0.0,
        rng=rng,
        sentinel=sentinel,
    )

    np.testing.assert_array_equal(result["displayed_ai"][sentinel], 1 - y_true[sentinel])
    np.testing.assert_array_equal(result["ai_error"], sentinel)
    np.testing.assert_array_equal(result["detected"], np.zeros_like(sentinel))
    np.testing.assert_array_equal(result["reported"][sentinel], 1 - y_true[sentinel])
    np.testing.assert_array_equal(result["correct"], ~sentinel)


def test_effort_zero_accepts_wrong_ai_and_effort_one_corrects_all_errors():
    y_true = np.array([0, 1, 0, 1])
    ai_prediction = 1 - y_true

    zero_effort = simulate_human_labels(
        y_true,
        ai_prediction,
        effort=0.0,
        rng=np.random.default_rng(10),
    )
    one_effort = simulate_human_labels(
        y_true,
        ai_prediction,
        effort=1.0,
        rng=np.random.default_rng(10),
    )

    np.testing.assert_array_equal(zero_effort["reported"], ai_prediction)
    np.testing.assert_array_equal(zero_effort["correct"], np.zeros_like(y_true, dtype=bool))
    np.testing.assert_array_equal(one_effort["reported"], y_true)
    np.testing.assert_array_equal(one_effort["detected"], np.ones_like(y_true, dtype=bool))
    np.testing.assert_array_equal(one_effort["correct"], np.ones_like(y_true, dtype=bool))


def test_continuous_sentinels_require_offset_or_false_label():
    y_true = np.array([1.25, 3.5])
    ai_prediction = y_true.copy()

    with pytest.raises(ValueError, match="continuous sentinel"):
        force_ai_error(y_true, ai_prediction, np.random.default_rng(1))
    with pytest.raises(ValueError, match="nonzero"):
        force_ai_error(
            y_true,
            ai_prediction,
            np.random.default_rng(1),
            continuous_sentinel_offset=0.0,
        )

    forced_offset = force_ai_error(
        y_true,
        ai_prediction,
        np.random.default_rng(1),
        continuous_sentinel_offset=0.5,
    )
    np.testing.assert_allclose(forced_offset, y_true + 0.5)

    forced_false_label = force_ai_error(
        y_true,
        ai_prediction,
        np.random.default_rng(1),
        false_label=np.array([9.0, 8.0]),
    )
    np.testing.assert_allclose(forced_false_label, [9.0, 8.0])


def test_simulation_broadcasts_scalar_effort_and_is_seed_deterministic():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    ai_prediction = np.array([0, 0, 1, 1, 0, 0])

    first = simulate_human_labels(
        y_true, ai_prediction, effort=0.4, rng=np.random.default_rng(222)
    )
    second = simulate_human_labels(
        y_true, ai_prediction, effort=0.4, rng=np.random.default_rng(222)
    )

    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


@pytest.mark.parametrize("effort", [-0.01, 1.01, np.inf])
def test_simulation_rejects_invalid_effort(effort):
    with pytest.raises(ValueError, match="effort"):
        simulate_human_labels(
            np.array([0, 1]),
            np.array([0, 1]),
            effort=effort,
            rng=np.random.default_rng(0),
        )


def test_simulation_rejects_shape_mismatches():
    with pytest.raises(ValueError, match="same shape"):
        simulate_human_labels(
            np.array([0, 1]),
            np.array([0, 1, 0]),
            effort=0.5,
            rng=np.random.default_rng(0),
        )
    with pytest.raises(ValueError, match="effort"):
        simulate_human_labels(
            np.array([0, 1]),
            np.array([0, 1]),
            effort=np.array([0.5, 0.5, 0.5]),
            rng=np.random.default_rng(0),
        )
