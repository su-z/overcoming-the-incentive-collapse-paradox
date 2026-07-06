from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.common import (  # noqa: E402
    TrialResult,
    budget_savings_curve,
    crossfit_binary_predictions,
    crossfit_regression_predictions,
    summarize_intervals,
    tau_from_binary_predictions,
    tau_from_regression_residuals,
)


def test_crossfit_predictions_are_deterministic_and_have_expected_shape():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(50, 4))
    logits = X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2]
    y_binary = (logits > np.median(logits)).astype(float)
    y_regression = 1.5 * X[:, 0] - X[:, 2] + rng.normal(scale=0.05, size=50)
    seeds = [3, 7]

    first_binary = crossfit_binary_predictions(X, y_binary, seeds, model="sklearn")
    second_binary = crossfit_binary_predictions(X, y_binary, seeds, model="sklearn")
    first_regression = crossfit_regression_predictions(X, y_regression, seeds, model="sklearn")
    second_regression = crossfit_regression_predictions(
        X, y_regression, seeds, model="sklearn"
    )

    assert first_binary.predictions.shape == (len(seeds), X.shape[0])
    assert first_regression.predictions.shape == (len(seeds), X.shape[0])
    assert first_binary.tau.shape == (X.shape[0],)
    assert first_regression.tau.shape == (X.shape[0],)
    np.testing.assert_allclose(first_binary.predictions, second_binary.predictions)
    np.testing.assert_allclose(first_regression.predictions, second_regression.predictions)
    assert np.all((first_binary.predictions >= 0.0) & (first_binary.predictions <= 1.0))


def test_tau_helpers_are_positive_even_for_exact_predictions():
    binary_tau = tau_from_binary_predictions(
        y_true=np.array([0.0, 1.0, 1.0]),
        y_prob=np.array([0.0, 1.0, 0.25]),
    )
    regression_tau = tau_from_regression_residuals(
        y_true=np.array([1.0, 2.0, 4.0]),
        y_pred=np.array([1.0, 1.5, 5.0]),
    )

    assert np.all(binary_tau > 0.0)
    assert np.all(regression_tau > 0.0)
    assert binary_tau[0] == pytest.approx(1e-6)
    assert regression_tau[0] == pytest.approx(1e-6)
    assert binary_tau[2] == pytest.approx((1.0 - 0.25) ** 2)


def test_summarize_intervals_reports_average_width_and_empirical_coverage():
    trials = [
        TrialResult(
            method="robust",
            budget=100.0,
            estimate=0.50,
            lower=0.40,
            upper=0.60,
            seed=0,
            metadata={"experiment": "synthetic"},
        ),
        TrialResult(
            method="robust",
            budget=100.0,
            estimate=0.80,
            lower=0.70,
            upper=0.90,
            seed=1,
            metadata={"experiment": "synthetic"},
        ),
    ]

    summary = summarize_intervals(trials, truth=0.50)

    assert list(summary.columns[:8]) == [
        "method",
        "budget",
        "estimate",
        "lower",
        "upper",
        "width",
        "covered",
        "seed",
    ]
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["method"] == "robust"
    assert row["width"] == pytest.approx(0.20)
    assert row["covered"] == pytest.approx(0.50)
    assert row["seed"] == (0, 1)
    assert row["experiment"] == "synthetic"
    assert row["budget_model"] == "main_text"


def test_budget_savings_curve_interpolates_monotone_target_budget():
    budgets = np.array([10.0, 20.0, 30.0])
    widths_by_method = {
        "robust": np.array([8.0, 4.0, 2.0]),
        "uniform": np.array([9.0, 7.0, 5.0]),
    }

    savings = budget_savings_curve(
        widths_by_method,
        budgets,
        reference_method="uniform",
        target_method="robust",
    )

    assert list(savings.columns[:8]) == [
        "method",
        "budget",
        "estimate",
        "lower",
        "upper",
        "width",
        "covered",
        "seed",
    ]
    np.testing.assert_allclose(savings["target_budget"], [10.0, 12.5, 17.5])
    assert np.all(np.diff(savings["target_budget"]) >= 0.0)
    assert np.all(savings["savings_fraction"] >= 0.0)
    assert savings.loc[1, "estimate"] == pytest.approx((20.0 - 12.5) / 20.0)
    assert savings.loc[0, "budget_model"] == "main_text"
