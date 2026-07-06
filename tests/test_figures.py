from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plotting.figures import (  # noqa: E402
    _budget_for_matching_width,
    _budget_savings_points,
    _select_representative_trials,
    plot_all_figures,
    plot_budget_savings,
    plot_interval_width_coverage,
)


def _summary_frame() -> pd.DataFrame:
    rows = []
    widths = {
        "classical": [12.0, 8.0, 4.0],
        "uniform": [11.0, 7.0, 3.5],
        "active": [10.0, 6.0, 3.0],
        "robust": [8.0, 4.0, 2.0],
    }
    coverage = {
        "classical": [0.90, 0.95, 0.90],
        "uniform": [0.85, 0.90, 0.95],
        "active": [0.90, 0.90, 0.90],
        "robust": [0.95, 0.90, 0.90],
    }
    for method, values in widths.items():
        for budget, width, covered in zip([10.0, 20.0, 40.0], values, coverage[method], strict=True):
            estimate = 0.5 + 0.01 * budget
            rows.append(
                {
                    "method": method,
                    "budget": budget,
                    "estimate": estimate,
                    "lower": estimate - width / 2.0,
                    "upper": estimate + width / 2.0,
                    "width": width,
                    "covered": covered,
                    "seed": "(0, 1)",
                    "n_trials": 2,
                    "truth": 0.75,
                    "confidence_level": 0.90,
                    "budget_model": "main_text",
                    "k_cost_interpretation": "fixed_global_sentinel_cost",
                }
            )
    return pd.DataFrame(rows)


def _trials_frame() -> pd.DataFrame:
    rows = []
    for seed in [0, 1]:
        for method_index, method in enumerate(["classical", "uniform", "active", "robust"]):
            for budget in [10.0, 20.0, 40.0]:
                width = 10.0 - method_index - budget / 10.0
                estimate = 0.7 + 0.02 * method_index + 0.01 * seed
                rows.append(
                    {
                        "method": method,
                        "budget": budget,
                        "estimate": estimate,
                        "lower": estimate - width / 2.0,
                        "upper": estimate + width / 2.0,
                        "width": width,
                        "covered": True,
                        "seed": seed,
                        "truth": 0.75,
                        "confidence_level": 0.90,
                    }
                )
    return pd.DataFrame(rows)


def test_plot_interval_width_coverage_writes_real_output(tmp_path: Path):
    summary_csv = tmp_path / "summary.csv"
    trials_csv = tmp_path / "trials.csv"
    output_path = tmp_path / "figure.pdf"
    _summary_frame().to_csv(summary_csv, index=False)
    _trials_frame().to_csv(trials_csv, index=False)

    returned = plot_interval_width_coverage(summary_csv, trials_csv, output_path, "Synthetic performance")

    assert returned == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_representative_interval_selection_keeps_multiple_random_trials():
    trials = _trials_frame()

    selection = _select_representative_trials(
        trials,
        ["classical", "uniform", "active", "robust"],
    )

    assert selection.budget == pytest.approx(20.0)
    assert len(selection.frame) == 8
    assert set(selection.frame["seed"]) == {0, 1}
    assert selection.frame.groupby("method")["seed"].nunique().to_dict() == {
        "active": 2,
        "classical": 2,
        "robust": 2,
        "uniform": 2,
    }


def test_budget_savings_interpolates_matching_widths_and_writes_output(tmp_path: Path):
    summary_csv = tmp_path / "summary.csv"
    output_path = tmp_path / "budget.png"
    _summary_frame().to_csv(summary_csv, index=False)

    assert _budget_for_matching_width(
        np.array([10.0, 20.0, 40.0]),
        np.array([12.0, 8.0, 4.0]),
        target_width=6.0,
    ) == pytest.approx(30.0)

    returned = plot_budget_savings(summary_csv, output_path)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_budget_savings_uses_baseline_reference_budgets():
    points = _budget_savings_points(
        baseline_budgets=np.array([10.0, 20.0, 40.0]),
        baseline_widths=np.array([8.0, 6.0, 4.0]),
        robust_budgets=np.array([10.0, 20.0, 40.0]),
        robust_widths=np.array([2.0, 1.5, 1.0]),
    )

    np.testing.assert_allclose(
        points,
        np.array(
            [
                [10.0, 93.75],
                [20.0, 100.0 * (20.0 - 10.0 * (2.0 / 6.0) ** 2) / 20.0],
                [40.0, 100.0 * (40.0 - 10.0 * (2.0 / 4.0) ** 2) / 40.0],
            ]
        ),
    )


def test_budget_savings_interpolates_robust_budget_needed_for_baseline_width():
    points = _budget_savings_points(
        baseline_budgets=np.array([10.0, 20.0, 30.0]),
        baseline_widths=np.array([9.0, 7.0, 5.0]),
        robust_budgets=np.array([10.0, 20.0, 30.0]),
        robust_widths=np.array([8.0, 4.0, 2.0]),
    )

    np.testing.assert_allclose(
        points,
        np.array(
            [
                [10.0, 100.0 * (10.0 - 10.0 * (8.0 / 9.0) ** 2) / 10.0],
                [20.0, 37.5],
                [30.0, 100.0 * (30.0 - 17.5) / 30.0],
            ]
        ),
    )


def test_plot_all_figures_generates_available_inputs_and_warns_for_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    tables_dir = tmp_path / "outputs" / "tables"
    tables_dir.mkdir(parents=True)
    _summary_frame().to_csv(tables_dir / "protein_summary.csv", index=False)
    _trials_frame().to_csv(tables_dir / "protein_trials.csv", index=False)

    with pytest.warns(RuntimeWarning, match="Skipping"):
        generated = plot_all_figures(output_dir=tmp_path / "figures")

    expected = {
        tmp_path / "figures" / "widths_and_coverage_alphafold_robust_uniform_payment.pdf",
        tmp_path / "figures" / "budget_alphafold_robust_uniform_payment.pdf",
    }
    assert set(generated) == expected
    for path in expected:
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_interval_width_coverage_rejects_missing_required_columns(tmp_path: Path):
    summary_csv = tmp_path / "summary.csv"
    trials_csv = tmp_path / "trials.csv"
    output_path = tmp_path / "figure.pdf"
    pd.DataFrame({"method": ["robust"], "budget": [10.0], "width": [1.0]}).to_csv(
        summary_csv, index=False
    )
    _trials_frame().to_csv(trials_csv, index=False)

    with pytest.raises(ValueError, match="covered"):
        plot_interval_width_coverage(summary_csv, trials_csv, output_path, "Bad summary")
