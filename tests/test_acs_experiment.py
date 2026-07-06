from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments import acs  # noqa: E402
from experiments.acs import (  # noqa: E402
    acs_truth_linear_coef,
    prepare_acs_dataset,
    run_acs_experiment,
)


def _synthetic_acs_frame(n: int = 80) -> pd.DataFrame:
    age = np.linspace(18.0, 78.0, n)
    sex = np.where(np.arange(n) % 2 == 0, 1.0, 2.0)
    education = np.arange(n) % 5
    income = 1000.0 + 125.0 * age + 750.0 * (sex == 2.0)
    return pd.DataFrame(
        {
            "PINCP": income,
            "AGEP": age,
            "SEX": sex,
            "SCHL": education,
        }
    )


def _write_config(tmp_path: Path, csv_path: Path, budget: float = 80.0) -> Path:
    config = {
        "experiment": "acs_income_regression_test",
        "seeds": [0],
        "smoke_seeds": [0],
        "confidence_level": 0.90,
        "baseline_effort": 1.0,
        "data": {
            "raw_file": str(csv_path),
            "outcome": "PINCP",
            "target_covariates": ["AGEP", "SEX"],
            "preprocessing": {
                "smoke_sample_size_cap": 200,
            },
        },
        "prediction_model": {
            "family": "linear_regression",
            "features": ["AGEP", "SEX", "SCHL"],
            "crossfit_seeds": [0],
        },
        "incentive": {
            "w0": 1.0,
            "k": 0.0,
            "rho_grid": [0.1],
        },
        "sampling": {
            "baseline_tau_grid": [0.0, 1.0],
        },
        "budget_grid": {
            "smoke": [budget],
            "full": [budget],
        },
        "outputs": {
            "tables_dir": str(tmp_path / "outputs" / "tables"),
        },
        "assumptions": ["synthetic ACS test fixture"],
    }
    config_path = tmp_path / "acs.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_acs_truth_linear_coef_recovers_age_coefficient():
    df = _synthetic_acs_frame()

    coefficient = acs_truth_linear_coef(df)

    assert coefficient == pytest.approx(125.0, abs=1e-9)


def test_prepare_acs_dataset_smoke_uses_nrows_cap_and_filters(monkeypatch):
    calls = {}

    def fake_loader(path, columns=None, nrows=None):
        calls["path"] = path
        calls["columns"] = columns
        calls["nrows"] = nrows
        return pd.DataFrame(
            {
                "PINCP": [10.0, 20.0, np.nan, 40.0],
                "AGEP": [17.0, 25.0, 30.0, 35.0],
                "SEX": [1.0, 2.0, 1.0, np.inf],
                "SCHL": [1.0, 2.0, 3.0, 4.0],
            }
        )

    monkeypatch.setattr(acs, "load_acs_pums", fake_loader)
    config = {
        "data": {
            "raw_file": "unused.csv",
            "outcome": "PINCP",
            "target_covariates": ["AGEP", "SEX"],
            "preprocessing": {
                "adult_age_min": 18,
                "smoke_sample_size_cap": 2,
            },
        },
        "prediction_model": {
            "features": ["AGEP", "SEX", "SCHL"],
        },
    }

    prepared = prepare_acs_dataset(config, smoke=True)

    assert calls["nrows"] == 2
    assert calls["columns"] == ["PINCP", "AGEP", "SEX", "SCHL"]
    assert prepared["AGEP"].tolist() == [25.0]
    assert prepared["PINCP"].tolist() == [20.0]


def test_acs_truth_rejects_singular_target_design():
    df = pd.DataFrame(
        {
            "PINCP": [1.0, 2.0, 3.0],
            "AGEP": [30.0, 30.0, 30.0],
            "SEX": [1.0, 1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="singular"):
        acs_truth_linear_coef(df)


def test_run_acs_experiment_writes_expected_schema_and_recovers_target(tmp_path):
    csv_path = tmp_path / "synthetic_acs.csv"
    _synthetic_acs_frame().to_csv(csv_path, index=False)
    config_path = _write_config(tmp_path, csv_path)

    result = run_acs_experiment(config_path=config_path, smoke=True)

    trials_path = tmp_path / "outputs" / "tables" / "acs_trials.csv"
    summary_path = tmp_path / "outputs" / "tables" / "acs_summary.csv"
    assert result["trials_path"] == trials_path
    assert result["summary_path"] == summary_path
    assert trials_path.exists()
    assert summary_path.exists()

    trials = pd.read_csv(trials_path)
    summary = pd.read_csv(summary_path)
    expected_columns = {
        "method",
        "budget",
        "estimate",
        "lower",
        "upper",
        "width",
        "covered",
        "seed",
        "coefficient",
        "budget_model",
        "k_cost_interpretation",
        "expected_budget",
        "expected_queries",
        "budget_edge_case",
        "n_rows",
    }
    assert expected_columns.issubset(trials.columns)
    assert expected_columns.union({"n_trials"}).issubset(summary.columns)
    assert set(trials["method"]) == {"classical", "uniform", "active", "robust"}
    assert set(summary["method"]) == {"classical", "uniform", "active", "robust"}
    assert np.all(np.isfinite(trials["estimate"]))
    assert np.all(trials["width"] >= 0.0)
    assert trials["budget_model"].eq("main_text").all()
    assert np.all(trials["expected_budget"] <= trials["budget"] + 1e-9)
    assert trials["budget_edge_case"].eq("none").all()
    assert result["truth"] == pytest.approx(125.0, abs=1e-8)
    assert trials.set_index("method").loc["robust", "estimate"] == pytest.approx(125.0, abs=1e-8)


def test_acs_baseline_budgets_include_effort_payment_cost():
    tau = np.linspace(1.0, 4.0, 80)
    budget = 12.5
    w0 = 0.5
    config = {
        "incentive": {
            "w0": w0,
            "k": 0.0,
            "rho_grid": [0.25],
        },
        "sampling": {
            "baseline_tau_grid": [0.0],
        },
    }

    designs = acs._method_designs(tau, budget, baseline_effort=0.8, config=config)

    baseline_cost = w0 + 0.8**2
    for method in ("classical", "uniform", "active"):
        assert np.sum(designs[method]["pi"]) == pytest.approx(budget / baseline_cost)
        assert designs[method]["expected_budget"] == pytest.approx(budget)

    assert designs["robust"]["expected_budget"] == pytest.approx(budget)
    assert np.sum(designs["robust"]["pi"]) == pytest.approx(budget / (2.0 * w0))
    for method, design in designs.items():
        assert design["expected_budget"] <= budget + 1e-9, method
        assert design["budget_edge_case"] == "none"
