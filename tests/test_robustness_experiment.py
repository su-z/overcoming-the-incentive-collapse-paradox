from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments import pew, robustness  # noqa: E402


COVARIATES = [
    "F_AGECAT",
    "F_GENDER",
    "F_EDUCCAT",
    "F_RACETHNMOD",
    "F_PARTYSUM_FINAL",
    "F_IDEO",
    "F_INC_SDT1",
    "F_CREGION",
    "F_REG",
    "WEIGHT_W79",
]


def _synthetic_pew_frame(n: int = 48) -> pd.DataFrame:
    row = np.arange(n)
    biden_yes = (row % 3) == 0
    biden = np.where(biden_yes, np.where(row % 2 == 0, 1, 2), np.where(row % 2 == 0, 3, 4))
    biden[-1] = 99

    return pd.DataFrame(
        {
            "ELECTBIDENMSSG_W79": biden,
            "F_AGECAT": np.where(row % 2 == 0, "18-29", "30-49"),
            "F_GENDER": np.where(row % 2 == 0, "Woman", "Man"),
            "F_EDUCCAT": np.where(row % 3 == 0, "College", "No college"),
            "F_RACETHNMOD": np.where(row % 4 == 0, "White", "Other"),
            "F_PARTYSUM_FINAL": np.where(row % 3 == 0, "Dem", np.where(row % 3 == 1, "Rep", "Ind")),
            "F_IDEO": np.where(row % 2 == 0, "Liberal", "Conservative"),
            "F_INC_SDT1": np.where(row % 5 == 0, "High", "Middle"),
            "F_CREGION": np.where(row % 2 == 0, "Northeast", "South"),
            "F_REG": np.where(row % 3 == 0, "Urban", "Suburban"),
            "WEIGHT_W79": 0.75 + (row % 5) * 0.1,
        }
    )


def _robustness_config(
    *, posterior_scale: float = 0.75, kappa_uniform: list[float] | None = None
) -> dict:
    return {
        "experiment": "pew_robustness_test",
        "seeds": [0, 1],
        "smoke_seeds": [0, 1],
        "confidence_level": 0.90,
        "baseline_effort": 0.8,
        "data": {
            "raw_file": "unused.sav",
            "weight_column": "WEIGHT_W79",
            "outcome": {
                "name": "biden",
                "column_aliases": ["ELECTBIDENMSSG_W79", "ELECTBID"],
                "estimand": "weighted_mean_approval",
            },
            "covariates": COVARIATES,
        },
        "prediction_model": {"preferred_package": "sklearn"},
        "incentive": {
            "w0": 0.64,
            "k": 1.0,
            "rho_grid": [0.10, 0.20, 0.30],
            "budget_model": "sum_i ((rho*b_i*q(e_i)+w0)*pi_i)+rho*k <= B",
        },
        "sampling": {
            "baseline_tau_grid": [1.0, 0.0, 0.5],
            "baseline_tau_selection": "minimize_estimated_variance",
        },
        "robustness": {
            "posterior_belief": {"posterior_scale": posterior_scale},
            "misspecification": {"kappa_uniform": kappa_uniform or [0.30, 0.40]},
        },
        "budget_grid": {"smoke": [5.0, 9.0], "full": [20.0], "units": "expected_query_cost"},
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "robustness.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_posterior_belief_scales_robust_regular_effort_and_writes_schema(
    monkeypatch, tmp_path
):
    config_path = _write_config(tmp_path, _robustness_config(posterior_scale=0.75))
    monkeypatch.setattr(pew, "load_pew_wave79", lambda path: _synthetic_pew_frame())
    monkeypatch.chdir(tmp_path)

    trials, summary = robustness.run_posterior_belief_experiment(config_path, smoke=True)

    assert len(trials) == 2 * 2 * 4
    assert set(trials["method"]) == {"classical", "uniform", "active", "robust"}
    required_columns = {
        "method",
        "budget",
        "lower",
        "upper",
        "width",
        "covered",
        "sensitivity",
        "predicted_effort",
        "actual_effort",
        "posterior_scale",
        "kappa",
        "budget_model",
    }
    assert required_columns.issubset(trials.columns)
    assert required_columns.difference({"kappa"}).issubset(summary.columns)
    assert trials["sensitivity"].eq("posterior").all()
    assert trials["confidence_level"].eq(0.90).all()
    assert np.all(trials["width"] >= 0.0)

    robust = trials[trials["method"] == "robust"]
    np.testing.assert_allclose(robust["actual_effort"], 0.75 * robust["predicted_effort"])
    assert robust["posterior_scale"].eq(0.75).all()
    assert np.all(robust["expected_budget"] <= robust["budget"] + 1e-9)

    baselines = trials[trials["method"] != "robust"]
    np.testing.assert_allclose(baselines["actual_effort"], baselines["predicted_effort"])
    assert (tmp_path / "outputs" / "tables" / "robustness_posterior_trials.csv").exists()
    assert (tmp_path / "outputs" / "tables" / "robustness_posterior_summary.csv").exists()


def test_misspecification_draws_deterministic_kappa_and_scales_robust_effort(
    monkeypatch, tmp_path
):
    config_path = _write_config(tmp_path, _robustness_config(kappa_uniform=[0.30, 0.40]))
    monkeypatch.setattr(pew, "load_pew_wave79", lambda path: _synthetic_pew_frame())
    monkeypatch.chdir(tmp_path)

    first_trials, first_summary = robustness.run_misspecification_experiment(config_path, smoke=True)
    second_trials, second_summary = robustness.run_misspecification_experiment(config_path, smoke=True)

    first_robust = first_trials[first_trials["method"] == "robust"].sort_values(["seed", "budget"])
    second_robust = second_trials[second_trials["method"] == "robust"].sort_values(["seed", "budget"])
    np.testing.assert_allclose(first_robust["kappa"], second_robust["kappa"])
    assert np.all((first_robust["kappa"] >= 0.30) & (first_robust["kappa"] <= 0.40))
    np.testing.assert_allclose(
        first_robust["actual_effort"],
        first_robust["predicted_effort"] * first_robust["kappa"],
    )

    baselines = first_trials[first_trials["method"] != "robust"]
    assert baselines["kappa"].isna().all()
    np.testing.assert_allclose(baselines["actual_effort"], baselines["predicted_effort"])
    assert first_trials["sensitivity"].eq("misspecification").all()
    assert first_summary["covered"].between(0.0, 1.0).all()
    pd.testing.assert_frame_equal(first_trials, second_trials)
    pd.testing.assert_frame_equal(first_summary, second_summary)
    assert (tmp_path / "outputs" / "tables" / "robustness_misspecification_trials.csv").exists()
    assert (tmp_path / "outputs" / "tables" / "robustness_misspecification_summary.csv").exists()


def test_cli_dispatch_and_allow_missing_data(monkeypatch, capsys):
    captured = {}

    def fake_posterior(config_path, smoke):
        captured["posterior"] = (config_path, smoke)
        return pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]})

    monkeypatch.setattr(robustness, "run_posterior_belief_experiment", fake_posterior)

    exit_code = robustness._main(["--config", "custom.yaml", "--which", "posterior", "--smoke"])

    assert exit_code == 0
    assert captured["posterior"] == ("custom.yaml", True)
    assert "robustness posterior" in capsys.readouterr().out

    def raise_missing(*args, **kwargs):
        raise ImportError("optional pyreadstat is unavailable")

    monkeypatch.setattr(robustness, "run_posterior_belief_experiment", raise_missing)
    exit_code = robustness._main(
        ["--config", "custom.yaml", "--which", "posterior", "--smoke", "--allow-missing-data"]
    )

    assert exit_code == 0
    assert "Skipping robustness smoke run" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        robustness._main(["--which", "unknown"])


def test_unknown_sensitivity_names_fail():
    with pytest.raises(ValueError, match="unknown robustness sensitivity"):
        robustness._normalize_which("not-a-sensitivity")
    with pytest.raises(ValueError, match="unknown robustness sensitivity"):
        robustness._output_paths("not-a-sensitivity")
