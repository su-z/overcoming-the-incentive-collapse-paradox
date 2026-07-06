from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments import pew  # noqa: E402


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


def _synthetic_pew_frame(n: int = 48, include_missing_outcome: bool = True) -> pd.DataFrame:
    row = np.arange(n)
    biden_yes = (row % 3) == 0
    biden = np.where(biden_yes, np.where(row % 2 == 0, 1, 2), np.where(row % 2 == 0, 3, 4))
    if include_missing_outcome:
        biden[-1] = 99

    trump_yes = (row % 4) <= 1
    trump = np.where(
        trump_yes,
        "The RIGHT message for this respondent",
        "The WRONG message for this respondent",
    )

    return pd.DataFrame(
        {
            "ELECTBIDENMSSG_W79": biden,
            "ELECTTRUMPMSSG_W79": trump,
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


def _pew_config() -> dict:
    return {
        "experiment": "pew_post_election_test",
        "seeds": [0, 1, 2],
        "smoke_seeds": [0, 1],
        "confidence_level": 0.90,
        "baseline_effort": 0.8,
        "data": {
            "raw_file": "unused.sav",
            "weight_column": "WEIGHT_W79",
            "outcomes": {
                "biden": {
                    "column_aliases": ["ELECTBIDENMSSG_W79", "ELECTBID"],
                    "estimand": "weighted_mean_approval",
                },
                "trump": {
                    "column_aliases": ["ELECTTRUMPMSSG_W79", "ELECTTRU"],
                    "estimand": "weighted_mean_approval",
                },
            },
            "covariates": COVARIATES,
        },
        "prediction_model": {"preferred_package": "sklearn"},
        "incentive": {
            "q": "linear",
            "cost": "quadratic",
            "utility": "risk_neutral",
            "w0": 0.64,
            "k": 1.0,
            "rho_grid": [0.10, 0.20, 0.30],
            "budget_model": "sum_i ((rho*b_i*q(e_i)+w0)*pi_i)+rho*k <= B",
        },
        "sampling": {
            "active_score": "sqrt_conditional_error",
            "baseline_tau_grid": [1.0, 0.0, 0.5],
            "baseline_tau_selection": "minimize_estimated_variance",
        },
        "budget_grid": {"smoke": [5.0, 9.0], "full": [20.0], "units": "expected_query_cost"},
    }


def test_recode_message_outcome_handles_numeric_codes_and_labels():
    series = pd.Series(
        [
            1,
            2.0,
            "3",
            4,
            99,
            np.nan,
            "Refused",
            "This was the RIGHT message",
            "This was the WRONG message",
            "Missing",
        ]
    )

    recoded = pew.recode_message_outcome(series)

    assert recoded.iloc[:4].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert recoded.iloc[4:7].isna().all()
    assert recoded.iloc[7] == 1.0
    assert recoded.iloc[8] == 0.0
    assert pd.isna(recoded.iloc[9])


def test_prepare_pew_dataset_uses_aliases_weights_and_one_hot_covariates(monkeypatch):
    frame = _synthetic_pew_frame(n=24, include_missing_outcome=True)
    config = _pew_config()
    monkeypatch.setattr(pew, "load_pew_wave79", lambda path: frame)

    prepared = pew.prepare_pew_dataset(config, "biden")

    assert prepared.outcome_column == "ELECTBIDENMSSG_W79"
    assert prepared.n == 23
    assert set(np.unique(prepared.y)) == {0.0, 1.0}
    assert prepared.X.shape[0] == prepared.n
    assert "WEIGHT_W79" in prepared.feature_names
    assert any(name.startswith("F_GENDER_") for name in prepared.feature_names)
    assert prepared.weights.shape == prepared.y.shape
    assert np.all(prepared.weights > 0.0)


def test_run_pew_mean_experiment_on_synthetic_data_writes_expected_schema(
    monkeypatch, tmp_path
):
    frame = _synthetic_pew_frame(n=48, include_missing_outcome=True)
    config = _pew_config()
    config_path = tmp_path / "pew.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(pew, "load_pew_wave79", lambda path: frame)
    monkeypatch.chdir(tmp_path)

    trials, summary = pew.run_pew_mean_experiment(config_path, outcome="biden", smoke=True)

    assert len(trials) == 2 * 2 * 4
    assert set(trials["method"]) == {"classical", "uniform", "active", "robust"}
    assert set(summary["method"]) == {"classical", "uniform", "active", "robust"}
    required_trial_columns = {
        "method",
        "budget",
        "seed",
        "estimate",
        "lower",
        "upper",
        "width",
        "covered",
        "outcome",
        "truth",
        "rho",
        "baseline_tau_mix",
        "expected_budget",
        "budget_model",
        "k_cost_interpretation",
    }
    assert required_trial_columns.issubset(trials.columns)
    assert np.all(trials["width"] >= 0.0)
    assert trials["budget_model"].eq("main_text").all()
    robust = trials[trials["method"] == "robust"]
    assert np.all(robust["rho"] > 0.0)
    assert np.all(robust["expected_budget"] <= robust["budget"] + 1e-9)

    trial_path = tmp_path / "outputs" / "tables" / "pew_biden_trials.csv"
    summary_path = tmp_path / "outputs" / "tables" / "pew_biden_summary.csv"
    assert trial_path.exists()
    assert summary_path.exists()
    assert len(pd.read_csv(trial_path)) == len(trials)
    assert len(pd.read_csv(summary_path)) == len(summary)


def test_active_design_tunes_mixture_instead_of_using_first_grid_value():
    tau = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
    norm_weights = np.full(tau.size, 1.0 / tau.size)

    design = pew._select_active_design(
        tau=tau,
        budget=1.5,
        w0=1.0,
        baseline_tau_grid=[1.0, 0.0, 0.5],
        norm_weights=norm_weights,
        q_effort=0.8,
    )

    uniform_pi = np.full(tau.size, 1.5 / tau.size)
    uniform_objective = pew._estimated_design_variance(
        tau, uniform_pi, norm_weights, q_effort=0.8, rho=0.0
    )
    assert design.mix == 0.0
    assert design.pi[0] > design.pi[1]
    assert design.objective < uniform_objective


def test_classical_correction_uses_final_report_accuracy_not_raw_effort():
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    f = np.array([0.9, 0.1, 0.3, 0.8])
    q_effort = 0.6
    label_accuracy = pew._final_label_accuracy_probability(y_true, f, q_effort)
    expected_report = np.where(y_true == 1.0, label_accuracy, 1.0 - label_accuracy)

    corrected = pew._mean_contributions(
        "classical",
        expected_report,
        f,
        xi=np.ones_like(y_true),
        zeta=np.ones_like(y_true),
        pi=np.ones_like(y_true),
        rho=0.0,
        q_effort=q_effort,
        classical_label_accuracy=label_accuracy,
    )
    raw_effort_corrected = pew._mean_contributions(
        "classical",
        expected_report,
        f,
        xi=np.ones_like(y_true),
        zeta=np.ones_like(y_true),
        pi=np.ones_like(y_true),
        rho=0.0,
        q_effort=q_effort,
    )

    np.testing.assert_allclose(corrected, y_true)
    assert not np.allclose(raw_effort_corrected, y_true)


def test_smoke_cli_skips_missing_spss_when_explicitly_allowed(monkeypatch, capsys):
    def raise_missing(*args, **kwargs):
        raise ImportError("optional pyreadstat is unavailable")

    monkeypatch.setattr(pew, "run_pew_mean_experiment", raise_missing)

    exit_code = pew._main(
        ["--config", "configs/pew.yaml", "--outcome", "biden", "--smoke", "--allow-missing-data"]
    )

    assert exit_code == 0
    assert "Skipping Pew smoke run" in capsys.readouterr().out
    with pytest.raises(ImportError, match="pyreadstat"):
        pew._main(["--config", "configs/pew.yaml", "--outcome", "biden", "--smoke"])
