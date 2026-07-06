"""ACS PUMS continuous-outcome income regression experiment.

This module implements the appendix ACS experiment from Yin, Su, and Li,
"Overcoming the Incentive Collapse Paradox" (ICML 2026). The target is the
population least-squares coefficient of ``AGEP`` in a regression of individual
income ``PINCP`` on age and sex.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from experiments.common import (
    TrialResult,
    crossfit_regression_predictions,
    load_yaml_config,
    make_rng,
    summarize_intervals,
)
from incentive.design import (
    effort_from_bonus,
    optimized_bonus,
    risk_neutral_fixed_rho_design,
    theory_budget,
)
from incentive.simulation import simulate_human_labels
from inference.sampling import mixed_active_uniform_probabilities, uniform_probabilities


__all__ = [
    "load_acs_pums",
    "prepare_acs_dataset",
    "acs_truth_linear_coef",
    "run_acs_experiment",
]


_OUTCOME = "PINCP"
_TARGET_COVARIATES = ("AGEP", "SEX")
_AGE_COEFFICIENT_NAME = "AGEP"
_METHODS = ("classical", "uniform", "active", "robust")
_Z_TOL = 1e-12


def load_acs_pums(
    path: str | Path,
    columns: Sequence[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load an ACS PUMS CSV file.

    ``columns`` is passed through as ``usecols`` so smoke tests and explicit
    feature configurations can avoid reading unnecessary columns from the large
    raw ACS file.
    """

    csv_path = Path(path)
    usecols = None if columns is None else list(dict.fromkeys(columns))
    return pd.read_csv(csv_path, usecols=usecols, nrows=nrows, low_memory=False)


def prepare_acs_dataset(config: Mapping[str, Any] | str | Path, smoke: bool = False) -> pd.DataFrame:
    """Load and filter the ACS income regression dataset.

    Defaults match the ICML paper and config assumptions: finite ``PINCP``,
    finite target covariates, and adult records with ``AGEP >= 18``. Income is
    left on its raw scale unless the config explicitly requests scaling.
    """

    config_dict = _coerce_config(config)
    data_config = _mapping(config_dict.get("data"))
    preprocessing = _mapping(data_config.get("preprocessing"))
    outcome = str(data_config.get("outcome", _OUTCOME))
    target_covariates = tuple(data_config.get("target_covariates", _TARGET_COVARIATES))
    if _AGE_COEFFICIENT_NAME not in target_covariates:
        raise ValueError("target_covariates must include AGEP")

    raw_file = data_config.get("raw_file")
    if raw_file is None:
        raise ValueError("config data.raw_file is required")
    raw_path = _resolve_path(raw_file, config_dict.get("_config_dir"))

    smoke_cap = preprocessing.get("smoke_sample_size_cap", data_config.get("smoke_sample_size_cap"))
    nrows = data_config.get("nrows")
    if smoke and smoke_cap is not None:
        nrows = int(smoke_cap)
    elif nrows is not None:
        nrows = int(nrows)

    read_columns = _read_columns(config_dict, outcome, target_covariates)
    frame = load_acs_pums(raw_path, columns=read_columns, nrows=nrows)
    missing = [column for column in (outcome, *target_covariates) if column not in frame.columns]
    if missing:
        raise ValueError(f"ACS file is missing required columns: {missing}")

    cleaned = frame.copy()
    for column in (outcome, *target_covariates):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    finite_mask = np.ones(len(cleaned), dtype=bool)
    for column in (outcome, *target_covariates):
        finite_mask &= np.isfinite(cleaned[column].to_numpy(dtype=float))

    adult_age_min = _adult_age_min(preprocessing)
    if adult_age_min is not None:
        finite_mask &= cleaned[_AGE_COEFFICIENT_NAME].to_numpy(dtype=float) >= adult_age_min

    cleaned = cleaned.loc[finite_mask].reset_index(drop=True)
    if smoke and smoke_cap is not None and len(cleaned) > int(smoke_cap):
        cleaned = cleaned.iloc[: int(smoke_cap)].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("ACS preprocessing removed all rows")

    scale_description = _maybe_scale_income(cleaned, outcome, preprocessing, data_config)
    cleaned.attrs["acs_outcome"] = outcome
    cleaned.attrs["acs_target_covariates"] = target_covariates
    cleaned.attrs["acs_adult_age_min"] = adult_age_min
    cleaned.attrs["acs_income_scale"] = scale_description
    cleaned.attrs["acs_assumptions"] = tuple(config_dict.get("assumptions", ()))
    return cleaned


def acs_truth_linear_coef(df: pd.DataFrame) -> float:
    """Return the full-data least-squares coefficient on ``AGEP``."""

    outcome = str(df.attrs.get("acs_outcome", _OUTCOME))
    design, columns = _target_design_matrix(df)
    y = _outcome_array(df, outcome)
    beta = _solve_normal_equations(design.T @ design, design.T @ y, "target design matrix")
    return float(beta[_coefficient_index(columns, _AGE_COEFFICIENT_NAME)])


def run_acs_experiment(
    config_path: str | Path | Mapping[str, Any] = "configs/acs.yaml",
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the ACS income regression Monte Carlo experiment and save CSV outputs."""

    config = _coerce_config(config_path)
    df = prepare_acs_dataset(config, smoke=smoke)
    outcome = str(df.attrs.get("acs_outcome", _OUTCOME))
    design, design_columns = _target_design_matrix(df)
    coefficient_index = _coefficient_index(design_columns, _AGE_COEFFICIENT_NAME)
    y_true = _outcome_array(df, outcome)
    truth = float(_linear_coef(design, y_true)[coefficient_index])

    feature_matrix, feature_columns, feature_policy = _prediction_feature_matrix(df, config, outcome)
    seeds = _experiment_seeds(config, smoke)
    prediction_seeds = _prediction_seeds(config, seeds, smoke)
    model = _prediction_model(config, smoke)
    prediction_data = crossfit_regression_predictions(
        feature_matrix,
        y_true,
        seeds=prediction_seeds,
        model=model,
    )
    f = prediction_data.mean_prediction
    tau = prediction_data.tau

    budgets = _budget_grid(config, smoke)
    confidence_level = float(config.get("confidence_level", 0.90))
    z_value = _z_value(confidence_level)
    baseline_effort = float(config.get("baseline_effort", 0.8))
    if not np.isfinite(baseline_effort) or baseline_effort <= 0.0 or baseline_effort > 1.0:
        raise ValueError("baseline_effort must lie in (0, 1]")

    trial_results: list[TrialResult] = []
    base_metadata = _base_metadata(
        config,
        df,
        smoke=smoke,
        outcome=outcome,
        feature_columns=feature_columns,
        feature_policy=feature_policy,
        prediction_seeds=prediction_seeds,
    )
    for budget in budgets:
        designs = _method_designs(tau, float(budget), baseline_effort, config)
        for seed in seeds:
            rng = make_rng(seed)
            for method in _METHODS:
                design_spec = designs[method]
                interval = _simulate_and_estimate(
                    method=method,
                    X=design,
                    y_true=y_true,
                    f=f,
                    pi=design_spec["pi"],
                    effort=design_spec["effort"],
                    rho=design_spec["rho"],
                    rng=rng,
                    coefficient_index=coefficient_index,
                    z_value=z_value,
                    config=config,
                )
                metadata = dict(base_metadata)
                metadata.update(
                    {
                        "rho": float(design_spec["rho"]),
                        "effort": float(design_spec["effort"]),
                        "q_effort": float(design_spec["effort"]),
                        "bonus": float(design_spec.get("bonus", 0.0)),
                        "baseline_mix_tau": design_spec.get("baseline_mix_tau", np.nan),
                        "expected_budget": float(design_spec["expected_budget"]),
                        "expected_queries": float(np.sum(design_spec["pi"])),
                        "budget_edge_case": str(design_spec.get("budget_edge_case", "none")),
                        "coefficient": _AGE_COEFFICIENT_NAME,
                    }
                )
                trial_results.append(
                    TrialResult(
                        method=method,
                        budget=float(budget),
                        estimate=interval["estimate"],
                        lower=interval["lower"],
                        upper=interval["upper"],
                        seed=int(seed),
                        metadata=metadata,
                    )
                )

    trials = _trial_results_frame(trial_results, truth)
    summary = summarize_intervals(trial_results, truth=truth)

    output_dir = _output_tables_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "acs_trials.csv"
    summary_path = output_dir / "acs_summary.csv"
    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)

    return {
        "trials": trials,
        "summary": summary,
        "truth": truth,
        "trials_path": trials_path,
        "summary_path": summary_path,
    }


def _simulate_and_estimate(
    *,
    method: str,
    X: np.ndarray,
    y_true: np.ndarray,
    f: np.ndarray,
    pi: np.ndarray,
    effort: float,
    rho: float,
    rng: np.random.Generator,
    coefficient_index: int,
    z_value: float,
    config: Mapping[str, Any],
) -> dict[str, float]:
    xi = rng.random(y_true.size) < pi
    sentinel = np.zeros(y_true.size, dtype=bool)
    zeta = np.asarray(xi, dtype=bool)
    if method == "robust":
        sampled = np.flatnonzero(xi)
        if sampled.size:
            sentinel[sampled] = rng.random(sampled.size) < rho
        zeta = xi & ~sentinel

    labels = simulate_human_labels(
        y_true,
        f,
        effort=effort,
        rng=rng,
        sentinel=sentinel,
        continuous_sentinel_offset=_continuous_sentinel_offset(y_true, config),
    )
    y_reported = labels["reported"].astype(float, copy=False)
    q_effort = np.full(y_true.size, effort, dtype=float)

    if method == "classical":
        beta, covariance = _classical_linear_estimator(X, y_reported, f, xi, pi, q_effort)
    else:
        gamma = _correction_factor(xi, pi, q_effort, zeta=zeta, rho=rho if method == "robust" else 0.0)
        beta, covariance = _model_assisted_linear_estimator(X, y_reported, f, gamma)

    estimate = float(beta[coefficient_index])
    variance = max(float(covariance[coefficient_index, coefficient_index]), 0.0)
    se = float(np.sqrt(variance / y_true.size))
    return {
        "estimate": estimate,
        "lower": estimate - z_value * se,
        "upper": estimate + z_value * se,
    }


def _model_assisted_linear_estimator(
    X: np.ndarray,
    y_reported: np.ndarray,
    f: np.ndarray,
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pseudo_y = f + gamma * (y_reported - f)
    beta = _linear_coef(X, pseudo_y)
    score_residual = X @ beta - pseudo_y
    hessian = (X.T @ X) / X.shape[0]
    scores = X * score_residual[:, None]
    meat = (scores.T @ scores) / X.shape[0]
    covariance = _sandwich(hessian, meat, "model-assisted ACS estimator")
    return beta, covariance


def _classical_linear_estimator(
    X: np.ndarray,
    y_reported: np.ndarray,
    f: np.ndarray,
    xi: np.ndarray,
    pi: np.ndarray,
    q_effort: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    label_correction = np.divide(
        y_reported - f,
        q_effort,
        out=np.zeros_like(y_reported, dtype=float),
        where=q_effort > 0.0,
    )
    corrected_y = f + label_correction
    weights = _sampling_weights(xi, pi)
    weighted_X = X * weights[:, None]
    hessian = (X.T @ weighted_X) / X.shape[0]
    beta = _solve_normal_equations(X.T @ weighted_X, X.T @ (weights * corrected_y), "classical sampled design")
    score_residual = X @ beta - corrected_y
    scores = X * (weights * score_residual)[:, None]
    meat = (scores.T @ scores) / X.shape[0]
    covariance = _sandwich(hessian, meat, "classical ACS estimator")
    return beta, covariance


def _method_designs(
    tau: np.ndarray,
    budget: float,
    baseline_effort: float,
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    n = tau.size
    incentive = _mapping(config.get("incentive"))
    w0 = float(incentive.get("w0", 0.64))
    k = float(incentive.get("k", 1.0))
    baseline_cost = _baseline_per_sample_cost(w0, baseline_effort, incentive)
    uniform_pi = uniform_probabilities(n, budget=budget, per_sample_cost=baseline_cost)
    active_pi, active_mix_tau = _active_baseline_probabilities(
        tau, budget, baseline_effort, baseline_cost, config
    )
    robust_design = _robust_probabilities(tau, budget, config)
    uniform_expected_budget = float(baseline_cost * np.sum(uniform_pi))
    active_expected_budget = float(baseline_cost * np.sum(active_pi))
    return {
        "classical": {
            "pi": uniform_pi,
            "effort": baseline_effort,
            "rho": 0.0,
            "bonus": 0.0,
            "baseline_mix_tau": 1.0,
            "expected_budget": uniform_expected_budget,
            "budget_edge_case": _budget_edge_case(uniform_pi, uniform_expected_budget, budget),
        },
        "uniform": {
            "pi": uniform_pi,
            "effort": baseline_effort,
            "rho": 0.0,
            "bonus": 0.0,
            "baseline_mix_tau": 1.0,
            "expected_budget": uniform_expected_budget,
            "budget_edge_case": _budget_edge_case(uniform_pi, uniform_expected_budget, budget),
        },
        "active": {
            "pi": active_pi,
            "effort": baseline_effort,
            "rho": 0.0,
            "bonus": 0.0,
            "baseline_mix_tau": active_mix_tau,
            "expected_budget": active_expected_budget,
            "budget_edge_case": _budget_edge_case(active_pi, active_expected_budget, budget),
        },
        "robust": robust_design,
    }


def _active_baseline_probabilities(
    tau: np.ndarray,
    budget: float,
    q_effort: float,
    baseline_cost: float,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    sampling = _mapping(config.get("sampling"))
    grid = sampling.get("baseline_tau_grid")
    if grid is None:
        grid = [0.0]
    candidates: list[tuple[float, float, np.ndarray]] = []
    for mix_tau in grid:
        mix = float(mix_tau)
        pi = mixed_active_uniform_probabilities(
            tau, mix, budget=budget, per_sample_cost=baseline_cost
        )
        objective = _sampling_objective(tau, pi, q_effort, rho=0.0)
        candidates.append((objective, mix, pi))
    _, best_mix, best_pi = min(candidates, key=lambda item: (item[0], item[1]))
    return best_pi, best_mix


def _baseline_per_sample_cost(
    w0: float,
    baseline_effort: float,
    incentive_config: Mapping[str, Any],
) -> float:
    model = str(incentive_config.get("baseline_payment_model", "effort_squared")).lower()
    effort = float(baseline_effort)
    if not np.isfinite(effort) or effort <= 0.0 or effort > 1.0:
        raise ValueError("baseline_effort must lie in (0, 1]")
    if model in {"none", "w0_only"}:
        return float(w0)
    if model in {"effort_squared", "sentinel_equivalent"}:
        return float(w0) + effort**2
    raise ValueError(f"unknown ACS baseline_payment_model {model!r}")


def _robust_probabilities(
    tau: np.ndarray,
    budget: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    incentive = _mapping(config.get("incentive"))
    w0 = float(incentive.get("w0", 0.64))
    k = float(incentive.get("k", 1.0))
    rho_grid = incentive.get("rho_grid", [0.1])
    candidates: list[tuple[float, float, float, float, float, np.ndarray]] = []
    for raw_rho in rho_grid:
        rho = float(raw_rho)
        if not np.isfinite(rho) or rho <= 0.0 or rho >= 1.0:
            continue
        bonus = optimized_bonus(w0, rho)
        effort = float(effort_from_bonus(rho, bonus))
        if effort <= 0.0:
            continue
        pi = risk_neutral_fixed_rho_design(tau, budget=budget, rho=rho, w0=w0, k=k)
        expected_budget = theory_budget(pi, rho, bonus, effort, w0, k, q_values=effort)
        objective = _sampling_objective(tau, pi, effort, rho=rho)
        candidates.append((objective, rho, bonus, effort, expected_budget, pi))
    if not candidates:
        raise ValueError("incentive.rho_grid must contain at least one rho in (0, 1)")
    _, rho, bonus, effort, expected_budget, pi = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return {
        "pi": pi,
        "effort": effort,
        "rho": rho,
        "bonus": bonus,
        "baseline_mix_tau": np.nan,
        "expected_budget": expected_budget,
        "budget_edge_case": _budget_edge_case(pi, expected_budget, budget, fixed_cost=rho * k),
    }


def _budget_edge_case(
    pi: np.ndarray,
    expected_budget: float,
    budget: float,
    *,
    fixed_cost: float = 0.0,
) -> str:
    tolerance = 1e-9
    if expected_budget > budget + tolerance:
        if fixed_cost > budget + tolerance:
            return "fixed_cost_exceeds_budget"
        return "rounding_budget_excess"
    if expected_budget < budget - tolerance and np.all(np.asarray(pi, dtype=float) >= 1.0 - tolerance):
        return "probability_capped_under_budget"
    return "none"


def _sampling_objective(tau: np.ndarray, pi: np.ndarray, q_effort: float, rho: float) -> float:
    positive_tau = tau > 0.0
    if np.any(positive_tau & (pi <= 0.0)):
        return float("inf")
    denominator = (1.0 - rho) * pi * q_effort
    valid = positive_tau & (denominator > 0.0)
    if not np.any(valid):
        return float("inf")
    return float(np.sum(tau[valid] / denominator[valid]))


def _correction_factor(
    xi: np.ndarray,
    pi: np.ndarray,
    q_effort: np.ndarray,
    *,
    zeta: np.ndarray | None = None,
    rho: float = 0.0,
) -> np.ndarray:
    xi_array = np.asarray(xi, dtype=float)
    if zeta is None:
        regular = np.ones_like(xi_array)
        regular_probability = 1.0
    else:
        regular = np.asarray(zeta, dtype=float)
        regular_probability = 1.0 - float(rho)
    denominator = regular_probability * pi * q_effort
    numerator = xi_array * regular
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(pi, dtype=float),
        where=denominator > 0.0,
    )


def _sampling_weights(xi: np.ndarray, pi: np.ndarray) -> np.ndarray:
    return np.divide(
        np.asarray(xi, dtype=float),
        pi,
        out=np.zeros_like(pi, dtype=float),
        where=pi > 0.0,
    )


def _target_design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    required = [_OUTCOME, *_TARGET_COVARIATES]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"ACS data is missing required columns: {missing}")

    age = pd.to_numeric(df[_AGE_COEFFICIENT_NAME], errors="coerce").to_numpy(dtype=float)
    sex = pd.to_numeric(df["SEX"], errors="coerce")
    sex_dummies = pd.get_dummies(sex.astype("Int64").astype("category"), prefix="SEX", drop_first=True)
    columns = ["intercept", _AGE_COEFFICIENT_NAME, *sex_dummies.columns.astype(str).tolist()]
    pieces = [np.ones(age.size, dtype=float), age]
    for column in sex_dummies.columns:
        pieces.append(sex_dummies[column].to_numpy(dtype=float))
    design = np.column_stack(pieces)
    _validate_design_matrix(design, "target design matrix")
    return design, columns


def _prediction_feature_matrix(
    df: pd.DataFrame,
    config: Mapping[str, Any],
    outcome: str,
) -> tuple[np.ndarray, list[str], str]:
    explicit_features = _explicit_feature_columns(config)
    if explicit_features is None:
        candidate_columns = [column for column in df.columns if column != outcome]
        policy = "all_numeric_non_outcome_columns_with_median_imputation"
    else:
        candidate_columns = list(explicit_features)
        policy = "explicit_config_features_with_median_imputation"

    feature_data: dict[str, np.ndarray] = {}
    for column in candidate_columns:
        if column == outcome or column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            feature_data[str(column)] = numeric.to_numpy(dtype=float)
            continue
        if explicit_features is not None:
            dummies = pd.get_dummies(df[column].astype("category"), prefix=str(column), drop_first=False)
            for dummy_column in dummies.columns:
                feature_data[str(dummy_column)] = dummies[dummy_column].to_numpy(dtype=float)

    if not feature_data:
        feature_data[_AGE_COEFFICIENT_NAME] = pd.to_numeric(
            df[_AGE_COEFFICIENT_NAME], errors="coerce"
        ).to_numpy(dtype=float)
        policy = "fallback_age_only"

    imputed_data: dict[str, np.ndarray] = {}
    for column, raw_values in feature_data.items():
        values = np.asarray(raw_values, dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        fill_value = float(np.median(values[finite]))
        imputed_data[column] = np.where(finite, values, fill_value)

    if not imputed_data:
        raise ValueError("prediction feature construction produced no usable columns")
    feature_frame = pd.DataFrame(imputed_data, index=df.index)
    matrix = feature_frame.to_numpy(dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("prediction feature matrix contains non-finite values after imputation")
    return matrix, feature_frame.columns.astype(str).tolist(), policy


def _prediction_model(config: Mapping[str, Any], smoke: bool) -> Any:
    prediction = _mapping(config.get("prediction_model"))
    family = str(prediction.get("family", "sklearn")).lower()
    if family in {"linear", "linear_regression", "ols"}:
        return LinearRegression()
    if family in {"dummy", "mean", "constant"}:
        return DummyRegressor(strategy="mean")
    if family in {"random_forest", "random_forest_regressor", "rf"}:
        n_estimators = int(prediction.get("n_estimators", 12 if smoke else 80))
        max_depth = prediction.get("max_depth", 12 if smoke else None)
        min_samples_leaf = int(prediction.get("min_samples_leaf", 5))
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=1,
            random_state=0,
        )
    if family in {"gradient_boosted_trees", "hist_gradient_boosting", "hgb", "sklearn"}:
        return "sklearn"
    if family in {"auto", "xgboost", "xgb"}:
        return "auto" if family == "auto" else "xgboost"
    raise ValueError(f"unknown ACS prediction model family {family!r}")


def _linear_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _solve_normal_equations(X.T @ X, X.T @ y, "linear regression design")


def _solve_normal_equations(matrix: np.ndarray, rhs: np.ndarray, context: str) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{context} normal matrix must be square")
    if np.linalg.matrix_rank(matrix, tol=_Z_TOL) < matrix.shape[0]:
        raise ValueError(f"{context} is singular")
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{context} is singular") from exc


def _sandwich(hessian: np.ndarray, meat: np.ndarray, context: str) -> np.ndarray:
    if np.linalg.matrix_rank(hessian, tol=_Z_TOL) < hessian.shape[0]:
        raise ValueError(f"{context} Hessian is singular")
    try:
        hessian_inv = np.linalg.solve(hessian, np.eye(hessian.shape[0]))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{context} Hessian is singular") from exc
    return hessian_inv @ meat @ hessian_inv


def _validate_design_matrix(X: np.ndarray, context: str) -> None:
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{context} must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{context} must contain only finite values")
    if X.shape[0] < X.shape[1] or np.linalg.matrix_rank(X, tol=_Z_TOL) < X.shape[1]:
        raise ValueError(f"{context} is singular")


def _outcome_array(df: pd.DataFrame, outcome: str) -> np.ndarray:
    values = pd.to_numeric(df[outcome], errors="coerce").to_numpy(dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{outcome} must be a non-empty finite numeric vector")
    return values


def _coefficient_index(columns: Sequence[str], coefficient: str) -> int:
    try:
        return list(columns).index(coefficient)
    except ValueError as exc:
        raise ValueError(f"coefficient {coefficient!r} is not in the design matrix") from exc


def _trial_results_frame(results: Sequence[TrialResult], truth: float) -> pd.DataFrame:
    records = []
    for result in results:
        metadata = dict(result.metadata)
        lower = float(result.lower)
        upper = float(result.upper)
        row = {
            "method": result.method,
            "budget": float(result.budget),
            "estimate": float(result.estimate),
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "covered": bool(lower <= truth <= upper),
            "seed": result.seed,
        }
        row.update(metadata)
        records.append(row)
    return pd.DataFrame.from_records(records)


def _base_metadata(
    config: Mapping[str, Any],
    df: pd.DataFrame,
    *,
    smoke: bool,
    outcome: str,
    feature_columns: Sequence[str],
    feature_policy: str,
    prediction_seeds: Sequence[int],
) -> dict[str, Any]:
    prediction = _mapping(config.get("prediction_model"))
    return {
        "experiment": str(config.get("experiment", "acs_income_regression")),
        "budget_model": "main_text",
        "k_cost_interpretation": "fixed_global_sentinel_cost",
        "estimand": "linear_regression_coefficient",
        "outcome": outcome,
        "target_covariates": "|".join(df.attrs.get("acs_target_covariates", _TARGET_COVARIATES)),
        "n_rows": int(len(df)),
        "confidence_level": float(config.get("confidence_level", 0.90)),
        "adult_age_min": df.attrs.get("acs_adult_age_min"),
        "income_scale": df.attrs.get("acs_income_scale", "none"),
        "smoke": bool(smoke),
        "prediction_model_family": str(prediction.get("family", "sklearn")),
        "prediction_feature_policy": feature_policy,
        "n_prediction_features": int(len(feature_columns)),
        "prediction_seeds": "|".join(str(seed) for seed in prediction_seeds),
        "assumptions": " | ".join(str(item) for item in config.get("assumptions", ())),
    }


def _coerce_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    config_path = Path(config)
    loaded = load_yaml_config(config_path)
    loaded["_config_dir"] = config_path.resolve().parent
    return loaded


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("expected a mapping in ACS config")
    return dict(value)


def _resolve_path(path: Any, base_dir: Any = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    if base_dir is not None:
        relative = Path(base_dir) / candidate
        if relative.exists():
            return relative
    return candidate


def _read_columns(
    config: Mapping[str, Any],
    outcome: str,
    target_covariates: Sequence[str],
) -> list[str] | None:
    data_config = _mapping(config.get("data"))
    explicit_columns = data_config.get("columns")
    explicit_features = _explicit_feature_columns(config)
    if explicit_columns is None and explicit_features is None:
        return None
    columns: list[str] = [outcome, *target_covariates]
    if explicit_columns is not None:
        columns.extend(str(column) for column in explicit_columns)
    if explicit_features is not None:
        columns.extend(explicit_features)
    return list(dict.fromkeys(columns))


def _explicit_feature_columns(config: Mapping[str, Any]) -> list[str] | None:
    prediction = _mapping(config.get("prediction_model"))
    data_config = _mapping(config.get("data"))
    for key in ("features", "feature_columns", "covariates"):
        if key in prediction:
            return [str(column) for column in prediction[key]]
        if key in data_config:
            return [str(column) for column in data_config[key]]
    return None


def _adult_age_min(preprocessing: Mapping[str, Any]) -> float | None:
    if preprocessing.get("adult_records", True) is False:
        return None
    if "adult_age_min" in preprocessing:
        value = preprocessing["adult_age_min"]
    elif "min_age" in preprocessing:
        value = preprocessing["min_age"]
    else:
        value = 18.0
    if value is None:
        return None
    age_min = float(value)
    if not np.isfinite(age_min):
        raise ValueError("adult age minimum must be finite")
    return age_min


def _maybe_scale_income(
    df: pd.DataFrame,
    outcome: str,
    preprocessing: Mapping[str, Any],
    data_config: Mapping[str, Any],
) -> str:
    if preprocessing.get("standardize_income", False):
        values = df[outcome].to_numpy(dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        if std <= 0.0 or not np.isfinite(std):
            raise ValueError("cannot standardize income with zero or non-finite standard deviation")
        df[outcome] = (values - mean) / std
        return f"standardized_mean_{mean:.12g}_sd_{std:.12g}"

    scale_keys = ("income_scale", "outcome_scale", "scale_income")
    for key in scale_keys:
        if key in preprocessing:
            scale = preprocessing[key]
            break
        if key in data_config:
            scale = data_config[key]
            break
    else:
        return "none"

    if isinstance(scale, bool):
        if not scale:
            return "none"
        raise ValueError("boolean scale_income=true is ambiguous; provide a numeric scale")
    scale_value = float(scale)
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError("income scale must be a finite positive number")
    df[outcome] = df[outcome].to_numpy(dtype=float) / scale_value
    return f"divided_by_{scale_value:.12g}"


def _experiment_seeds(config: Mapping[str, Any], smoke: bool) -> tuple[int, ...]:
    key = "smoke_seeds" if smoke and "smoke_seeds" in config else "seeds"
    seeds = tuple(int(seed) for seed in config.get(key, [0]))
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    return seeds


def _prediction_seeds(
    config: Mapping[str, Any],
    experiment_seeds: Sequence[int],
    smoke: bool,
) -> tuple[int, ...]:
    prediction = _mapping(config.get("prediction_model"))
    raw = prediction.get("crossfit_seeds", prediction.get("seeds"))
    if raw is not None:
        seeds = tuple(int(seed) for seed in raw)
    elif smoke:
        seeds = (int(experiment_seeds[0]),)
    else:
        seeds = tuple(int(seed) for seed in experiment_seeds[: min(5, len(experiment_seeds))])
    if not seeds:
        raise ValueError("prediction seeds must contain at least one seed")
    return seeds


def _budget_grid(config: Mapping[str, Any], smoke: bool) -> tuple[float, ...]:
    grid_config = _mapping(config.get("budget_grid"))
    key = "smoke" if smoke and "smoke" in grid_config else "full"
    budgets = grid_config.get(key, grid_config.get("budgets", [100.0]))
    budget_tuple = tuple(float(budget) for budget in budgets)
    if not budget_tuple or any((not np.isfinite(budget) or budget < 0.0) for budget in budget_tuple):
        raise ValueError("budget grid must contain finite nonnegative values")
    return budget_tuple


def _continuous_sentinel_offset(y_true: np.ndarray, config: Mapping[str, Any]) -> float:
    simulation = _mapping(config.get("simulation"))
    if "continuous_sentinel_offset" in simulation:
        offset = float(simulation["continuous_sentinel_offset"])
    else:
        scale = float(np.std(y_true))
        offset = scale if np.isfinite(scale) and scale > 0.0 else 1.0
    if not np.isfinite(offset) or offset == 0.0:
        raise ValueError("continuous_sentinel_offset must be finite and nonzero")
    return offset


def _z_value(confidence_level: float) -> float:
    if not np.isfinite(confidence_level) or confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    alpha = 1.0 - confidence_level
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def _output_tables_dir(config: Mapping[str, Any]) -> Path:
    outputs = _mapping(config.get("outputs"))
    return Path(outputs.get("tables_dir", "outputs/tables"))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ACS PUMS income regression experiment.")
    parser.add_argument("--config", default="configs/acs.yaml", help="Path to the ACS YAML config.")
    parser.add_argument("--smoke", action="store_true", help="Run the smoke budget/seed/data subset.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_acs_experiment(config_path=args.config, smoke=args.smoke)
    print(f"wrote {result['trials_path']}")
    print(f"wrote {result['summary_path']}")


if __name__ == "__main__":
    main()
