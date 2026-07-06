"""Pew Wave 79 mean-estimation experiments.

This module implements the Biden and Trump post-election message experiments
using the main-text budget model

``sum_i ((rho * b_i * q(e_i) + w0) * pi_i) + rho * k <= B``

with the ``rho * k`` term treated as a single global sentinel construction cost.
Baseline methods use the same fixed overhead ``w0`` per queried item and no
sentinel term. Experiment-specific engineering choices are documented in output
metadata.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any
import zlib

import numpy as np
import pandas as pd

from experiments.common import (
    crossfit_binary_predictions,
    load_yaml_config,
)
from incentive.design import (
    effort_from_bonus,
    optimized_bonus,
    risk_neutral_fixed_rho_design,
    theory_budget,
)
from incentive.simulation import simulate_human_labels
from inference.sampling import expected_budget as sampling_expected_budget
from inference.sampling import scale_probabilities_to_budget


_BUDGET_MODEL = "main_text"
_K_COST_INTERPRETATION = "fixed_global_sentinel_cost"
_MISSING_CATEGORY = "__MISSING__"
_OUTPUT_DIR = Path("outputs") / "tables"
_METHODS = ("classical", "uniform", "active", "robust")


@dataclass(frozen=True)
class PewPreparedData:
    """Clean Pew outcome, survey weights, and encoded covariates."""

    X: pd.DataFrame
    y: np.ndarray
    weights: np.ndarray
    feature_names: tuple[str, ...]
    frame: pd.DataFrame
    outcome: str
    outcome_column: str

    @property
    def n(self) -> int:
        return int(self.y.size)


@dataclass(frozen=True)
class _ActiveDesign:
    pi: np.ndarray
    mix: float
    objective: float


@dataclass(frozen=True)
class _RobustDesign:
    pi: np.ndarray
    rho: float
    bonus: float
    effort: float
    expected_budget: float
    objective: float


def load_pew_wave79(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load the Pew ATP Wave 79 SPSS file.

    Pew distributes these data as ``.sav``. Pandas delegates SPSS loading to the
    optional ``pyreadstat`` dependency, so this function raises a clearer error
    when that dependency is unavailable.
    """

    kwargs: dict[str, Any] = {"convert_categoricals": False}
    if columns is not None:
        kwargs["usecols"] = list(columns)
    try:
        return pd.read_spss(Path(path), **kwargs)
    except ImportError as exc:
        raise ImportError(
            "Loading Pew Wave 79 SPSS data requires the optional `pyreadstat` "
            "dependency. Install the Pew extras or install pyreadstat before "
            "running the full experiment."
        ) from exc


def recode_message_outcome(series: pd.Series) -> pd.Series:
    """Recode Pew message responses to binary approval.

    Numeric codes ``1`` and ``2`` map to ``1``; codes ``3`` and ``4`` map to
    ``0``. Refused, missing, skipped, and code ``99`` are returned as missing.
    If categorical labels are already present, labels containing
    ``RIGHT message`` map to ``1`` and labels containing ``WRONG message`` map to
    ``0``.
    """

    values = pd.Series(series, copy=False)
    recoded = pd.Series(np.nan, index=values.index, dtype=float, name=values.name)

    numeric = pd.to_numeric(values, errors="coerce")
    recoded.loc[numeric.isin([1, 2])] = 1.0
    recoded.loc[numeric.isin([3, 4])] = 0.0

    labels = values.astype("string")
    right_message = labels.str.contains(r"RIGHT\s+message", case=False, na=False, regex=True)
    wrong_message = labels.str.contains(r"WRONG\s+message", case=False, na=False, regex=True)
    recoded.loc[right_message] = 1.0
    recoded.loc[wrong_message] = 0.0
    recoded.loc[numeric.eq(99)] = np.nan
    return recoded


def prepare_pew_dataset(config: Mapping[str, Any] | str | Path, outcome: str) -> PewPreparedData:
    """Return cleaned outcome, weights, and one-hot encoded covariates."""

    config_map = _coerce_config(config)
    data_config = _require_mapping(config_map, "data")
    outcome_key = str(outcome).lower()
    outcome_config = _require_mapping(_require_mapping(data_config, "outcomes"), outcome_key)
    aliases = [str(column) for column in outcome_config.get("column_aliases", [])]
    if not aliases:
        raise ValueError(f"outcome {outcome_key!r} must define column_aliases")

    covariates = [str(column) for column in data_config.get("covariates", [])]
    if not covariates:
        raise ValueError("Pew config must define at least one covariate")

    raw_file = data_config.get("raw_file")
    if raw_file is None:
        raise ValueError("Pew config is missing data.raw_file")
    frame = load_pew_wave79(raw_file)
    outcome_column = _first_existing_column(frame, aliases, f"outcome {outcome_key!r}")
    _require_columns(frame, covariates, "Pew covariates")

    outcome_values = recode_message_outcome(frame[outcome_column])
    weight_column = data_config.get("weight_column")
    weights = _extract_weights(frame, weight_column)

    clean = frame.loc[:, covariates].copy()
    clean["_outcome"] = outcome_values
    clean["_weight"] = weights
    valid = clean["_outcome"].notna() & np.isfinite(clean["_weight"]) & (clean["_weight"] > 0.0)
    clean = clean.loc[valid].reset_index(drop=True)
    if clean.empty:
        raise ValueError(f"outcome {outcome_key!r} has no complete non-missing rows")

    y = clean["_outcome"].to_numpy(dtype=float)
    sample_weights = clean["_weight"].to_numpy(dtype=float)
    X = _encode_covariates(clean, covariates, str(weight_column) if weight_column else None)
    if X.shape[0] != y.size:
        raise RuntimeError("encoded covariates and outcome length do not match")

    return PewPreparedData(
        X=X,
        y=y,
        weights=sample_weights,
        feature_names=tuple(str(column) for column in X.columns),
        frame=clean.drop(columns=["_outcome", "_weight"]),
        outcome=outcome_key,
        outcome_column=outcome_column,
    )


def run_pew_mean_experiment(
    config_path: str | Path = "configs/pew.yaml",
    outcome: str = "biden",
    smoke: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the Pew mean-estimation experiment and save trial/summary CSVs."""

    config = load_yaml_config(config_path)
    prepared = prepare_pew_dataset(config, outcome)
    seeds = _select_seeds(config, smoke)
    budgets = _select_budgets(config, smoke)
    confidence_level = float(config.get("confidence_level", 0.90))
    baseline_effort = float(config.get("baseline_effort", 0.8))
    incentive_config = _require_mapping(config, "incentive")
    sampling_config = _require_mapping(config, "sampling")

    model = _prediction_model(config)
    prediction_seeds = _select_prediction_seeds(config, smoke, seeds)
    predictions = crossfit_binary_predictions(
        prepared.X.to_numpy(dtype=float),
        prepared.y,
        prediction_seeds,
        model=model,
    )
    f = np.clip(predictions.mean_prediction, 0.0, 1.0)
    tau = np.asarray(predictions.tau, dtype=float)
    truth = _weighted_mean(prepared.y, prepared.weights)
    norm_weights = _normalized_weights(prepared.weights)

    trial_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for budget in budgets:
            trial_rows.extend(
                _run_budget_trials(
                    y_true=prepared.y,
                    f=f,
                    tau=tau,
                    norm_weights=norm_weights,
                    outcome=prepared.outcome,
                    budget=float(budget),
                    seed=int(seed),
                    confidence_level=confidence_level,
                    baseline_effort=baseline_effort,
                    incentive_config=incentive_config,
                    sampling_config=sampling_config,
                    truth=truth,
                    n=prepared.n,
                )
            )

    trials = pd.DataFrame.from_records(trial_rows)
    trials = trials[_trial_columns()]
    summary = _summarize_trials(trials)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trial_path = _OUTPUT_DIR / f"pew_{prepared.outcome}_trials.csv"
    summary_path = _OUTPUT_DIR / f"pew_{prepared.outcome}_summary.csv"
    trials.to_csv(trial_path, index=False)
    summary.to_csv(summary_path, index=False)
    return trials, summary


def _run_budget_trials(
    *,
    y_true: np.ndarray,
    f: np.ndarray,
    tau: np.ndarray,
    norm_weights: np.ndarray,
    outcome: str,
    budget: float,
    seed: int,
    confidence_level: float,
    baseline_effort: float,
    incentive_config: Mapping[str, Any],
    sampling_config: Mapping[str, Any],
    truth: float,
    n: int,
) -> list[dict[str, Any]]:
    w0 = float(incentive_config.get("w0", 0.64))
    k = float(incentive_config.get("k", 1.0))
    baseline_tau_grid = [float(value) for value in sampling_config.get("baseline_tau_grid", [0.0])]
    baseline_cost = _baseline_accuracy_cost(y_true, f, baseline_effort, w0, incentive_config)

    uniform_pi = _uniform_probabilities(n, budget, per_item_cost=baseline_cost)
    active_design = _select_active_design(
        tau=tau,
        budget=budget,
        baseline_cost=baseline_cost,
        baseline_tau_grid=baseline_tau_grid,
        norm_weights=norm_weights,
        q_effort=baseline_effort,
    )
    robust_design = _select_robust_design(
        tau=tau,
        budget=budget,
        rho_grid=incentive_config.get("rho_grid", [0.2]),
        w0=w0,
        k=k,
        norm_weights=norm_weights,
    )

    designs = {
        "classical": {
            "pi": uniform_pi,
            "rho": 0.0,
            "bonus": 0.0,
            "effort": baseline_effort,
            "expected_budget": sampling_expected_budget(uniform_pi, baseline_cost),
            "mix": 1.0,
            "objective": np.nan,
        },
        "uniform": {
            "pi": uniform_pi,
            "rho": 0.0,
            "bonus": 0.0,
            "effort": baseline_effort,
            "expected_budget": sampling_expected_budget(uniform_pi, baseline_cost),
            "mix": 1.0,
            "objective": np.nan,
        },
        "active": {
            "pi": active_design.pi,
            "rho": 0.0,
            "bonus": 0.0,
            "effort": baseline_effort,
            "expected_budget": sampling_expected_budget(active_design.pi, baseline_cost),
            "mix": active_design.mix,
            "objective": active_design.objective,
        },
        "robust": {
            "pi": robust_design.pi,
            "rho": robust_design.rho,
            "bonus": robust_design.bonus,
            "effort": robust_design.effort,
            "expected_budget": robust_design.expected_budget,
            "mix": 0.0,
            "objective": robust_design.objective,
        },
    }

    rows = []
    for method in _METHODS:
        design = designs[method]
        method_seed = _method_seed(seed, budget, method)
        row = _evaluate_method(
            method=method,
            y_true=y_true,
            f=f,
            pi=np.asarray(design["pi"], dtype=float),
            rho=float(design["rho"]),
            bonus=float(design["bonus"]),
            effort=float(design["effort"]),
            expected_budget=float(design["expected_budget"]),
            mix=float(design["mix"]),
            objective=float(design["objective"]) if np.isfinite(design["objective"]) else np.nan,
            norm_weights=norm_weights,
            confidence_level=confidence_level,
            truth=truth,
            seed=seed,
            method_seed=method_seed,
            budget=budget,
            outcome=outcome,
            n=n,
            baseline_effort_config=baseline_effort,
        )
        rows.append(row)
    return rows


def _evaluate_method(
    *,
    method: str,
    y_true: np.ndarray,
    f: np.ndarray,
    pi: np.ndarray,
    rho: float,
    bonus: float,
    effort: float,
    expected_budget: float,
    mix: float,
    objective: float,
    norm_weights: np.ndarray,
    confidence_level: float,
    truth: float,
    seed: int,
    method_seed: int,
    budget: float,
    outcome: str,
    n: int,
    baseline_effort_config: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(method_seed)
    xi = (rng.random(y_true.size) < pi).astype(float)
    ai_labels = (rng.random(y_true.size) < f).astype(float)

    if method == "robust":
        sentinel = (xi == 1.0) & (rng.random(y_true.size) < rho)
        zeta = ((xi == 1.0) & ~sentinel).astype(float)
    else:
        sentinel = np.zeros(y_true.size, dtype=bool)
        zeta = xi.copy()

    simulated = simulate_human_labels(
        y_true,
        ai_labels,
        effort=effort,
        rng=rng,
        sentinel=sentinel,
    )
    y_reported = simulated["reported"].astype(float)
    classical_label_accuracy = (
        _final_label_accuracy_probability(y_true, f, effort) if method == "classical" else None
    )
    contributions = _mean_contributions(
        method,
        y_reported,
        f,
        xi,
        zeta,
        pi,
        rho,
        effort,
        classical_label_accuracy=classical_label_accuracy,
    )
    estimate, lower, upper = _weighted_interval(contributions, norm_weights, confidence_level)

    width = upper - lower
    return {
        "method": method,
        "budget": float(budget),
        "seed": int(seed),
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "width": width,
        "covered": bool(lower <= truth <= upper),
        "outcome": outcome,
        "truth": float(truth),
        "n": int(n),
        "confidence_level": float(confidence_level),
        "baseline_effort": float(baseline_effort_config),
        "rho": float(rho),
        "bonus": float(bonus),
        "effort": float(effort),
        "q_effort": float(effort),
        "baseline_tau_mix": float(mix),
        "design_objective": objective,
        "expected_budget": float(expected_budget),
        "expected_queries": float(pi.sum()),
        "query_probability_mean": float(np.mean(pi)),
        "sampled_count": int(xi.sum()),
        "sentinel_count": int(sentinel.sum()),
        "budget_model": _BUDGET_MODEL,
        "k_cost_interpretation": _K_COST_INTERPRETATION,
        "assumption_ai_label_generation": "bernoulli_crossfit_probability",
        "assumption_baseline_cost": "linear_accuracy_based_payment_for_e_0.8",
    }


def _mean_contributions(
    method: str,
    y_reported: np.ndarray,
    f: np.ndarray,
    xi: np.ndarray,
    zeta: np.ndarray,
    pi: np.ndarray,
    rho: float,
    q_effort: float,
    classical_label_accuracy: np.ndarray | None = None,
) -> np.ndarray:
    pi_safe = _positive_probabilities(pi)
    if method == "classical":
        label_accuracy = (
            np.full(y_reported.size, float(q_effort), dtype=float)
            if classical_label_accuracy is None
            else np.asarray(classical_label_accuracy, dtype=float)
        )
        if label_accuracy.shape != y_reported.shape:
            raise ValueError("classical_label_accuracy must match y_reported")
        if np.any(~np.isfinite(label_accuracy)) or np.any(label_accuracy <= 0.5) or np.any(label_accuracy > 1.0):
            raise ValueError("classical label accuracy must lie in (0.5, 1]")
        corrected = (y_reported + label_accuracy - 1.0) / (2.0 * label_accuracy - 1.0)
        return xi / pi_safe * corrected
    if method in {"uniform", "active"}:
        return f + (y_reported - f) * xi / (pi_safe * q_effort)
    if method == "robust":
        if not (0.0 < rho < 1.0):
            raise ValueError("robust estimator requires rho in (0, 1)")
        correction = (xi * zeta / (1.0 - rho)) / (pi_safe * q_effort)
        return f + (y_reported - f) * correction
    raise ValueError(f"unknown method {method!r}")


def _final_label_accuracy_probability(
    y_true: np.ndarray,
    f: np.ndarray,
    q_effort: float,
) -> np.ndarray:
    if not np.isfinite(q_effort) or q_effort <= 0.0 or q_effort > 1.0:
        raise ValueError("q_effort must lie in (0, 1]")
    y = np.asarray(y_true, dtype=float)
    probability = np.clip(np.asarray(f, dtype=float), 0.0, 1.0)
    if y.shape != probability.shape:
        raise ValueError("y_true and f must have the same shape")
    p_error = np.where(y >= 0.5, 1.0 - probability, probability)
    return 1.0 - p_error * (1.0 - float(q_effort))


def _select_active_design(
    *,
    tau: np.ndarray,
    budget: float,
    baseline_cost: np.ndarray | float | None = None,
    w0: float | None = None,
    baseline_tau_grid: Sequence[float],
    norm_weights: np.ndarray,
    q_effort: float,
) -> _ActiveDesign:
    if baseline_cost is None:
        if w0 is None:
            raise ValueError("baseline_cost or w0 must be provided")
        baseline_cost = float(w0)
    baseline_cost_array = np.broadcast_to(np.asarray(baseline_cost, dtype=float), np.asarray(tau).shape)
    best: _ActiveDesign | None = None
    for mix in baseline_tau_grid:
        mixed_tau = _mix_tau_with_uniform(tau, float(mix))
        weights = _cost_aware_active_weights(mixed_tau, baseline_cost_array)
        pi = scale_probabilities_to_budget(weights, budget, baseline_cost_array)
        objective = _estimated_design_variance(tau, pi, norm_weights, q_effort=q_effort, rho=0.0)
        candidate = _ActiveDesign(pi=pi, mix=float(mix), objective=objective)
        if best is None or candidate.objective < best.objective:
            best = candidate
    if best is None:
        raise ValueError("baseline_tau_grid must contain at least one value")
    return best


def _select_robust_design(
    *,
    tau: np.ndarray,
    budget: float,
    rho_grid: Sequence[float],
    w0: float,
    k: float,
    norm_weights: np.ndarray,
) -> _RobustDesign:
    best: _RobustDesign | None = None
    for rho_value in rho_grid:
        rho = float(rho_value)
        if not (0.0 < rho < 1.0):
            continue
        bonus = optimized_bonus(w0, rho)
        effort = float(effort_from_bonus(rho, bonus))
        pi = risk_neutral_fixed_rho_design(tau, budget, rho, w0, k)
        expected_budget = theory_budget(pi, rho, bonus, effort, w0, k, q_values=effort)
        objective = _estimated_design_variance(tau, pi, norm_weights, q_effort=effort, rho=rho)
        candidate = _RobustDesign(
            pi=pi,
            rho=rho,
            bonus=bonus,
            effort=effort,
            expected_budget=expected_budget,
            objective=objective,
        )
        if best is None or candidate.objective < best.objective:
            best = candidate
    if best is None:
        raise ValueError("rho_grid must contain at least one value in (0, 1)")
    return best


def _estimated_design_variance(
    tau: np.ndarray,
    pi: np.ndarray,
    norm_weights: np.ndarray,
    *,
    q_effort: float,
    rho: float,
) -> float:
    regular_probability = 1.0 - rho
    if q_effort <= 0.0 or regular_probability <= 0.0:
        return float("inf")
    positive_tau = tau > 0.0
    if np.any(pi[positive_tau] <= 0.0):
        return float("inf")
    return float(np.sum((norm_weights**2) * tau / (regular_probability * _positive_probabilities(pi) * q_effort)))


def _weighted_interval(
    contributions: np.ndarray, norm_weights: np.ndarray, confidence_level: float
) -> tuple[float, float, float]:
    estimate = float(np.sum(norm_weights * contributions))
    variance = float(np.sum((norm_weights**2) * (contributions - estimate) ** 2))
    se = float(np.sqrt(max(variance, 0.0)))
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return estimate, estimate - z_value * se, estimate + z_value * se


def _summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["method", "budget", "outcome"]
    summary = (
        trials.groupby(group_columns, sort=True, dropna=False)
        .agg(
            estimate=("estimate", "mean"),
            lower=("lower", "mean"),
            upper=("upper", "mean"),
            width=("width", "mean"),
            covered=("covered", "mean"),
            seed=("seed", _combine_seeds),
            n_trials=("seed", "size"),
            truth=("truth", "first"),
            n=("n", "first"),
            confidence_level=("confidence_level", "first"),
            baseline_effort=("baseline_effort", "first"),
            rho=("rho", "first"),
            bonus=("bonus", "first"),
            effort=("effort", "first"),
            q_effort=("q_effort", "first"),
            baseline_tau_mix=("baseline_tau_mix", "first"),
            design_objective=("design_objective", "first"),
            expected_budget=("expected_budget", "first"),
            expected_queries=("expected_queries", "first"),
            query_probability_mean=("query_probability_mean", "first"),
            budget_model=("budget_model", "first"),
            k_cost_interpretation=("k_cost_interpretation", "first"),
            assumption_ai_label_generation=("assumption_ai_label_generation", "first"),
            assumption_baseline_cost=("assumption_baseline_cost", "first"),
        )
        .reset_index()
    )
    return summary[_summary_columns()]


def _encode_covariates(clean: pd.DataFrame, covariates: Sequence[str], weight_column: str | None) -> pd.DataFrame:
    categorical_columns = [column for column in covariates if column != weight_column]
    pieces: list[pd.DataFrame] = []
    if categorical_columns:
        categorical = clean.loc[:, categorical_columns].astype("object")
        categorical = categorical.where(pd.notna(categorical), _MISSING_CATEGORY)
        pieces.append(pd.get_dummies(categorical, columns=categorical_columns, dtype=float))
    if weight_column and weight_column in covariates:
        numeric = pd.to_numeric(clean[weight_column], errors="coerce")
        fill_value = float(numeric.median()) if numeric.notna().any() else 1.0
        pieces.append(pd.DataFrame({weight_column: numeric.fillna(fill_value).astype(float)}))
    if not pieces:
        raise ValueError("no Pew covariates were available for feature encoding")
    encoded = pd.concat(pieces, axis=1)
    encoded.columns = [str(column) for column in encoded.columns]
    return encoded.astype(float)


def _coerce_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    return load_yaml_config(config)


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config key {key!r} must be a mapping")
    return value


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for column in candidates:
        if column in frame.columns:
            return str(column)
    raise KeyError(f"Could not find {label}; tried columns {list(candidates)!r}")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} are missing columns {missing!r}")


def _extract_weights(frame: pd.DataFrame, weight_column: Any) -> np.ndarray:
    if weight_column is None or str(weight_column) not in frame.columns:
        return np.ones(len(frame), dtype=float)
    weights = pd.to_numeric(frame[str(weight_column)], errors="coerce").to_numpy(dtype=float)
    return weights


def _select_seeds(config: Mapping[str, Any], smoke: bool) -> tuple[int, ...]:
    key = "smoke_seeds" if smoke else "seeds"
    seeds = tuple(int(seed) for seed in config.get(key, []))
    if not seeds:
        raise ValueError(f"config must define non-empty {key}")
    return seeds


def _select_budgets(config: Mapping[str, Any], smoke: bool) -> tuple[float, ...]:
    budget_config = _require_mapping(config, "budget_grid")
    key = "smoke" if smoke else "full"
    budgets = tuple(float(budget) for budget in budget_config.get(key, []))
    if not budgets:
        raise ValueError(f"config must define non-empty budget_grid.{key}")
    if any(budget <= 0.0 for budget in budgets):
        raise ValueError("budget grid values must be positive")
    return budgets


def _prediction_model(config: Mapping[str, Any]) -> str:
    prediction_config = config.get("prediction_model", {})
    if not isinstance(prediction_config, Mapping):
        return "auto"
    preferred = str(prediction_config.get("preferred_package", "auto")).lower()
    if preferred in {"xgboost", "xgb"}:
        return "auto"
    if preferred in {"sklearn", "hist_gradient_boosting", "hgb"}:
        return "sklearn"
    return preferred


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    norm_weights = _normalized_weights(weights)
    return float(np.sum(norm_weights * values))


def _normalized_weights(weights: np.ndarray) -> np.ndarray:
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim != 1 or weight_array.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    total = float(weight_array.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    return weight_array / total


def _uniform_probabilities(
    n: int,
    budget_or_target_mass: float,
    per_item_cost: np.ndarray | float | None = None,
) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    if per_item_cost is None:
        probability = min(max(float(budget_or_target_mass) / float(n), 0.0), 1.0)
        return np.full(n, probability, dtype=float)
    return scale_probabilities_to_budget(np.ones(n, dtype=float), budget_or_target_mass, per_item_cost)


def _mix_tau_with_uniform(tau: np.ndarray, mix: float) -> np.ndarray:
    if not (0.0 <= mix <= 1.0):
        raise ValueError("baseline_tau_grid values must lie in [0, 1]")
    tau_array = np.asarray(tau, dtype=float)
    if np.any(~np.isfinite(tau_array)) or np.any(tau_array < 0.0):
        raise ValueError("tau must contain finite nonnegative values")
    baseline = float(np.mean(tau_array))
    mixed = (1.0 - mix) * tau_array + mix * baseline
    return np.maximum(mixed, 1e-12)


def _probabilities_from_scores(scores: np.ndarray, target_mass: float) -> np.ndarray:
    score_array = np.asarray(scores, dtype=float)
    if np.any(~np.isfinite(score_array)) or np.any(score_array < 0.0):
        raise ValueError("sampling scores must be finite and nonnegative")
    n = score_array.size
    if n == 0:
        raise ValueError("sampling scores must not be empty")
    remaining = min(max(float(target_mass), 0.0), float(n))
    if remaining <= 0.0 or score_array.sum() <= 0.0:
        return np.zeros(n, dtype=float)
    allocation = np.zeros(n, dtype=float)
    active = score_array > 0.0
    while remaining > 0.0 and np.any(active):
        active_scores = score_array[active]
        candidate = remaining * active_scores / float(active_scores.sum())
        saturated = candidate >= 1.0
        active_indices = np.flatnonzero(active)
        if not np.any(saturated):
            allocation[active] = candidate
            return allocation
        saturated_indices = active_indices[saturated]
        allocation[saturated_indices] = 1.0
        active[saturated_indices] = False
        remaining -= float(len(saturated_indices))
    return allocation


def _select_prediction_seeds(
    config: Mapping[str, Any],
    smoke: bool,
    experiment_seeds: Sequence[int],
) -> tuple[int, ...]:
    if smoke and "smoke_prediction_seeds" in config:
        raw = config["smoke_prediction_seeds"]
    elif "prediction_seeds" in config:
        raw = config["prediction_seeds"]
    elif smoke:
        raw = [experiment_seeds[0]]
    else:
        raw = experiment_seeds[: min(5, len(experiment_seeds))]
    seeds = tuple(int(seed) for seed in raw)
    if not seeds:
        raise ValueError("prediction seeds must contain at least one seed")
    return seeds


def _baseline_accuracy_cost(
    y_true: np.ndarray,
    f: np.ndarray,
    baseline_effort: float,
    w0: float,
    incentive_config: Mapping[str, Any],
) -> np.ndarray:
    model = str(incentive_config.get("baseline_payment_model", "linear_accuracy_based")).lower()
    effort = float(baseline_effort)
    if not np.isfinite(effort) or effort <= 0.0 or effort > 1.0:
        raise ValueError("baseline_effort must lie in (0, 1]")
    if model in {"none", "w0_only"}:
        return np.full(y_true.size, float(w0), dtype=float)
    if model in {"effort_squared", "sentinel_equivalent"}:
        return np.full(y_true.size, float(w0) + effort**2, dtype=float)
    if model != "linear_accuracy_based":
        raise ValueError(f"unknown baseline_payment_model {model!r}")

    error_floor = float(incentive_config.get("ai_error_floor", 0.02))
    if not np.isfinite(error_floor) or error_floor <= 0.0 or error_floor > 1.0:
        raise ValueError("ai_error_floor must lie in (0, 1]")
    y = np.asarray(y_true, dtype=float)
    probability = np.clip(np.asarray(f, dtype=float), 0.0, 1.0)
    p_error = np.where(y >= 0.5, 1.0 - probability, probability)
    p_error = np.clip(p_error, error_floor, 1.0)
    expected_accuracy_payment = effort * (1.0 - p_error * (1.0 - effort)) / p_error
    return np.asarray(float(w0) + expected_accuracy_payment, dtype=float)


def _cost_aware_active_weights(tau: np.ndarray, per_item_cost: np.ndarray) -> np.ndarray:
    tau_array = np.asarray(tau, dtype=float)
    cost_array = np.asarray(per_item_cost, dtype=float)
    if np.any(~np.isfinite(cost_array)) or np.any(cost_array <= 0.0):
        raise ValueError("per-item costs must be positive and finite")
    weights = np.sqrt(np.maximum(tau_array, 0.0) / cost_array)
    if not np.any(weights > 0.0):
        weights = np.ones_like(tau_array, dtype=float)
    return weights


def _positive_probabilities(pi: np.ndarray) -> np.ndarray:
    pi_array = np.asarray(pi, dtype=float)
    if np.any(~np.isfinite(pi_array)) or np.any(pi_array < 0.0) or np.any(pi_array > 1.0):
        raise ValueError("sampling probabilities must lie in [0, 1]")
    if np.any(pi_array <= 0.0):
        return np.maximum(pi_array, np.finfo(float).tiny)
    return pi_array


def _method_seed(seed: int, budget: float, method: str) -> int:
    method_code = zlib.crc32(method.encode("utf-8"))
    budget_code = int(round(float(budget) * 1000.0))
    sequence = np.random.SeedSequence([int(seed), budget_code, method_code])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _combine_seeds(values: pd.Series) -> tuple[int, ...] | int:
    unique = tuple(dict.fromkeys(int(value) for value in values))
    return unique[0] if len(unique) == 1 else unique


def _trial_columns() -> list[str]:
    return [
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
        "n",
        "confidence_level",
        "baseline_effort",
        "rho",
        "bonus",
        "effort",
        "q_effort",
        "baseline_tau_mix",
        "design_objective",
        "expected_budget",
        "expected_queries",
        "query_probability_mean",
        "sampled_count",
        "sentinel_count",
        "budget_model",
        "k_cost_interpretation",
        "assumption_ai_label_generation",
        "assumption_baseline_cost",
    ]


def _summary_columns() -> list[str]:
    return [
        "method",
        "budget",
        "outcome",
        "estimate",
        "lower",
        "upper",
        "width",
        "covered",
        "seed",
        "n_trials",
        "truth",
        "n",
        "confidence_level",
        "baseline_effort",
        "rho",
        "bonus",
        "effort",
        "q_effort",
        "baseline_tau_mix",
        "design_objective",
        "expected_budget",
        "expected_queries",
        "query_probability_mean",
        "budget_model",
        "k_cost_interpretation",
        "assumption_ai_label_generation",
        "assumption_baseline_cost",
    ]


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Pew Wave 79 mean-estimation experiments.")
    parser.add_argument("--config", default="configs/pew.yaml", help="Path to the Pew YAML config.")
    parser.add_argument("--outcome", default="biden", choices=("biden", "trump"), help="Outcome to run.")
    parser.add_argument("--smoke", action="store_true", help="Use the smoke seed and budget grids.")
    parser.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="In smoke mode only, skip with exit 0 if the local SPSS reader/data are unavailable.",
    )
    args = parser.parse_args(argv)

    try:
        trials, summary = run_pew_mean_experiment(args.config, args.outcome, args.smoke)
    except (ImportError, FileNotFoundError) as exc:
        if args.smoke and args.allow_missing_data:
            print(f"Skipping Pew smoke run: {exc}")
            return 0
        raise

    print(
        f"Wrote {len(trials)} trial rows and {len(summary)} summary rows for "
        f"Pew {args.outcome} to {_OUTPUT_DIR}."
    )
    return 0


__all__ = [
    "PewPreparedData",
    "load_pew_wave79",
    "recode_message_outcome",
    "prepare_pew_dataset",
    "run_pew_mean_experiment",
]


if __name__ == "__main__":
    raise SystemExit(_main())
