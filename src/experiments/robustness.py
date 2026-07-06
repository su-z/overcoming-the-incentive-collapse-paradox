"""Pew Biden robustness sensitivity experiments from the ICML paper.

The experiments in this module reuse the Pew Wave 79 preprocessing, prediction,
sampling, and nominal budget design from :mod:`experiments.pew`.  The sensitivity
intervention is applied only when simulating actual annotator effort:

* posterior belief: non-sentinel regular robust tasks use
  ``rho_posterior = posterior_scale * rho`` in the first-order condition;
* misspecification: the robust annotator's actual effort is ``kappa`` times the
  principal's predicted effort, with ``kappa`` drawn once from the trial RNG.

The nominal sampling and budget policy remains the main-text Pew design.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments import pew
from experiments.common import (
    crossfit_binary_predictions,
    load_yaml_config,
)
from incentive.design import effort_from_bonus, misspecified_effort
from incentive.simulation import simulate_human_labels


_OUTPUT_DIR = Path("outputs") / "tables"
_METHODS = ("classical", "uniform", "active", "robust")
_POSTERIOR = "posterior"
_MISSPECIFICATION = "misspecification"
_SENSITIVITY_ALIASES = {
    _POSTERIOR: _POSTERIOR,
    "posterior_belief": _POSTERIOR,
    "rho_posterior": _POSTERIOR,
    _MISSPECIFICATION: _MISSPECIFICATION,
    "misspecified": _MISSPECIFICATION,
}


def run_posterior_belief_experiment(
    config_path: str | Path = "configs/robustness.yaml", smoke: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the posterior-belief sensitivity experiment and save CSV outputs."""

    config = load_yaml_config(config_path)
    posterior_config = _sensitivity_config(config, "posterior_belief")
    posterior_scale = _posterior_scale(posterior_config)
    return _run_pew_biden_sensitivity(
        config,
        smoke=smoke,
        sensitivity=_POSTERIOR,
        posterior_scale=posterior_scale,
        kappa_bounds=(np.nan, np.nan),
    )


def run_misspecification_experiment(
    config_path: str | Path = "configs/robustness.yaml", smoke: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the utility/cost misspecification sensitivity and save CSV outputs."""

    config = load_yaml_config(config_path)
    misspecification_config = _sensitivity_config(config, "misspecification")
    return _run_pew_biden_sensitivity(
        config,
        smoke=smoke,
        sensitivity=_MISSPECIFICATION,
        posterior_scale=np.nan,
        kappa_bounds=_kappa_bounds(misspecification_config),
    )


def _run_pew_biden_sensitivity(
    config: Mapping[str, Any],
    *,
    smoke: bool,
    sensitivity: str,
    posterior_scale: float,
    kappa_bounds: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_map, outcome = _pew_biden_config(config)
    prepared = pew.prepare_pew_dataset(config_map, outcome)
    seeds = pew._select_seeds(config_map, smoke)
    budgets = pew._select_budgets(config_map, smoke)
    confidence_level = float(config_map.get("confidence_level", 0.90))
    baseline_effort = float(config_map.get("baseline_effort", 0.8))
    incentive_config = _require_mapping(config_map, "incentive")
    sampling_config = _require_mapping(config_map, "sampling")

    model = pew._prediction_model(config_map)
    prediction_seeds = pew._select_prediction_seeds(config_map, smoke, seeds)
    predictions = crossfit_binary_predictions(
        prepared.X.to_numpy(dtype=float),
        prepared.y,
        prediction_seeds,
        model=model,
    )
    f = np.clip(predictions.mean_prediction, 0.0, 1.0)
    tau = np.asarray(predictions.tau, dtype=float)
    truth = pew._weighted_mean(prepared.y, prepared.weights)
    norm_weights = pew._normalized_weights(prepared.weights)

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
                    sensitivity=sensitivity,
                    posterior_scale=posterior_scale,
                    kappa_bounds=kappa_bounds,
                )
            )

    trials = pd.DataFrame.from_records(trial_rows)
    trials = trials[_trial_columns()]
    summary = _summarize_trials(trials)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trial_path, summary_path = _output_paths(sensitivity)
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
    sensitivity: str,
    posterior_scale: float,
    kappa_bounds: tuple[float, float],
) -> list[dict[str, Any]]:
    w0 = float(incentive_config.get("w0", 0.64))
    k = float(incentive_config.get("k", 1.0))
    baseline_tau_grid = [float(value) for value in sampling_config.get("baseline_tau_grid", [0.0])]
    baseline_cost = pew._baseline_accuracy_cost(y_true, f, baseline_effort, w0, incentive_config)

    uniform_pi = pew._uniform_probabilities(n, budget, per_item_cost=baseline_cost)
    active_design = pew._select_active_design(
        tau=tau,
        budget=budget,
        baseline_cost=baseline_cost,
        baseline_tau_grid=baseline_tau_grid,
        norm_weights=norm_weights,
        q_effort=baseline_effort,
    )
    robust_design = pew._select_robust_design(
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
            "predicted_effort": baseline_effort,
            "expected_budget": float(pew.sampling_expected_budget(uniform_pi, baseline_cost)),
            "mix": 1.0,
            "objective": np.nan,
        },
        "uniform": {
            "pi": uniform_pi,
            "rho": 0.0,
            "bonus": 0.0,
            "predicted_effort": baseline_effort,
            "expected_budget": float(pew.sampling_expected_budget(uniform_pi, baseline_cost)),
            "mix": 1.0,
            "objective": np.nan,
        },
        "active": {
            "pi": active_design.pi,
            "rho": 0.0,
            "bonus": 0.0,
            "predicted_effort": baseline_effort,
            "expected_budget": float(pew.sampling_expected_budget(active_design.pi, baseline_cost)),
            "mix": active_design.mix,
            "objective": active_design.objective,
        },
        "robust": {
            "pi": robust_design.pi,
            "rho": robust_design.rho,
            "bonus": robust_design.bonus,
            "predicted_effort": robust_design.effort,
            "expected_budget": robust_design.expected_budget,
            "mix": 0.0,
            "objective": robust_design.objective,
        },
    }

    rows = []
    for method in _METHODS:
        design = designs[method]
        method_seed = pew._method_seed(seed, budget, method)
        rows.append(
            _evaluate_method(
                method=method,
                y_true=y_true,
                f=f,
                pi=np.asarray(design["pi"], dtype=float),
                rho=float(design["rho"]),
                bonus=float(design["bonus"]),
                predicted_effort=float(design["predicted_effort"]),
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
                sensitivity=sensitivity,
                posterior_scale=posterior_scale,
                kappa_bounds=kappa_bounds,
            )
        )
    return rows


def _evaluate_method(
    *,
    method: str,
    y_true: np.ndarray,
    f: np.ndarray,
    pi: np.ndarray,
    rho: float,
    bonus: float,
    predicted_effort: float,
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
    sensitivity: str,
    posterior_scale: float,
    kappa_bounds: tuple[float, float],
) -> dict[str, Any]:
    rng = np.random.default_rng(method_seed)
    kappa = np.nan
    actual_effort = float(predicted_effort)
    sentinel_effort = float(predicted_effort)

    if sensitivity == _POSTERIOR:
        if method == "robust":
            actual_effort = float(effort_from_bonus(rho, bonus, posterior_scale=posterior_scale))
    elif sensitivity == _MISSPECIFICATION:
        if method == "robust":
            kappa = float(rng.uniform(kappa_bounds[0], kappa_bounds[1]))
            actual_effort = float(misspecified_effort(predicted_effort, kappa))
            sentinel_effort = actual_effort
    else:
        raise ValueError(f"unknown robustness sensitivity {sensitivity!r}")

    xi = (rng.random(y_true.size) < pi).astype(float)
    ai_labels = (rng.random(y_true.size) < f).astype(float)

    if method == "robust":
        sentinel = (xi == 1.0) & (rng.random(y_true.size) < rho)
        zeta = ((xi == 1.0) & ~sentinel).astype(float)
    else:
        sentinel = np.zeros(y_true.size, dtype=bool)
        zeta = xi.copy()

    effort_for_simulation: float | np.ndarray
    if sensitivity == _POSTERIOR and method == "robust":
        effort_for_simulation = np.where(sentinel, sentinel_effort, actual_effort)
    else:
        effort_for_simulation = actual_effort

    simulated = simulate_human_labels(
        y_true,
        ai_labels,
        effort=effort_for_simulation,
        rng=rng,
        sentinel=sentinel,
    )
    y_reported = simulated["reported"].astype(float)
    classical_label_accuracy = (
        pew._final_label_accuracy_probability(y_true, f, actual_effort)
        if method == "classical"
        else None
    )
    contributions = pew._mean_contributions(
        method,
        y_reported,
        f,
        xi,
        zeta,
        pi,
        rho,
        actual_effort,
        classical_label_accuracy=classical_label_accuracy,
    )
    estimate, lower, upper = pew._weighted_interval(contributions, norm_weights, confidence_level)

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
        "effort": float(actual_effort),
        "q_effort": float(actual_effort),
        "baseline_tau_mix": float(mix),
        "design_objective": objective,
        "expected_budget": float(expected_budget),
        "expected_queries": float(pi.sum()),
        "query_probability_mean": float(np.mean(pi)),
        "sampled_count": int(xi.sum()),
        "sentinel_count": int(sentinel.sum()),
        "budget_model": pew._BUDGET_MODEL,
        "k_cost_interpretation": pew._K_COST_INTERPRETATION,
        "assumption_ai_label_generation": "bernoulli_crossfit_probability",
        "assumption_baseline_cost": "linear_accuracy_based_payment_for_e_0.8",
        "sensitivity": sensitivity,
        "predicted_effort": float(predicted_effort),
        "actual_effort": float(actual_effort),
        "posterior_scale": float(posterior_scale) if np.isfinite(posterior_scale) else np.nan,
        "kappa": float(kappa) if np.isfinite(kappa) else np.nan,
        "kappa_low": float(kappa_bounds[0]) if np.isfinite(kappa_bounds[0]) else np.nan,
        "kappa_high": float(kappa_bounds[1]) if np.isfinite(kappa_bounds[1]) else np.nan,
        "assumption_sensitivity": _sensitivity_assumption(sensitivity),
    }


def _summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["method", "budget", "outcome", "sensitivity"]
    summary = (
        trials.groupby(group_columns, sort=True, dropna=False)
        .agg(
            estimate=("estimate", "mean"),
            lower=("lower", "mean"),
            upper=("upper", "mean"),
            width=("width", "mean"),
            covered=("covered", "mean"),
            seed=("seed", pew._combine_seeds),
            n_trials=("seed", "size"),
            truth=("truth", "first"),
            n=("n", "first"),
            confidence_level=("confidence_level", "first"),
            baseline_effort=("baseline_effort", "first"),
            rho=("rho", "first"),
            bonus=("bonus", "first"),
            effort=("effort", "mean"),
            q_effort=("q_effort", "mean"),
            baseline_tau_mix=("baseline_tau_mix", "first"),
            design_objective=("design_objective", "first"),
            expected_budget=("expected_budget", "first"),
            expected_queries=("expected_queries", "first"),
            query_probability_mean=("query_probability_mean", "first"),
            budget_model=("budget_model", "first"),
            k_cost_interpretation=("k_cost_interpretation", "first"),
            assumption_ai_label_generation=("assumption_ai_label_generation", "first"),
            assumption_baseline_cost=("assumption_baseline_cost", "first"),
            predicted_effort=("predicted_effort", "mean"),
            actual_effort=("actual_effort", "mean"),
            posterior_scale=("posterior_scale", "first"),
            kappa=("kappa", "mean"),
            kappa_low=("kappa_low", "first"),
            kappa_high=("kappa_high", "first"),
            assumption_sensitivity=("assumption_sensitivity", "first"),
        )
        .reset_index()
    )
    return summary[_summary_columns()]


def _pew_biden_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    config_map = dict(config)
    data_config = dict(_require_mapping(config_map, "data"))

    outcome = "biden"
    if isinstance(data_config.get("outcome"), Mapping):
        outcome_config = dict(data_config["outcome"])
        outcome = str(outcome_config.pop("name", "biden")).lower()
        if "outcomes" not in data_config:
            data_config["outcomes"] = {outcome: outcome_config}
    elif isinstance(data_config.get("outcome"), str):
        outcome = str(data_config["outcome"]).lower()

    if "outcomes" in data_config:
        outcomes = _require_mapping(data_config, "outcomes")
        if outcome not in outcomes and "biden" in outcomes:
            outcome = "biden"
    config_map["data"] = data_config
    if outcome != "biden":
        raise ValueError("robustness experiments use the Pew Biden outcome")
    return config_map, outcome


def _sensitivity_config(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    robustness = _require_mapping(config, "robustness")
    if name not in robustness:
        known = ", ".join(sorted(str(key) for key in robustness))
        raise ValueError(f"unknown robustness sensitivity {name!r}; known sensitivities: {known}")
    value = robustness[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"robustness.{name} must be a mapping")
    return value


def _posterior_scale(config: Mapping[str, Any]) -> float:
    scale = float(config.get("posterior_scale", 0.8))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("posterior_scale must be positive")
    return scale


def _kappa_bounds(config: Mapping[str, Any]) -> tuple[float, float]:
    raw_bounds = config.get("kappa_uniform", [0.25, 0.75])
    if not isinstance(raw_bounds, Sequence) or isinstance(raw_bounds, (str, bytes)) or len(raw_bounds) != 2:
        raise ValueError("kappa_uniform must contain exactly two numeric bounds")
    low, high = (float(raw_bounds[0]), float(raw_bounds[1]))
    if not np.isfinite(low) or not np.isfinite(high) or low < 0.0 or high <= low:
        raise ValueError("kappa_uniform must satisfy 0 <= low < high")
    return low, high


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config key {key!r} must be a mapping")
    return value


def _normalize_which(which: str) -> str:
    key = str(which).lower()
    try:
        return _SENSITIVITY_ALIASES[key]
    except KeyError as exc:
        known = ", ".join(sorted(_SENSITIVITY_ALIASES))
        raise ValueError(f"unknown robustness sensitivity {which!r}; expected one of: {known}") from exc


def _run_selected(
    which: str, config_path: str | Path, smoke: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensitivity = _normalize_which(which)
    if sensitivity == _POSTERIOR:
        return run_posterior_belief_experiment(config_path=config_path, smoke=smoke)
    if sensitivity == _MISSPECIFICATION:
        return run_misspecification_experiment(config_path=config_path, smoke=smoke)
    raise ValueError(f"unknown robustness sensitivity {which!r}")


def _output_paths(sensitivity: str) -> tuple[Path, Path]:
    if sensitivity == _POSTERIOR:
        return (
            _OUTPUT_DIR / "robustness_posterior_trials.csv",
            _OUTPUT_DIR / "robustness_posterior_summary.csv",
        )
    if sensitivity == _MISSPECIFICATION:
        return (
            _OUTPUT_DIR / "robustness_misspecification_trials.csv",
            _OUTPUT_DIR / "robustness_misspecification_summary.csv",
        )
    raise ValueError(f"unknown robustness sensitivity {sensitivity!r}")


def _sensitivity_assumption(sensitivity: str) -> str:
    if sensitivity == _POSTERIOR:
        return "robust regular-task effort uses rho_posterior=posterior_scale*rho; nominal budget design uses rho"
    if sensitivity == _MISSPECIFICATION:
        return "robust actual effort is kappa times the principal predicted effort; baselines retain configured effort"
    raise ValueError(f"unknown robustness sensitivity {sensitivity!r}")


def _trial_columns() -> list[str]:
    return [
        *pew._trial_columns(),
        "sensitivity",
        "predicted_effort",
        "actual_effort",
        "posterior_scale",
        "kappa",
        "kappa_low",
        "kappa_high",
        "assumption_sensitivity",
    ]


def _summary_columns() -> list[str]:
    return [
        *pew._summary_columns(),
        "sensitivity",
        "predicted_effort",
        "actual_effort",
        "posterior_scale",
        "kappa",
        "kappa_low",
        "kappa_high",
        "assumption_sensitivity",
    ]


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Pew Biden robustness sensitivity experiments.")
    parser.add_argument("--config", default="configs/robustness.yaml", help="Path to the robustness YAML config.")
    parser.add_argument(
        "--which",
        default=_POSTERIOR,
        help="Sensitivity to run: posterior or misspecification.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use the smoke seed and budget grids.")
    parser.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="In smoke mode only, skip with exit 0 if the local SPSS reader/data are unavailable.",
    )
    args = parser.parse_args(argv)

    try:
        sensitivity = _normalize_which(args.which)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        trials, summary = _run_selected(sensitivity, args.config, args.smoke)
    except (ImportError, FileNotFoundError) as exc:
        if args.smoke and args.allow_missing_data:
            print(f"Skipping robustness smoke run: {exc}")
            return 0
        raise

    print(
        f"Wrote {len(trials)} trial rows and {len(summary)} summary rows for "
        f"robustness {sensitivity} to {_OUTPUT_DIR}."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point wrapper."""

    return _main(argv)


__all__ = [
    "run_posterior_belief_experiment",
    "run_misspecification_experiment",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
