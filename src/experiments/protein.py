"""AlphaFold protein odds-ratio experiment from the ICML paper.

The paper's main text names the protein estimand as the odds ratio between IDR
status and phosphorylation status. This module therefore targets

``odds(IDR = 1 | phosphorylated = 1) / odds(IDR = 1 | phosphorylated = 0)``.

The implementation uses the bundled AlphaFold arrays only: ``Y`` is the true
IDR label, ``Yhat`` is retained as the AI probability, and ``Yhat > 0.5`` is
the displayed binary AI label used in the label-correction estimators.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from experiments.common import (
    TrialResult,
    load_yaml_config,
    make_rng,
    summarize_intervals,
    tau_from_binary_predictions,
)
from inference.intervals import interval_width, z_value
from inference.sampling import (
    expected_budget,
    incentive_aware_probabilities,
    scale_probabilities_to_budget,
)


REQUIRED_ALPHAFOLD_KEYS = ("Y", "Yhat", "phosphorylated", "ubiquitinated", "acetylated")
METHODS = ("classical", "uniform", "active", "robust")
TRIAL_COLUMNS = ["method", "budget", "estimate", "lower", "upper", "width", "covered", "seed"]
ODDS_BOUNDARY_TOLERANCE = 1e-10
UNSTABLE_RATIO_RADIUS = 1e6


@dataclass(frozen=True)
class AlphaFoldData:
    """Validated arrays from the bundled AlphaFold archive."""

    y_idr: np.ndarray
    ai_probability: np.ndarray
    ai_label: np.ndarray
    phosphorylated: np.ndarray
    ubiquitinated: np.ndarray
    acetylated: np.ndarray

    @property
    def n(self) -> int:
        return int(self.y_idr.size)


@dataclass(frozen=True)
class ProteinDataset:
    """Prepared data for the protein odds-ratio experiment."""

    y_idr: np.ndarray
    ai_probability: np.ndarray
    ai_label: np.ndarray
    phosphorylated: np.ndarray
    tau: np.ndarray
    truth: float
    metadata: Mapping[str, Any]

    @property
    def n(self) -> int:
        return int(self.y_idr.size)


def load_alphafold_npz(path: str | Path) -> AlphaFoldData:
    """Load and validate the bundled AlphaFold ``.npz`` arrays.

    Required keys are ``Y``, ``Yhat``, ``phosphorylated``, ``ubiquitinated``,
    and ``acetylated``.  Arrays are coerced to 1-D NumPy arrays.  ``Y`` is the
    true IDR label, ``Yhat`` is validated as an AI probability in ``[0, 1]``,
    and ``Yhat > 0.5`` is exposed as the binary AI label shown to annotators.
    The phosphorylated array is the subgroup indicator used by the estimand.
    """

    npz_path = Path(path)
    with np.load(npz_path) as archive:
        missing = [key for key in REQUIRED_ALPHAFOLD_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"AlphaFold archive is missing required keys: {missing}")
        arrays = {key: _as_1d_array(key, archive[key]) for key in REQUIRED_ALPHAFOLD_KEYS}

    lengths = {key: value.size for key, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"AlphaFold arrays must have a common length, got {lengths}")
    if next(iter(lengths.values())) == 0:
        raise ValueError("AlphaFold arrays must not be empty")

    y_idr = _validate_probability_array("Y", arrays["Y"])
    ai_probability = _validate_probability_array("Yhat", arrays["Yhat"])
    phosphorylated = _validate_binary_array("phosphorylated", arrays["phosphorylated"])
    ubiquitinated = _validate_binary_array("ubiquitinated", arrays["ubiquitinated"])
    acetylated = _validate_binary_array("acetylated", arrays["acetylated"])
    _validate_subgroups(phosphorylated)

    return AlphaFoldData(
        y_idr=y_idr,
        ai_probability=ai_probability,
        ai_label=(ai_probability > 0.5).astype(float),
        phosphorylated=phosphorylated,
        ubiquitinated=ubiquitinated,
        acetylated=acetylated,
    )


def prepare_protein_dataset(config: Mapping[str, Any] | str | Path) -> ProteinDataset:
    """Prepare AlphaFold data and squared-residual ``tau`` scores.

    ``tau`` follows the paper's active-inference notation and is computed from
    squared residuals between the true IDR label ``Y`` and the AI probability
    ``Yhat``.
    """

    config_map, base_dir = _coerce_config(config)
    data_config = _mapping(config_map.get("data", {}), "data")
    raw_file = data_config.get("raw_file", "data/alphafold/alphafold.npz")
    raw_path = _resolve_path(raw_file, base_dir)
    loaded = load_alphafold_npz(raw_path)
    tau = tau_from_binary_predictions(loaded.y_idr, loaded.ai_probability)
    truth = protein_truth(loaded.y_idr, loaded.phosphorylated)
    metadata = {
        "experiment": str(config_map.get("experiment", "alphafold_protein_odds_ratio")),
        "estimand": str(data_config.get("estimand", "odds_ratio")),
        "target_parameter": "odds(IDR=1|phosphorylated=1)/odds(IDR=1|phosphorylated=0)",
        "target_assumption": "paper main-text odds-ratio wording is implemented as a binary odds ratio",
        "tau_source": "squared_prediction_residuals",
        "ai_label_threshold": 0.5,
        "subgroup": "phosphorylated",
        "n": loaded.n,
    }
    return ProteinDataset(
        y_idr=loaded.y_idr,
        ai_probability=loaded.ai_probability,
        ai_label=loaded.ai_label,
        phosphorylated=loaded.phosphorylated,
        tau=tau,
        truth=truth,
        metadata=metadata,
    )


def protein_truth(y_idr: Any, phosphorylated: Any) -> float:
    """Return the binary odds ratio for IDR status by phosphorylation group."""

    y = _validate_probability_array("y_idr", _as_1d_array("y_idr", y_idr))
    subgroup = _validate_binary_array("phosphorylated", _as_1d_array("phosphorylated", phosphorylated))
    if subgroup.size != y.size:
        raise ValueError("phosphorylated must have the same length as y_idr")
    _validate_subgroups(subgroup)

    treated = subgroup.astype(bool)
    treated_mean = float(np.mean(y[treated]))
    untreated_mean = float(np.mean(y[~treated]))
    _validate_probability_for_odds("phosphorylated IDR mean", treated_mean)
    _validate_probability_for_odds("unphosphorylated IDR mean", untreated_mean)
    return _odds_ratio_from_means(treated_mean, untreated_mean)


def run_protein_experiment(
    config_path: str | Path = "configs/protein.yaml", smoke: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the AlphaFold protein experiment and save trial and summary CSVs."""

    config, base_dir = _coerce_config(config_path)
    dataset = prepare_protein_dataset({**config, "_config_dir": base_dir})
    confidence_level = _confidence_level(config)
    seeds = _seeds(config, smoke=smoke)
    budgets = _budgets(config, smoke=smoke)
    design = _design_config(config)

    trial_results: list[TrialResult] = []
    trial_rows: list[dict[str, Any]] = []
    for budget in budgets:
        designs = _method_designs(dataset, budget, design)
        for seed in seeds:
            for method in METHODS:
                row, result = _run_method_trial(
                    dataset=dataset,
                    method=method,
                    budget=budget,
                    seed=seed,
                    method_design=designs[method],
                    confidence_level=confidence_level,
                    base_metadata=dataset.metadata,
                )
                trial_rows.append(row)
                trial_results.append(result)

    trial_frame = _ordered_frame(pd.DataFrame.from_records(trial_rows), TRIAL_COLUMNS)
    summary_frame = summarize_intervals(trial_results, truth=dataset.truth)
    summary_frame = _ordered_frame(summary_frame, TRIAL_COLUMNS)

    tables_dir = _tables_dir(config, base_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    trial_frame.to_csv(tables_dir / "protein_trials.csv", index=False)
    summary_frame.to_csv(tables_dir / "protein_summary.csv", index=False)
    return trial_frame, summary_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/protein.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run_protein_experiment(config_path=args.config, smoke=args.smoke)
    return 0


def _method_designs(
    dataset: ProteinDataset, budget: float, design: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    n = dataset.n
    w0 = float(design["w0"])
    k = float(design["k"])
    baseline_effort = float(design["baseline_effort"])
    baseline_mix_tau = float(design["baseline_mix_tau"])
    baseline_cost = _baseline_accuracy_cost(
        dataset.y_idr,
        dataset.ai_probability,
        baseline_effort,
        w0,
        design,
    )

    classical_pi = scale_probabilities_to_budget(np.ones(n, dtype=float), budget, baseline_cost)
    uniform_pi = scale_probabilities_to_budget(np.ones(n, dtype=float), budget, baseline_cost)
    active_weights = _mixed_cost_aware_weights(dataset.tau, baseline_mix_tau, baseline_cost)
    active_pi = scale_probabilities_to_budget(active_weights, budget, baseline_cost)
    robust_rho, robust_pi = _select_robust_design(dataset.tau, budget, design)
    robust_effort = float(np.sqrt(w0))
    _validate_effort_probability("baseline_effort", baseline_effort)
    _validate_effort_probability("robust_effort", robust_effort)

    return {
        "classical": {
            "pi": _positive_probabilities("classical pi", classical_pi),
            "q_effort": baseline_effort,
            "rho": np.nan,
            "expected_budget": expected_budget(classical_pi, baseline_cost),
            "bonus": np.nan,
        },
        "uniform": {
            "pi": _positive_probabilities("uniform pi", uniform_pi),
            "q_effort": baseline_effort,
            "rho": np.nan,
            "expected_budget": expected_budget(uniform_pi, baseline_cost),
            "bonus": np.nan,
        },
        "active": {
            "pi": _positive_probabilities("active pi", active_pi),
            "q_effort": baseline_effort,
            "rho": np.nan,
            "expected_budget": expected_budget(active_pi, baseline_cost),
            "bonus": np.nan,
        },
        "robust": {
            "pi": _positive_probabilities("robust pi", robust_pi),
            "q_effort": robust_effort,
            "rho": robust_rho,
            "expected_budget": expected_budget(
                robust_pi,
                per_item_cost=2.0 * w0,
                fixed_cost=robust_rho * k,
            ),
            "bonus": float(np.sqrt(w0) / robust_rho),
        },
    }


def _run_method_trial(
    *,
    dataset: ProteinDataset,
    method: str,
    budget: float,
    seed: int,
    method_design: Mapping[str, Any],
    confidence_level: float,
    base_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], TrialResult]:
    rng = make_rng(_method_seed(seed, method))
    y_reported = _simulate_reported_labels(
        dataset.y_idr,
        dataset.ai_label,
        q_effort=float(method_design["q_effort"]),
        rng=rng,
    )
    pi = np.asarray(method_design["pi"], dtype=float)
    xi = rng.binomial(1, pi).astype(float)

    if method == "classical":
        label_accuracy = _final_label_accuracy(
            dataset.y_idr,
            dataset.ai_label,
            float(method_design["q_effort"]),
        )
        corrected = _classical_corrected_values(y_reported, xi, pi, label_accuracy)
        variance = _classical_conditional_variance(
            dataset.y_idr,
            pi,
            label_accuracy,
        )
        boundary_counts = _effective_group_counts(dataset.phosphorylated, pi)
    elif method in {"uniform", "active"}:
        q_effort = float(method_design["q_effort"])
        corrected = _active_corrected_values(
            y_reported,
            dataset.ai_label,
            xi,
            pi,
            q_effort,
        )
        variance = _residual_conditional_variance(
            dataset.y_idr,
            dataset.ai_label,
            pi,
            q_effort,
            regular_probability=1.0,
        )
        boundary_counts = None
    elif method == "robust":
        rho = float(method_design["rho"])
        q_effort = float(method_design["q_effort"])
        zeta = np.zeros(dataset.n, dtype=float)
        sampled = xi.astype(bool)
        zeta[sampled] = rng.binomial(1, 1.0 - rho, size=int(np.sum(sampled)))
        corrected = _robust_corrected_values(
            y_reported,
            dataset.ai_label,
            xi,
            zeta,
            pi,
            rho,
            q_effort,
        )
        variance = _residual_conditional_variance(
            dataset.y_idr,
            dataset.ai_label,
            pi,
            q_effort,
            regular_probability=1.0 - rho,
        )
        boundary_counts = None
    else:
        raise ValueError(f"unknown method {method!r}")

    estimate, lower, upper, numerator, denominator = _odds_ratio_interval(
        corrected,
        dataset.phosphorylated,
        variance,
        confidence_level,
        boundary_counts=boundary_counts,
    )
    metadata = {
        **base_metadata,
        "confidence_level": confidence_level,
        "truth": dataset.truth,
        "q_effort": float(method_design["q_effort"]),
        "rho": float(method_design["rho"]) if np.isfinite(method_design["rho"]) else np.nan,
        "bonus": float(method_design["bonus"]) if np.isfinite(method_design["bonus"]) else np.nan,
        "expected_budget": float(method_design["expected_budget"]),
    }
    row = {
        "method": method,
        "budget": float(budget),
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "width": interval_width((lower, upper)),
        "covered": bool(lower <= dataset.truth <= upper),
        "seed": int(seed),
        "treated_idr_mean": numerator,
        "untreated_idr_mean": denominator,
        **metadata,
    }
    result = TrialResult(
        method=method,
        budget=float(budget),
        estimate=estimate,
        lower=lower,
        upper=upper,
        seed=int(seed),
        metadata=metadata,
    )
    return row, result


def _simulate_reported_labels(
    y_true: np.ndarray,
    ai_label: np.ndarray,
    q_effort: float,
    rng: np.random.Generator,
) -> np.ndarray:
    _validate_effort_probability("q_effort", q_effort)
    ai_error = ai_label != y_true
    corrected = rng.binomial(1, q_effort, size=y_true.size).astype(bool)
    return np.where(ai_error & ~corrected, ai_label, y_true).astype(float)


def _classical_corrected_values(
    y_reported: np.ndarray,
    xi: np.ndarray,
    pi: np.ndarray,
    label_accuracy: np.ndarray,
) -> np.ndarray:
    accuracy = np.asarray(label_accuracy, dtype=float)
    if np.any(~np.isfinite(accuracy)) or np.any(accuracy <= 0.5) or np.any(accuracy > 1.0):
        raise ValueError("classical label accuracy must be in the interval (0.5, 1]")
    corrected_label = (y_reported + accuracy - 1.0) / (2.0 * accuracy - 1.0)
    return xi * corrected_label / pi


def _active_corrected_values(
    y_reported: np.ndarray,
    ai_label: np.ndarray,
    xi: np.ndarray,
    pi: np.ndarray,
    q_effort: float,
) -> np.ndarray:
    return ai_label + (y_reported - ai_label) * xi / (pi * q_effort)


def _robust_corrected_values(
    y_reported: np.ndarray,
    ai_label: np.ndarray,
    xi: np.ndarray,
    zeta: np.ndarray,
    pi: np.ndarray,
    rho: float,
    q_effort: float,
) -> np.ndarray:
    if not np.isfinite(rho) or rho <= 0.0 or rho >= 1.0:
        raise ValueError("rho must lie strictly between 0 and 1")
    return ai_label + (y_reported - ai_label) * (xi * zeta / (1.0 - rho)) / (pi * q_effort)


def _odds_ratio_interval(
    corrected_values: np.ndarray,
    phosphorylated: np.ndarray,
    corrected_variance: np.ndarray,
    confidence_level: float,
    boundary_counts: tuple[float, float] | None = None,
) -> tuple[float, float, float, float, float]:
    subgroup = _validate_binary_array("phosphorylated", phosphorylated)
    _validate_subgroups(subgroup)
    treated = subgroup.astype(bool)
    n_treated = int(np.sum(treated))
    n_untreated = int(np.sum(~treated))
    p_treated = float(np.mean(treated))
    p_untreated = 1.0 - p_treated
    numerator_values = treated.astype(float) * corrected_values / p_treated
    denominator_values = (~treated).astype(float) * corrected_values / p_untreated
    raw_numerator = float(np.mean(numerator_values))
    raw_denominator = float(np.mean(denominator_values))
    if boundary_counts is None:
        treated_boundary_count = float(n_treated)
        untreated_boundary_count = float(n_untreated)
    else:
        treated_boundary_count, untreated_boundary_count = boundary_counts
    numerator = _stabilized_group_probability(raw_numerator, treated_boundary_count)
    denominator = _stabilized_group_probability(raw_denominator, untreated_boundary_count)

    variance = np.asarray(corrected_variance, dtype=float)
    if variance.shape != corrected_values.shape:
        raise ValueError("corrected_variance must have the same shape as corrected_values")
    if np.any(~np.isfinite(variance)) or np.any(variance < 0.0):
        raise ValueError("corrected_variance must contain finite nonnegative values")

    n = corrected_values.size
    var_numerator = float(np.sum((treated.astype(float) / p_treated) ** 2 * variance) / n**2)
    var_denominator = float(np.sum(((~treated).astype(float) / p_untreated) ** 2 * variance) / n**2)
    gradient_numerator = 1.0 / numerator + 1.0 / (1.0 - numerator)
    gradient_denominator = -1.0 / denominator - 1.0 / (1.0 - denominator)
    log_variance = gradient_numerator**2 * var_numerator + gradient_denominator**2 * var_denominator
    if not np.isfinite(log_variance) or log_variance < 0.0:
        estimate = _odds_ratio_from_means(numerator, denominator)
        return estimate, -UNSTABLE_RATIO_RADIUS, UNSTABLE_RATIO_RADIUS, raw_numerator, raw_denominator

    estimate = _odds_ratio_from_means(numerator, denominator)
    radius = z_value(confidence_level) * float(np.sqrt(log_variance))
    log_estimate = float(np.log(estimate))
    lower = _capped_exp(log_estimate - radius)
    upper = _capped_exp(log_estimate + radius)
    return estimate, lower, upper, raw_numerator, raw_denominator


def _final_label_accuracy(
    y_true: np.ndarray,
    ai_label: np.ndarray,
    q_effort: float,
) -> np.ndarray:
    """Return final report accuracy under the simulated AI-error correction model."""

    _validate_effort_probability("q_effort", q_effort)
    y = np.asarray(y_true, dtype=float)
    f = np.asarray(ai_label, dtype=float)
    if y.shape != f.shape:
        raise ValueError("y_true and ai_label must have the same shape")
    return np.where(f != y, q_effort, 1.0).astype(float)


def _residual_conditional_variance(
    y_true: np.ndarray,
    ai_label: np.ndarray,
    pi: np.ndarray,
    q_effort: float,
    regular_probability: float,
) -> np.ndarray:
    y = np.asarray(y_true, dtype=float)
    f = np.asarray(ai_label, dtype=float)
    probabilities = _positive_probabilities("pi", pi)
    _validate_effort_probability("q_effort", q_effort)
    regular = float(regular_probability)
    if not np.isfinite(regular) or regular <= 0.0 or regular > 1.0:
        raise ValueError("regular_probability must lie in (0, 1]")
    if y.shape != f.shape or y.shape != probabilities.shape:
        raise ValueError("y_true, ai_label, and pi must have the same shape")
    residual_squared = (y - f) ** 2
    draw_probability = probabilities * regular * q_effort
    return residual_squared * (1.0 / draw_probability - 1.0)


def _classical_conditional_variance(
    y_true: np.ndarray,
    pi: np.ndarray,
    label_accuracy: np.ndarray,
) -> np.ndarray:
    y = _validate_binary_array("y_true", y_true)
    probabilities = _positive_probabilities("pi", pi)
    accuracy = np.asarray(label_accuracy, dtype=float)
    if y.shape != probabilities.shape or y.shape != accuracy.shape:
        raise ValueError("y_true, pi, and label_accuracy must have the same shape")
    if np.any(~np.isfinite(accuracy)) or np.any(accuracy <= 0.5) or np.any(accuracy > 1.0):
        raise ValueError("label_accuracy must be in the interval (0.5, 1]")

    denominator = (2.0 * accuracy - 1.0) ** 2
    second_moment_when_one = (accuracy**3 + (1.0 - accuracy) ** 3) / denominator
    second_moment_when_zero = accuracy * (1.0 - accuracy) / denominator
    label_second_moment = np.where(y == 1.0, second_moment_when_one, second_moment_when_zero)
    return label_second_moment / probabilities - y**2


def _effective_group_counts(phosphorylated: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    subgroup = _validate_binary_array("phosphorylated", phosphorylated).astype(bool)
    pi = _positive_probabilities("pi", probabilities)
    treated_count = float(np.sum(pi[subgroup]))
    untreated_count = float(np.sum(pi[~subgroup]))
    return max(treated_count, 1.0), max(untreated_count, 1.0)


def _odds_ratio_from_means(treated_mean: float, untreated_mean: float) -> float:
    _validate_probability_for_odds("treated mean", treated_mean)
    _validate_probability_for_odds("untreated mean", untreated_mean)
    treated_odds = treated_mean / (1.0 - treated_mean)
    untreated_odds = untreated_mean / (1.0 - untreated_mean)
    return float(treated_odds / untreated_odds)


def _stabilized_group_probability(value: float, effective_size: float) -> float:
    size = float(effective_size)
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("effective_size must be positive")
    boundary = max(ODDS_BOUNDARY_TOLERANCE, 0.5 / size)
    return float(np.clip(value, boundary, 1.0 - boundary))


def _capped_exp(value: float) -> float:
    log_cap = float(np.log(UNSTABLE_RATIO_RADIUS))
    return float(np.exp(np.clip(value, -log_cap, log_cap)))


def _validate_probability_for_odds(name: str, value: float) -> None:
    if not _is_interior_probability(value):
        raise ValueError(f"{name} must lie strictly between 0 and 1 for an odds ratio")


def _is_interior_probability(value: float) -> bool:
    value_float = float(value)
    return bool(
        np.isfinite(value_float)
        and ODDS_BOUNDARY_TOLERANCE < value_float < 1.0 - ODDS_BOUNDARY_TOLERANCE
    )


def _select_robust_design(
    tau: np.ndarray, budget: float, design: Mapping[str, Any]
) -> tuple[float, np.ndarray]:
    w0 = float(design["w0"])
    k = float(design["k"])
    q_effort = float(np.sqrt(w0))
    best: tuple[float, float, np.ndarray] | None = None
    for rho in design["rho_grid"]:
        rho_float = float(rho)
        pi = incentive_aware_probabilities(tau, budget=budget, rho=rho_float, w0=w0, k=k)
        if not np.any(pi > 0.0):
            continue
        denominator = (1.0 - rho_float) * pi * q_effort
        if np.any(denominator <= 0.0):
            continue
        criterion = float(np.mean(tau / denominator))
        if best is None or criterion < best[0]:
            best = (criterion, rho_float, pi)
    if best is None:
        raise ValueError("budget cannot cover the robust sentinel fixed cost for any rho")
    return best[1], best[2]


def _baseline_accuracy_cost(
    y_true: np.ndarray,
    ai_probability: np.ndarray,
    baseline_effort: float,
    w0: float,
    design: Mapping[str, Any],
) -> np.ndarray:
    model = str(design.get("baseline_payment_model", "linear_accuracy_based")).lower()
    effort = float(baseline_effort)
    if not np.isfinite(effort) or effort <= 0.0 or effort > 1.0:
        raise ValueError("baseline_effort must lie in (0, 1]")
    if model in {"none", "w0_only"}:
        return np.full(y_true.size, float(w0), dtype=float)
    if model in {"effort_squared", "sentinel_equivalent"}:
        return np.full(y_true.size, float(w0) + effort**2, dtype=float)
    if model != "linear_accuracy_based":
        raise ValueError(f"unknown baseline_payment_model {model!r}")

    error_floor = float(design.get("ai_error_floor", 0.02))
    if not np.isfinite(error_floor) or error_floor <= 0.0 or error_floor > 1.0:
        raise ValueError("ai_error_floor must lie in (0, 1]")
    y = np.asarray(y_true, dtype=float)
    probability = np.clip(np.asarray(ai_probability, dtype=float), 0.0, 1.0)
    p_error = np.where(y >= 0.5, 1.0 - probability, probability)
    p_error = np.clip(p_error, error_floor, 1.0)
    expected_accuracy_payment = effort * (1.0 - p_error * (1.0 - effort)) / p_error
    return np.asarray(float(w0) + expected_accuracy_payment, dtype=float)


def _mixed_cost_aware_weights(tau: np.ndarray, mix_tau: float, per_item_cost: np.ndarray) -> np.ndarray:
    mix = float(mix_tau)
    if not np.isfinite(mix) or mix < 0.0 or mix > 1.0:
        raise ValueError("baseline mix tau must lie in [0, 1]")
    tau_array = np.asarray(tau, dtype=float)
    if np.any(~np.isfinite(tau_array)) or np.any(tau_array < 0.0):
        raise ValueError("tau must contain finite nonnegative values")
    cost = np.asarray(per_item_cost, dtype=float)
    if np.any(~np.isfinite(cost)) or np.any(cost <= 0.0):
        raise ValueError("per-item costs must be finite and positive")
    mixed_tau = (1.0 - mix) * tau_array + mix * float(np.mean(tau_array))
    weights = np.sqrt(np.maximum(mixed_tau, 0.0) / cost)
    if not np.any(weights > 0.0):
        weights = np.ones_like(tau_array, dtype=float)
    return weights


def _design_config(config: Mapping[str, Any]) -> dict[str, Any]:
    incentive = _mapping(config.get("incentive", {}), "incentive")
    sampling = _mapping(config.get("sampling", {}), "sampling")
    baseline_mix = _mapping(sampling.get("baseline_mix", {}), "sampling.baseline_mix")
    w0 = float(incentive.get("w0", 0.64))
    k = float(incentive.get("k", 1.0))
    rho_grid = tuple(float(rho) for rho in incentive.get("rho_grid", (0.1, 0.2, 0.3, 0.4, 0.5)))
    if w0 <= 0.0 or not np.isfinite(w0):
        raise ValueError("incentive.w0 must be positive")
    if k < 0.0 or not np.isfinite(k):
        raise ValueError("incentive.k must be nonnegative")
    if not rho_grid:
        raise ValueError("incentive.rho_grid must contain at least one rho")
    for rho in rho_grid:
        if not np.isfinite(rho) or rho <= 0.0 or rho >= 1.0:
            raise ValueError("all rho_grid entries must lie strictly between 0 and 1")
    return {
        "w0": w0,
        "k": k,
        "rho_grid": rho_grid,
        "baseline_effort": float(config.get("baseline_effort", np.sqrt(w0))),
        "baseline_mix_tau": float(baseline_mix.get("tau", 0.5)),
        "baseline_payment_model": str(incentive.get("baseline_payment_model", "linear_accuracy_based")),
        "ai_error_floor": float(incentive.get("ai_error_floor", 0.02)),
    }


def _coerce_config(config: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], Path | None]:
    if isinstance(config, str | Path):
        config_path = Path(config)
        loaded = load_yaml_config(config_path)
        return loaded, config_path.resolve().parent
    config_map = dict(config)
    base = config_map.pop("_config_dir", None)
    return config_map, Path(base).resolve() if base is not None else None


def _resolve_path(path_value: Any, base_dir: Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    if base_dir is not None:
        return base_dir / path
    return path


def _tables_dir(config: Mapping[str, Any], base_dir: Path | None) -> Path:
    output = _mapping(config.get("output", config.get("outputs", {})), "output")
    path_value = output.get("tables_dir", "outputs/tables")
    return _resolve_path(path_value, base_dir)


def _seeds(config: Mapping[str, Any], smoke: bool) -> tuple[int, ...]:
    count_key = "smoke_seed_count" if smoke else "seed_count"
    if count_key in config:
        seed_count = int(config[count_key])
        if seed_count <= 0:
            raise ValueError(f"{count_key} must be a positive integer")
        return tuple(range(seed_count))

    key = "smoke_seeds" if smoke else "seeds"
    seeds = config.get(key, None)
    if seeds is None and smoke:
        seeds = list(config.get("seeds", [0]))[:2]
    if seeds is None:
        seeds = [0]
    seed_tuple = tuple(int(seed) for seed in seeds)
    if not seed_tuple:
        raise ValueError(f"{key} must contain at least one seed")
    return seed_tuple


def _budgets(config: Mapping[str, Any], smoke: bool) -> tuple[float, ...]:
    budget_grid = _mapping(config.get("budget_grid", {}), "budget_grid")
    key = "smoke" if smoke else "full"
    budgets = budget_grid.get(key, None)
    if budgets is None and smoke:
        budgets = list(budget_grid.get("full", [40.0]))[:2]
    if budgets is None:
        budgets = [40.0]
    budget_tuple = tuple(float(budget) for budget in budgets)
    if not budget_tuple or any(not np.isfinite(budget) or budget <= 0.0 for budget in budget_tuple):
        raise ValueError(f"budget_grid.{key} must contain positive budgets")
    return budget_tuple


def _confidence_level(config: Mapping[str, Any]) -> float:
    level = float(config.get("confidence_level", 0.90))
    if not np.isfinite(level) or level <= 0.0 or level >= 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    return level


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _as_1d_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_probability_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all((0.0 <= array) & (array <= 1.0)):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return array.astype(float, copy=False)


def _validate_binary_array(name: str, values: Any) -> np.ndarray:
    array = _validate_probability_array(name, values)
    if not np.all(np.isin(array, [0.0, 1.0])):
        raise ValueError(f"{name} must contain binary 0/1 values")
    return array.astype(float, copy=False)


def _validate_subgroups(phosphorylated: np.ndarray) -> None:
    if not np.any(phosphorylated == 1.0):
        raise ValueError("phosphorylated must include at least one phosphorylated protein")
    if not np.any(phosphorylated == 0.0):
        raise ValueError("phosphorylated must include at least one unphosphorylated protein")


def _validate_effort_probability(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError(f"{name} must lie in (0, 1]")


def _positive_probabilities(name: str, values: Any) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{name} must contain finite probabilities")
    if np.any(probabilities <= 0.0) or np.any(probabilities > 1.0):
        raise ValueError(f"{name} must contain probabilities in (0, 1]")
    return probabilities


def _method_seed(seed: int, method: str) -> int:
    return int(seed) * 1009 + METHODS.index(method) * 9173 + 12345


def _ordered_frame(frame: pd.DataFrame, leading_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    remaining = [column for column in frame.columns if column not in leading_columns]
    return frame[[*leading_columns, *remaining]]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AlphaFoldData",
    "ProteinDataset",
    "load_alphafold_npz",
    "prepare_protein_dataset",
    "protein_truth",
    "run_protein_experiment",
    "main",
]
