"""Shared experiment utilities for the paper experiments.

The helpers in this module keep experiment code focused on the estimator and
budget logic used throughout these experiments.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold


_TAU_FLOOR = 1e-6
_BASE_RESULT_COLUMNS = [
    "method",
    "budget",
    "estimate",
    "lower",
    "upper",
    "width",
    "covered",
    "seed",
]
_DEFAULT_RESULT_METADATA = {
    "budget_model": "main_text",
    "k_cost_interpretation": "fixed_global_sentinel_cost",
}


@dataclass(frozen=True)
class TrialResult:
    """One Monte Carlo interval result before aggregation."""

    method: str
    budget: float
    estimate: float
    lower: float
    upper: float
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetSummary:
    """Aggregated interval performance for one method-budget cell."""

    method: str
    budget: float
    estimate: float
    lower: float
    upper: float
    width: float
    covered: float
    seed: int | tuple[int, ...] | None = None
    n_trials: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedPredictionData:
    """Cross-fitted predictions and the derived active-sampling score."""

    y_true: np.ndarray
    predictions: np.ndarray
    tau: np.ndarray
    seeds: tuple[int, ...]
    model: str = "auto"
    task: str = "binary"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mean_prediction(self) -> np.ndarray:
        """Average prediction for each row across cross-fitting seeds."""

        return np.mean(self.predictions, axis=0)

    def to_frame(self) -> pd.DataFrame:
        """Return long-form predictions with one row per seed-observation pair."""

        rows = []
        metadata = _merge_metadata(self.metadata)
        for seed_index, seed in enumerate(self.seeds):
            for row_index, prediction in enumerate(self.predictions[seed_index]):
                row = {
                    "seed": seed,
                    "row": row_index,
                    "prediction": float(prediction),
                    "tau": float(self.tau[row_index]),
                    "y_true": float(self.y_true[row_index]),
                    "model": self.model,
                    "task": self.task,
                }
                row.update(metadata)
                rows.append(row)
        return pd.DataFrame(rows)


@dataclass
class _ConstantPredictionModel:
    value: float
    task: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ConstantPredictionModel":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_rows = np.asarray(X).shape[0]
        if self.task == "binary":
            return np.full(n_rows, float(self.value >= 0.5), dtype=float)
        return np.full(n_rows, self.value, dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "binary":
            raise AttributeError("regression constant model does not expose predict_proba")
        n_rows = np.asarray(X).shape[0]
        positive = np.full(n_rows, np.clip(self.value, 0.0, 1.0), dtype=float)
        return np.column_stack((1.0 - positive, positive))

    @property
    def classes_(self) -> np.ndarray:
        return np.array([0.0, 1.0])


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path`` using ``yaml.safe_load``."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError("YAML config must contain a mapping at the top level")
    return dict(loaded)


def make_rng(seed: int | np.integer | np.random.Generator | None) -> np.random.Generator:
    """Return a NumPy random generator for deterministic seeded experiments."""

    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def fit_prediction_model(
    X_train: Any,
    y_train: Any,
    model: str | Any = "auto",
    random_state: int = 0,
) -> Any:
    """Fit a prediction model for binary or continuous outcomes.

    ``model="auto"`` uses XGBoost when it is importable and otherwise uses
    scikit-learn's histogram gradient boosting estimators. Degenerate one-class
    binary folds and tiny regression folds return deterministic constant models.
    """

    X_array = _as_2d_array("X_train", X_train)
    y_array = _as_1d_array("y_train", y_train, expected_length=X_array.shape[0])
    task = "binary" if _is_binary_target(y_array) else "regression"

    if task == "binary" and np.unique(y_array).size < 2:
        return _ConstantPredictionModel(float(y_array[0]), task="binary")
    if task == "regression" and y_array.size < 2:
        return _ConstantPredictionModel(float(np.mean(y_array)), task="regression")

    estimator = _make_estimator(task, model, random_state)
    fitted = estimator.fit(X_array, y_array)
    _set_backend_metadata(fitted, task, model)
    return fitted


def crossfit_binary_predictions(
    X: Any,
    y: Any,
    seeds: Iterable[int],
    model: str | Any = "auto",
) -> PreparedPredictionData:
    """Return cross-fitted probabilities and squared-residual ``tau`` scores."""

    X_array = _as_2d_array("X", X)
    y_array = _as_binary_array("y", y, expected_length=X_array.shape[0])
    seed_tuple = _normalize_seeds(seeds)
    predictions = []

    for seed in seed_tuple:
        out_of_fold = np.empty(y_array.size, dtype=float)
        for train_idx, test_idx in _classification_splitter(y_array, seed).split(X_array, y_array):
            fitted = fit_prediction_model(
                X_array[train_idx],
                y_array[train_idx],
                model=model,
                random_state=seed,
            )
            out_of_fold[test_idx] = _predict_positive_probability(fitted, X_array[test_idx])
        predictions.append(np.clip(out_of_fold, 0.0, 1.0))

    prediction_matrix = np.vstack(predictions)
    mean_prediction = np.mean(prediction_matrix, axis=0)
    tau = tau_from_binary_predictions(y_array, mean_prediction)
    metadata = {
        "prediction_task": "binary",
        "crossfit_folds": _classification_n_splits(y_array),
        "tau_definition": "squared_residual_floor",
    }
    return PreparedPredictionData(
        y_true=y_array,
        predictions=prediction_matrix,
        tau=tau,
        seeds=seed_tuple,
        model=_model_name(model),
        task="binary",
        metadata=metadata,
    )


def crossfit_regression_predictions(
    X: Any,
    y: Any,
    seeds: Iterable[int],
    model: str | Any = "auto",
) -> PreparedPredictionData:
    """Return cross-fitted regression predictions and residual ``tau`` scores."""

    X_array = _as_2d_array("X", X)
    y_array = _as_1d_array("y", y, expected_length=X_array.shape[0])
    seed_tuple = _normalize_seeds(seeds)
    predictions = []

    for seed in seed_tuple:
        out_of_fold = np.empty(y_array.size, dtype=float)
        splitter = _regression_splitter(y_array, seed)
        for train_idx, test_idx in splitter.split(X_array):
            fitted = fit_prediction_model(
                X_array[train_idx],
                y_array[train_idx],
                model=model,
                random_state=seed,
            )
            out_of_fold[test_idx] = _predict_regression(fitted, X_array[test_idx])
        predictions.append(out_of_fold)

    prediction_matrix = np.vstack(predictions)
    mean_prediction = np.mean(prediction_matrix, axis=0)
    tau = tau_from_regression_residuals(y_array, mean_prediction)
    metadata = {
        "prediction_task": "regression",
        "crossfit_folds": _regression_n_splits(y_array),
        "tau_definition": "squared_residual_floor",
    }
    return PreparedPredictionData(
        y_true=y_array,
        predictions=prediction_matrix,
        tau=tau,
        seeds=seed_tuple,
        model=_model_name(model),
        task="regression",
        metadata=metadata,
    )


def tau_from_binary_predictions(
    y_true: Any, y_prob: Any, floor: float = _TAU_FLOOR
) -> np.ndarray:
    """Estimate ``tau_i`` for binary outcomes as floored squared residuals."""

    y_array = _as_binary_array("y_true", y_true)
    prob_array = np.asarray(y_prob, dtype=float)
    if prob_array.ndim == 2 and prob_array.shape[1] == 2:
        prob_array = prob_array[:, 1]
    prob_array = _as_1d_array("y_prob", prob_array, expected_length=y_array.size)
    if np.any((prob_array < 0.0) | (prob_array > 1.0)):
        raise ValueError("y_prob must contain probabilities in [0, 1]")
    floor = _validate_positive_floor(floor)
    return np.maximum((y_array - prob_array) ** 2, floor)


def tau_from_regression_residuals(
    y_true: Any, y_pred: Any, floor: float = _TAU_FLOOR
) -> np.ndarray:
    """Estimate ``tau_i`` for continuous outcomes as floored squared residuals."""

    y_array = _as_1d_array("y_true", y_true)
    pred_array = _as_1d_array("y_pred", y_pred, expected_length=y_array.size)
    floor = _validate_positive_floor(floor)
    return np.maximum((y_array - pred_array) ** 2, floor)


def summarize_intervals(
    trial_results: Iterable[TrialResult | Mapping[str, Any]],
    truth: float | Mapping[str, float],
) -> pd.DataFrame:
    """Aggregate trial intervals by method, budget, and metadata fields."""

    records = [_trial_record(result, truth) for result in trial_results]
    if not records:
        return pd.DataFrame(columns=[*_BASE_RESULT_COLUMNS, "n_trials"])

    raw = pd.DataFrame.from_records(records)
    metadata_columns = [
        column for column in raw.columns if column not in _BASE_RESULT_COLUMNS
    ]
    group_columns = ["method", "budget", *metadata_columns]

    summary = (
        raw.groupby(group_columns, dropna=False, sort=True)
        .agg(
            estimate=("estimate", "mean"),
            lower=("lower", "mean"),
            upper=("upper", "mean"),
            width=("width", "mean"),
            covered=("covered", "mean"),
            seed=("seed", _combine_seeds),
            n_trials=("covered", "size"),
        )
        .reset_index()
    )
    ordered_columns = [
        *_BASE_RESULT_COLUMNS,
        *metadata_columns,
        "n_trials",
    ]
    return summary[ordered_columns]


def budget_savings_curve(
    widths_by_method: Mapping[str, Any] | pd.DataFrame,
    budgets: Any,
    reference_method: str,
    target_method: str = "robust",
) -> pd.DataFrame:
    """Interpolate the target budget needed to match reference widths.

    The width curves are first converted to monotone nonincreasing frontiers, so
    small Monte Carlo noise cannot make the inverted budget curve nonmonotone.
    ``estimate`` is the saved-budget fraction at each reference budget.
    """

    budget_array = _as_1d_array("budgets", budgets)
    if np.any(budget_array <= 0.0):
        raise ValueError("budgets must be positive for savings fractions")
    width_map = _coerce_width_map(widths_by_method, budget_array)
    if reference_method not in width_map:
        raise KeyError(f"reference method {reference_method!r} is missing")
    if target_method not in width_map:
        raise KeyError(f"target method {target_method!r} is missing")

    order = np.argsort(budget_array)
    sorted_budgets = budget_array[order]
    reference_widths = _monotone_width_frontier(width_map[reference_method][order])
    target_widths = _monotone_width_frontier(width_map[target_method][order])
    curve_name = f"{target_method}_vs_{reference_method}"
    metadata = _merge_metadata(
        {
            "reference_method": reference_method,
            "target_method": target_method,
            "width_frontier": "cumulative_minimum",
        }
    )

    rows = []
    for reference_budget, target_width in zip(sorted_budgets, reference_widths):
        target_budget = _budget_to_match_width(
            sorted_budgets, target_widths, target_width
        )
        attainable = bool(np.isfinite(target_budget))
        saved_budget = reference_budget - target_budget if attainable else np.nan
        savings_fraction = saved_budget / reference_budget if attainable else np.nan
        row = {
            "method": curve_name,
            "budget": float(reference_budget),
            "estimate": float(savings_fraction) if attainable else np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "width": float(target_width),
            "covered": np.nan,
            "seed": np.nan,
            "reference_budget": float(reference_budget),
            "target_budget": float(target_budget) if attainable else np.nan,
            "target_width": float(target_width),
            "saved_budget": float(saved_budget) if attainable else np.nan,
            "savings_fraction": float(savings_fraction) if attainable else np.nan,
            "savings_percent": float(100.0 * savings_fraction) if attainable else np.nan,
            "attainable": attainable,
        }
        row.update(metadata)
        rows.append(row)

    frame = pd.DataFrame(rows)
    metadata_columns = [
        column for column in frame.columns if column not in _BASE_RESULT_COLUMNS
    ]
    return frame[[*_BASE_RESULT_COLUMNS, *metadata_columns]]


def _as_2d_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_1d_array(
    name: str, values: Any, expected_length: int | None = None
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if expected_length is not None and array.size != expected_length:
        raise ValueError(f"{name} must have length {expected_length}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_binary_array(
    name: str, values: Any, expected_length: int | None = None
) -> np.ndarray:
    array = _as_1d_array(name, values, expected_length=expected_length)
    if not np.all(np.isin(array, [0.0, 1.0])):
        raise ValueError(f"{name} must contain binary 0/1 values")
    return array.astype(float, copy=False)


def _is_binary_target(y: np.ndarray) -> bool:
    unique = np.unique(y)
    return unique.size <= 2 and np.all(np.isin(unique, [0.0, 1.0]))


def _normalize_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    seed_tuple = tuple(int(seed) for seed in seeds)
    if not seed_tuple:
        raise ValueError("seeds must contain at least one seed")
    return seed_tuple


def _classification_n_splits(y: np.ndarray) -> int:
    if y.size < 2:
        raise ValueError("cross-fitting requires at least two observations")
    _, counts = np.unique(y, return_counts=True)
    if counts.size == 2 and np.min(counts) >= 2:
        return int(min(5, np.min(counts), y.size))
    return int(min(5, y.size))


def _classification_splitter(y: np.ndarray, seed: int) -> KFold | StratifiedKFold:
    n_splits = _classification_n_splits(y)
    if n_splits < 2:
        raise ValueError("cross-fitting requires at least two folds")
    _, counts = np.unique(y, return_counts=True)
    if counts.size == 2 and np.min(counts) >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _regression_n_splits(y: np.ndarray) -> int:
    if y.size < 2:
        raise ValueError("cross-fitting requires at least two observations")
    return int(min(5, y.size))


def _regression_splitter(y: np.ndarray, seed: int) -> KFold:
    n_splits = _regression_n_splits(y)
    if n_splits < 2:
        raise ValueError("cross-fitting requires at least two folds")
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _make_estimator(task: str, model: str | Any, random_state: int) -> Any:
    if not isinstance(model, str):
        estimator = clone(model)
        if hasattr(estimator, "get_params") and "random_state" in estimator.get_params():
            estimator.set_params(random_state=random_state)
        return estimator

    model_key = model.lower()
    if model_key in {"auto", "xgboost", "xgb"}:
        xgb_module = _import_xgboost()
        if xgb_module is not None:
            if task == "binary":
                return xgb_module.XGBClassifier(
                    n_estimators=80,
                    max_depth=3,
                    learning_rate=0.08,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=1,
                    verbosity=0,
                )
            return xgb_module.XGBRegressor(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.08,
                subsample=1.0,
                colsample_bytree=1.0,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
            )
        if model_key in {"xgboost", "xgb"}:
            raise ImportError("xgboost is not installed")

    if model_key in {"auto", "sklearn", "hist_gradient_boosting", "hgb"}:
        if task == "binary":
            return HistGradientBoostingClassifier(
                random_state=random_state,
                max_iter=80,
                learning_rate=0.08,
                max_leaf_nodes=15,
                min_samples_leaf=2,
                early_stopping=False,
            )
        return HistGradientBoostingRegressor(
            random_state=random_state,
            max_iter=80,
            learning_rate=0.08,
            max_leaf_nodes=15,
            min_samples_leaf=2,
            early_stopping=False,
        )

    raise ValueError(f"unknown prediction model {model!r}")


def _import_xgboost() -> Any | None:
    try:
        import xgboost as xgb
    except ImportError:
        return None
    return xgb


def _set_backend_metadata(fitted: Any, task: str, model: str | Any) -> None:
    try:
        fitted._experiment_common_task = task
        fitted._experiment_common_model = _model_name(model)
    except Exception:
        pass


def _model_name(model: str | Any) -> str:
    if isinstance(model, str):
        return model
    return model.__class__.__name__


def _predict_positive_probability(fitted: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(fitted, "predict_proba"):
        probabilities = np.asarray(fitted.predict_proba(X), dtype=float)
        if probabilities.ndim == 1:
            return np.clip(probabilities, 0.0, 1.0)
        if probabilities.ndim != 2:
            raise ValueError("predict_proba must return a one- or two-dimensional array")
        if probabilities.shape[1] == 1:
            return np.clip(probabilities[:, 0], 0.0, 1.0)
        classes = getattr(fitted, "classes_", None)
        if classes is not None and 1 in classes:
            class_index = int(np.flatnonzero(np.asarray(classes) == 1)[0])
        else:
            class_index = probabilities.shape[1] - 1
        return np.clip(probabilities[:, class_index], 0.0, 1.0)

    predictions = np.asarray(fitted.predict(X), dtype=float)
    return np.clip(predictions.reshape(-1), 0.0, 1.0)


def _predict_regression(fitted: Any, X: np.ndarray) -> np.ndarray:
    return np.asarray(fitted.predict(X), dtype=float).reshape(-1)


def _validate_positive_floor(floor: float) -> float:
    floor = float(floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("floor must be a finite positive value")
    return floor


def _trial_record(
    result: TrialResult | Mapping[str, Any],
    truth: float | Mapping[str, float],
) -> dict[str, Any]:
    if isinstance(result, Mapping):
        data = dict(result)
    elif is_dataclass(result):
        data = asdict(result)
    else:
        data = dict(vars(result))

    metadata = dict(data.pop("metadata", {}) or {})
    extra_metadata = {
        key: data.pop(key)
        for key in list(data.keys())
        if key not in {"method", "budget", "estimate", "lower", "upper", "seed"}
    }
    metadata.update(extra_metadata)
    metadata = _merge_metadata(metadata)

    method = str(data["method"])
    lower = float(data["lower"])
    upper = float(data["upper"])
    if lower > upper:
        raise ValueError("interval lower endpoint must not exceed upper endpoint")
    truth_value = _truth_for_method(truth, method)

    record = {
        "method": method,
        "budget": float(data["budget"]),
        "estimate": float(data["estimate"]),
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
        "covered": lower <= truth_value <= upper,
        "seed": data.get("seed", np.nan),
    }
    record.update({key: _stable_metadata_value(value) for key, value in metadata.items()})
    return record


def _truth_for_method(truth: float | Mapping[str, float], method: str) -> float:
    if isinstance(truth, Mapping):
        if method not in truth:
            raise KeyError(f"truth mapping is missing method {method!r}")
        truth_value = float(truth[method])
    else:
        truth_value = float(truth)
    if not np.isfinite(truth_value):
        raise ValueError("truth must be finite")
    return truth_value


def _merge_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(_DEFAULT_RESULT_METADATA)
    if metadata:
        merged.update(dict(metadata))
    return merged


def _stable_metadata_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if isinstance(value, Mapping):
        return tuple(
            sorted((key, _stable_metadata_value(val)) for key, val in value.items())
        )
    if isinstance(value, list | tuple):
        return tuple(_stable_metadata_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_stable_metadata_value(item) for item in value))
    return value


def _combine_seeds(values: pd.Series) -> int | tuple[int, ...] | float:
    unique: list[int] = []
    for value in values:
        if pd.isna(value):
            continue
        seed = int(value)
        if seed not in unique:
            unique.append(seed)
    if not unique:
        return np.nan
    if len(unique) == 1:
        return unique[0]
    return tuple(unique)


def _coerce_width_map(
    widths_by_method: Mapping[str, Any] | pd.DataFrame, budgets: np.ndarray
) -> dict[str, np.ndarray]:
    if isinstance(widths_by_method, pd.DataFrame):
        required = {"method", "budget", "width"}
        if not required.issubset(widths_by_method.columns):
            raise ValueError("width DataFrame must contain method, budget, and width columns")
        width_map = {}
        for method, group in widths_by_method.groupby("method", sort=False):
            ordered = group.set_index("budget").reindex(budgets)
            if ordered["width"].isna().any():
                raise ValueError(f"widths are missing for method {method!r}")
            width_map[str(method)] = ordered["width"].to_numpy(dtype=float)
        return width_map

    width_map = {}
    for method, widths in widths_by_method.items():
        width_array = _as_1d_array(f"widths for {method}", widths)
        if width_array.size != budgets.size:
            raise ValueError(f"widths for {method!r} must match budgets length")
        if np.any(width_array <= 0.0):
            raise ValueError("interval widths must be positive")
        width_map[str(method)] = width_array
    return width_map


def _monotone_width_frontier(widths: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(widths.astype(float, copy=True))


def _budget_to_match_width(
    budgets: np.ndarray, target_widths: np.ndarray, desired_width: float
) -> float:
    if desired_width >= target_widths[0]:
        return float(budgets[0])
    if desired_width < target_widths[-1]:
        return float("nan")

    matches = np.flatnonzero(target_widths <= desired_width)
    if matches.size == 0:
        return float("nan")
    idx = int(matches[0])
    if idx == 0:
        return float(budgets[0])

    width_left = float(target_widths[idx - 1])
    width_right = float(target_widths[idx])
    budget_left = float(budgets[idx - 1])
    budget_right = float(budgets[idx])
    if width_left == width_right:
        return budget_right
    fraction = (width_left - desired_width) / (width_left - width_right)
    return float(budget_left + fraction * (budget_right - budget_left))


__all__ = [
    "TrialResult",
    "BudgetSummary",
    "PreparedPredictionData",
    "load_yaml_config",
    "make_rng",
    "fit_prediction_model",
    "crossfit_binary_predictions",
    "crossfit_regression_predictions",
    "tau_from_binary_predictions",
    "tau_from_regression_residuals",
    "summarize_intervals",
    "budget_savings_curve",
]
