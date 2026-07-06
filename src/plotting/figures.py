"""Generate paper-style figures from local experiment CSV outputs.

The functions in this module intentionally consume the generated tables under
``outputs/tables``. They do not hard-code experiment outputs; plot annotations
such as nominal coverage are taken from experiment metadata when present.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Iterable
import warnings

_DEFAULT_MPLCONFIGDIR = Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"
_DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_DEFAULT_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_TABLES_DIR = Path("outputs") / "tables"
_METHOD_ORDER = ("classical", "uniform", "active", "robust")
_METHOD_LABELS = {
    "classical": "Classical",
    "uniform": "Uniform",
    "active": "Active",
    "robust": "Robust",
}
_METHOD_COLORS = {
    "classical": "#6B7280",
    "uniform": "#2563EB",
    "active": "#059669",
    "robust": "#DC2626",
}
_METHOD_MARKERS = {
    "classical": "o",
    "uniform": "s",
    "active": "^",
    "robust": "D",
}
_MAX_INTERVAL_TRIALS_PER_METHOD = 50


@dataclass(frozen=True)
class _RepresentativeSelection:
    frame: pd.DataFrame
    budget: float
    total_trials_by_method: dict[str, int]


def plot_interval_width_coverage(
    summary_csv: str | Path,
    trials_csv: str | Path,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot trial-level intervals, average width, and empirical coverage.

    Parameters
    ----------
    summary_csv:
        CSV with one row per method-budget cell and columns including
        ``method``, ``budget``, ``width``, and ``covered``.
    trials_csv:
        CSV with trial-level intervals and columns including ``method``,
        ``budget``, ``seed``, ``estimate``, ``lower``, and ``upper``.
    output_path:
        Destination file. The extension controls the Matplotlib backend output
        format, e.g. ``.pdf`` or ``.png``.
    title:
        Figure title.
    """

    summary_path = Path(summary_csv)
    trials_path = Path(trials_csv)
    summary = _prepare_summary(_read_csv(summary_path, "summary"))
    trials = _prepare_trials(_read_csv(trials_path, "trials"))
    if summary.empty:
        raise ValueError(f"{summary_path} has no plottable summary rows")
    if trials.empty:
        raise ValueError(f"{trials_path} has no plottable trial rows")

    methods = _ordered_methods([*summary["method"].unique(), *trials["method"].unique()])
    representative = _select_representative_trials(trials, methods)
    nominal = _nominal_coverage(summary, trials)
    truth = _first_finite_value((trials, summary), "truth")

    _set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
    fig.suptitle(title, fontsize=12, fontweight="semibold")

    _plot_representative_intervals(axes[0], representative, methods, truth)
    _plot_widths(axes[1], summary, methods)
    _plot_coverage(axes[2], summary, methods, nominal)
    _add_metadata_note(fig, summary)

    return _save_figure(fig, output_path)


def plot_budget_savings(
    summary_csv: str | Path,
    output_path: str | Path,
    robust_method: str = "robust",
) -> Path:
    """Plot percent budget saved by the robust method at matched CI widths.

    For each baseline budget, the corresponding baseline interval width is used
    as the target. The robust budget required to reach that width is linearly
    interpolated from the robust summary curve. Targets outside the observed
    robust width range use the usual square-root CI-width scaling fallback,
    which avoids turning the smallest observed budget into a mechanical floor.
    """

    summary_path = Path(summary_csv)
    summary = _prepare_summary(_read_csv(summary_path, "summary"))
    if summary.empty:
        raise ValueError(f"{summary_path} has no plottable summary rows")

    robust_key = _normalize_method(robust_method)
    if robust_key not in set(summary["method"]):
        raise ValueError(f"{summary_path} does not contain robust method {robust_method!r}")

    methods = [method for method in _ordered_methods(summary["method"].unique()) if method != robust_key]
    robust_curve = _curve_for_method(summary, robust_key)
    if robust_curve.empty:
        raise ValueError(f"{summary_path} does not contain finite widths for {robust_method!r}")

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.4), constrained_layout=True)
    plotted = False
    plotted_budgets: list[np.ndarray] = []
    for method in methods:
        baseline_curve = _curve_for_method(summary, method)
        if baseline_curve.empty:
            continue
        points = _budget_savings_points(
            baseline_curve["budget"].to_numpy(dtype=float),
            baseline_curve["width"].to_numpy(dtype=float),
            robust_curve["budget"].to_numpy(dtype=float),
            robust_curve["width"].to_numpy(dtype=float),
        )
        if points.size == 0:
            continue
        ax.plot(
            points[:, 0],
            points[:, 1],
            marker=_METHOD_MARKERS.get(method, "o"),
            linewidth=2.0,
            markersize=5.0,
            color=_METHOD_COLORS.get(method, "#111827"),
            label=f"vs. {_method_label(method)}",
        )
        plotted = True
        plotted_budgets.append(points[:, 0])

    ax.axhline(0.0, color="#6B7280", linewidth=1.0, linestyle="--")
    ax.set_title(f"Budget Saved by {_method_label(robust_key)}")
    ax.set_xlabel("Baseline budget")
    ax.set_ylabel("Budget saved (%)")
    budget_axis_values = (
        np.concatenate(plotted_budgets)
        if plotted_budgets
        else summary["budget"].to_numpy(dtype=float)
    )
    _maybe_set_log_budget_axis(ax, budget_axis_values)
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    if plotted:
        ax.legend(frameon=False, fontsize=8, loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "No overlapping interval widths",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#6B7280",
        )

    return _save_figure(fig, output_path)


def plot_all_figures(output_dir: str | Path = "outputs/figures") -> list[Path]:
    """Generate all known figures whose required CSV inputs are available."""

    out_dir = Path(output_dir)
    generated: list[Path] = []

    performance_specs = [
        (
            "Pew Wave 79 Biden approval",
            _TABLES_DIR / "pew_biden_summary.csv",
            _TABLES_DIR / "pew_biden_trials.csv",
            out_dir / "widths_and_coverage_pew79_biden_robust_uniform_payment.pdf",
        ),
        (
            "Pew Wave 79 Trump approval",
            _TABLES_DIR / "pew_trump_summary.csv",
            _TABLES_DIR / "pew_trump_trials.csv",
            out_dir / "widths_and_coverage_pew79_trump_robust_uniform_payment.pdf",
        ),
        (
            "AlphaFold odds ratio",
            _TABLES_DIR / "protein_summary.csv",
            _TABLES_DIR / "protein_trials.csv",
            out_dir / "widths_and_coverage_alphafold_robust_uniform_payment.pdf",
        ),
        (
            "Pew Biden misspecification sensitivity",
            _TABLES_DIR / "robustness_misspecification_summary.csv",
            _TABLES_DIR / "robustness_misspecification_trials.csv",
            out_dir / "fig1.png",
        ),
        (
            "ACS PUMS age coefficient",
            _TABLES_DIR / "acs_summary.csv",
            _TABLES_DIR / "acs_trials.csv",
            out_dir / "fig2.png",
        ),
        (
            "Pew Biden posterior belief sensitivity",
            _TABLES_DIR / "robustness_posterior_summary.csv",
            _TABLES_DIR / "robustness_posterior_trials.csv",
            out_dir / "fig3.png",
        ),
    ]

    for title, summary_csv, trials_csv, output_path in performance_specs:
        if _warn_if_missing(f"{title} performance figure", (summary_csv, trials_csv)):
            continue
        generated.append(plot_interval_width_coverage(summary_csv, trials_csv, output_path, title))

    budget_specs = [
        (
            "Pew Wave 79 Biden budget savings",
            _TABLES_DIR / "pew_biden_summary.csv",
            out_dir / "budget_pew79_biden_robust_uniform_payment.pdf",
        ),
        (
            "Pew Wave 79 Trump budget savings",
            _TABLES_DIR / "pew_trump_summary.csv",
            out_dir / "budget_pew79_trump_robust_uniform_payment.pdf",
        ),
        (
            "AlphaFold budget savings",
            _TABLES_DIR / "protein_summary.csv",
            out_dir / "budget_alphafold_robust_uniform_payment.pdf",
        ),
    ]

    for label, summary_csv, output_path in budget_specs:
        if _warn_if_missing(label, (summary_csv,)):
            continue
        generated.append(plot_budget_savings(summary_csv, output_path))

    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate figures from local experiment CSV outputs.")
    parser.add_argument("--all", action="store_true", help="Generate all known figures with available inputs.")
    parser.add_argument("--output-dir", default="outputs/figures", help="Directory for generated figures.")
    args = parser.parse_args(argv)

    if not args.all:
        parser.error("pass --all to generate the full figure set")

    generated = plot_all_figures(args.output_dir)
    if generated:
        for path in generated:
            print(f"Wrote {path}")
    else:
        print("No figures were written because no expected CSV inputs were available.", file=sys.stderr)
    return 0


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV does not exist: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{label} CSV is empty: {path}") from exc


def _prepare_summary(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "budget", "covered"}
    _require_columns(frame, required, "summary")
    prepared = frame.copy()
    prepared["method"] = prepared["method"].map(_normalize_method)
    prepared["budget"] = _numeric_series(prepared["budget"])
    if "width" not in prepared.columns:
        _require_columns(prepared, {"lower", "upper"}, "summary")
        prepared["width"] = _numeric_series(prepared["upper"]) - _numeric_series(prepared["lower"])
    else:
        prepared["width"] = _numeric_series(prepared["width"])
    prepared["covered"] = _covered_series(prepared["covered"])
    for column in ("confidence_level", "truth"):
        if column in prepared.columns:
            prepared[column] = _numeric_series(prepared[column])

    subset = ["method", "budget", "width", "covered"]
    prepared = prepared.dropna(subset=subset)
    prepared = prepared[np.isfinite(prepared["budget"]) & np.isfinite(prepared["width"])]
    if prepared.empty:
        return prepared

    aggregations: dict[str, Any] = {"width": "mean", "covered": "mean"}
    for column in prepared.columns:
        if column in {"method", "budget", "width", "covered"}:
            continue
        aggregations[column] = "first"
    return (
        prepared.groupby(["method", "budget"], as_index=False, sort=True, dropna=False)
        .agg(aggregations)
        .sort_values(["method", "budget"])
        .reset_index(drop=True)
    )


def _prepare_trials(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "budget", "estimate", "lower", "upper"}
    _require_columns(frame, required, "trials")
    prepared = frame.copy()
    prepared["method"] = prepared["method"].map(_normalize_method)
    for column in ("budget", "estimate", "lower", "upper"):
        prepared[column] = _numeric_series(prepared[column])
    if "seed" not in prepared.columns:
        prepared["seed"] = np.nan
    if "truth" in prepared.columns:
        prepared["truth"] = _numeric_series(prepared["truth"])
    if "covered" in prepared.columns:
        prepared["covered"] = _covered_series(prepared["covered"])
    prepared = prepared.dropna(subset=["method", "budget", "estimate", "lower", "upper"])
    finite = np.isfinite(prepared[["budget", "estimate", "lower", "upper"]].to_numpy(dtype=float)).all(axis=1)
    return prepared.loc[finite].reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} CSV is missing required columns: {missing}")


def _normalize_method(value: Any) -> str:
    return str(value).strip().lower()


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _covered_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    mapped = series.astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(mapped)


def _ordered_methods(values: Iterable[Any]) -> list[str]:
    normalized = {_normalize_method(value) for value in values if pd.notna(value)}
    ordered = [method for method in _METHOD_ORDER if method in normalized]
    ordered.extend(sorted(normalized - set(ordered)))
    return ordered


def _select_representative_trials(trials: pd.DataFrame, methods: list[str]) -> _RepresentativeSelection:
    budgets = np.array(sorted(trials["budget"].dropna().unique()), dtype=float)
    if budgets.size == 0:
        raise ValueError("trials CSV has no finite budgets")
    median_budget = float(np.median(budgets))

    budget_scores: list[tuple[int, float, float]] = []
    for budget in budgets:
        frame = trials[np.isclose(trials["budget"], budget)]
        n_methods = frame["method"].nunique()
        distance = abs(float(budget) - median_budget)
        budget_scores.append((-n_methods, distance, float(budget)))
    selected_budget = min(budget_scores)[2]
    at_budget = trials[np.isclose(trials["budget"], selected_budget)].copy()
    total_trials_by_method = at_budget.groupby("method", dropna=False).size().to_dict()

    if at_budget["seed"].notna().any():
        seed_scores = (
            at_budget.groupby("seed", dropna=False)["method"]
            .nunique()
            .reset_index(name="n_methods")
        )
        seed_scores["_seed_sort"] = seed_scores["seed"].map(_seed_sort_key)
        seed_scores = seed_scores.sort_values(["n_methods", "_seed_sort"], ascending=[False, True])
        selected_seeds = seed_scores["seed"].head(_MAX_INTERVAL_TRIALS_PER_METHOD).tolist()
        selected = at_budget[at_budget["seed"].isin(selected_seeds)].copy()
        selected["_seed_rank"] = selected["seed"].map({seed: index for index, seed in enumerate(selected_seeds)})
        selected = selected.drop_duplicates(subset=["method", "seed"], keep="first")
    else:
        selected = (
            at_budget.sort_values(["method", "estimate", "lower", "upper"])
            .groupby("method", as_index=False, sort=False, dropna=False)
            .head(_MAX_INTERVAL_TRIALS_PER_METHOD)
            .copy()
        )
        selected["_seed_rank"] = selected.groupby("method", sort=False).cumcount()

    method_rank = {method: index for index, method in enumerate(methods)}
    selected["_method_rank"] = selected["method"].map(method_rank).fillna(len(method_rank))
    selected = selected.sort_values(["_method_rank", "_seed_rank"]).drop(columns=["_method_rank", "_seed_rank"])
    return _RepresentativeSelection(
        frame=selected.reset_index(drop=True),
        budget=selected_budget,
        total_trials_by_method={str(key): int(value) for key, value in total_trials_by_method.items()},
    )


def _plot_representative_intervals(
    ax: plt.Axes,
    selection: _RepresentativeSelection,
    methods: list[str],
    truth: float | None,
) -> None:
    frame = selection.frame
    plotted_methods = [method for method in methods if method in set(frame["method"])]
    if not plotted_methods:
        ax.text(0.5, 0.5, "No trial intervals", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    max_trials = max(len(frame[frame["method"] == method]) for method in plotted_methods)
    group_gap = max(2.0, 0.18 * max_trials)
    y_ticks: list[float] = []
    y_labels: list[str] = []
    all_y: list[float] = []

    for method_index, method in enumerate(plotted_methods):
        method_frame = frame[frame["method"] == method].reset_index(drop=True)
        group_start = method_index * (max_trials + group_gap)
        color = _METHOD_COLORS.get(method, "#111827")
        y_values = group_start + np.arange(len(method_frame))
        y_ticks.append(float(group_start + (len(method_frame) - 1) / 2.0))
        y_labels.append(_trial_group_label(method, method_frame, truth))
        all_y.extend(float(value) for value in y_values)

        for y_position, (_, row) in zip(y_values, method_frame.iterrows(), strict=False):
            estimate = float(row["estimate"])
            lower = float(min(row["lower"], row["upper"]))
            upper = float(max(row["lower"], row["upper"]))
            covered = _trial_covers_truth(row, truth)
            alpha = 0.32 if covered else 0.95
            linewidth = 0.75 if covered else 1.25
            marker = "." if covered else "x"
            markersize = 2.4 if covered else 3.6
            ax.hlines(y_position, lower, upper, color=color, alpha=alpha, linewidth=linewidth)
            ax.plot(estimate, y_position, marker=marker, color=color, alpha=alpha, markersize=markersize)

    if truth is not None:
        ax.axvline(truth, color="#111827", linestyle="--", linewidth=1.1, alpha=0.75, label="Truth")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    if all_y:
        ax.set_ylim(min(all_y) - 1.0, max(all_y) + 1.0)
    ax.invert_yaxis()
    max_total_trials = max(selection.total_trials_by_method.get(method, 0) for method in plotted_methods)
    shown_trials = min(max_trials, _MAX_INTERVAL_TRIALS_PER_METHOD)
    trial_text = f"{shown_trials} draws/method"
    if max_total_trials > shown_trials:
        trial_text = f"{shown_trials} of {max_total_trials} draws/method"
    ax.set_title(f"Trial CIs at One Budget\nB={_format_number(selection.budget)}, {trial_text}")
    ax.set_xlabel("Estimate")
    ax.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)


def _seed_sort_key(value: Any) -> tuple[int, float | str]:
    if pd.isna(value):
        return (2, "")
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _trial_group_label(method: str, frame: pd.DataFrame, truth: float | None) -> str:
    coverage = _coverage_for_trial_frame(frame, truth)
    suffix = "" if coverage is None else f" ({coverage:.2f})"
    return f"{_method_label(method)}{suffix}"


def _coverage_for_trial_frame(frame: pd.DataFrame, truth: float | None) -> float | None:
    if "covered" in frame.columns:
        values = pd.to_numeric(frame["covered"], errors="coerce").dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            return float(np.mean(values))
    if truth is None:
        return None
    lower = np.minimum(frame["lower"].to_numpy(dtype=float), frame["upper"].to_numpy(dtype=float))
    upper = np.maximum(frame["lower"].to_numpy(dtype=float), frame["upper"].to_numpy(dtype=float))
    finite = np.isfinite(lower) & np.isfinite(upper)
    if not np.any(finite):
        return None
    return float(np.mean((lower[finite] <= truth) & (truth <= upper[finite])))


def _trial_covers_truth(row: pd.Series, truth: float | None) -> bool:
    if "covered" in row.index and pd.notna(row["covered"]):
        return bool(float(row["covered"]) >= 0.5)
    if truth is None:
        return True
    lower = float(min(row["lower"], row["upper"]))
    upper = float(max(row["lower"], row["upper"]))
    return lower <= truth <= upper


def _plot_widths(ax: plt.Axes, summary: pd.DataFrame, methods: list[str]) -> None:
    for method in methods:
        curve = _curve_for_method(summary, method)
        if curve.empty:
            continue
        ax.plot(
            curve["budget"],
            curve["width"],
            marker=_METHOD_MARKERS.get(method, "o"),
            linewidth=2.0,
            markersize=5.0,
            color=_METHOD_COLORS.get(method, "#111827"),
            label=_method_label(method),
        )
    ax.set_title("Average CI Width")
    ax.set_xlabel("Budget")
    ax.set_ylabel("Width")
    _maybe_set_log_budget_axis(ax, summary["budget"].to_numpy(dtype=float))
    _maybe_set_log_width_axis(ax, summary["width"].to_numpy(dtype=float))
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="best")


def _plot_coverage(ax: plt.Axes, summary: pd.DataFrame, methods: list[str], nominal: float) -> None:
    for method in methods:
        curve = _curve_for_method(summary, method, value_column="covered")
        if curve.empty:
            continue
        ax.plot(
            curve["budget"],
            curve["covered"],
            marker=_METHOD_MARKERS.get(method, "o"),
            linewidth=2.0,
            markersize=5.0,
            color=_METHOD_COLORS.get(method, "#111827"),
            label=_method_label(method),
        )
    ax.axhline(nominal, color="#111827", linewidth=1.1, linestyle="--", alpha=0.7)
    ax.text(
        0.98,
        nominal,
        f"{nominal:.2f}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#111827",
    )
    ax.set_title("Empirical Coverage")
    ax.set_xlabel("Budget")
    ax.set_ylabel("Coverage")
    _maybe_set_log_budget_axis(ax, summary["budget"].to_numpy(dtype=float))
    y_values = summary["covered"].dropna().to_numpy(dtype=float)
    low = min(0.0, float(np.nanmin(y_values)) - 0.05) if y_values.size else 0.0
    high = max(1.0, float(np.nanmax(y_values)) + 0.05, nominal + 0.05) if y_values.size else 1.0
    ax.set_ylim(low, high)
    ax.grid(True, color="#E5E7EB", linewidth=0.8)


def _curve_for_method(summary: pd.DataFrame, method: str, value_column: str = "width") -> pd.DataFrame:
    method_key = _normalize_method(method)
    data = summary.loc[summary["method"] == method_key, ["budget", value_column]].copy()
    data = data.dropna()
    if data.empty:
        return data
    data = data[np.isfinite(data[["budget", value_column]].to_numpy(dtype=float)).all(axis=1)]
    return (
        data.groupby("budget", as_index=False, sort=True)[value_column]
        .mean()
        .sort_values("budget")
        .reset_index(drop=True)
    )


def _budget_savings_points(
    baseline_budgets: np.ndarray,
    baseline_widths: np.ndarray,
    robust_budgets: np.ndarray,
    robust_widths: np.ndarray,
) -> np.ndarray:
    points: list[tuple[float, float]] = []
    for baseline_budget, baseline_width in zip(baseline_budgets, baseline_widths, strict=False):
        required_budget = _budget_for_matching_width(
            robust_budgets,
            robust_widths,
            baseline_width,
            extrapolate=True,
        )
        if required_budget is None or required_budget <= 0.0:
            continue
        saved = 100.0 * (baseline_budget - required_budget) / baseline_budget
        if not np.isfinite(saved):
            continue
        points.append((float(baseline_budget), float(saved)))
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.asarray(points, dtype=float)


def _budget_for_matching_width(
    budgets: np.ndarray,
    widths: np.ndarray,
    target_width: float,
    *,
    extrapolate: bool = False,
) -> float | None:
    """Return interpolated budget required to reach ``target_width``.

    Width curves from Monte Carlo summaries can be slightly non-monotone. The
    interpolation therefore uses the best width attained up to each budget.
    When extrapolation is enabled, targets outside the observed frontier use
    the standard CI-width approximation ``width proportional to 1 / sqrt(B)``.
    """

    budgets = np.asarray(budgets, dtype=float)
    widths = np.asarray(widths, dtype=float)
    target = float(target_width)
    finite = np.isfinite(budgets) & np.isfinite(widths) & (budgets > 0.0)
    budgets = budgets[finite]
    widths = widths[finite]
    if budgets.size == 0 or not np.isfinite(target):
        return None

    order = np.argsort(budgets)
    budgets = budgets[order]
    widths = widths[order]
    best_widths = np.minimum.accumulate(widths)

    if target >= best_widths[0]:
        if extrapolate:
            return _sqrt_width_budget_extrapolation(budgets[0], best_widths[0], target)
        return float(budgets[0])
    if target < best_widths[-1]:
        if extrapolate:
            return _sqrt_width_budget_extrapolation(budgets[-1], best_widths[-1], target)
        return None

    matches = np.flatnonzero(best_widths <= target)
    if matches.size == 0:
        return None
    index = int(matches[0])
    if index == 0:
        return float(budgets[0])

    previous_width = float(best_widths[index - 1])
    current_width = float(best_widths[index])
    previous_budget = float(budgets[index - 1])
    current_budget = float(budgets[index])
    if np.isclose(previous_width, current_width):
        return current_budget
    fraction = (previous_width - target) / (previous_width - current_width)
    return float(previous_budget + fraction * (current_budget - previous_budget))


def _sqrt_width_budget_extrapolation(
    anchor_budget: float,
    anchor_width: float,
    target_width: float,
) -> float | None:
    anchor_budget = float(anchor_budget)
    anchor_width = float(anchor_width)
    target_width = float(target_width)
    if anchor_budget <= 0.0 or anchor_width <= 0.0 or target_width <= 0.0:
        return None
    budget = anchor_budget * (anchor_width / target_width) ** 2
    if not np.isfinite(budget) or budget <= 0.0:
        return None
    return float(budget)


def _nominal_coverage(summary: pd.DataFrame, trials: pd.DataFrame) -> float:
    for frame in (summary, trials):
        value = _first_finite_value((frame,), "confidence_level")
        if value is not None and 0.0 < value < 1.0:
            return value
    return 0.90


def _first_finite_value(frames: Iterable[pd.DataFrame], column: str) -> float | None:
    for frame in frames:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            return float(values[0])
    return None


def _add_metadata_note(fig: plt.Figure, summary: pd.DataFrame) -> None:
    pieces: list[str] = []
    for column in ("budget_model", "k_cost_interpretation"):
        if column in summary.columns:
            value = summary[column].dropna()
            if not value.empty:
                pieces.append(str(value.iloc[0]).replace("_", " "))
    if pieces:
        fig.text(0.995, 0.005, " | ".join(pieces), ha="right", va="bottom", fontsize=6, color="#6B7280")


def _maybe_set_log_budget_axis(ax: plt.Axes, budgets: np.ndarray) -> None:
    finite = np.asarray(budgets, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size >= 2 and float(np.nanmax(finite) / np.nanmin(finite)) >= 20.0:
        ax.set_xscale("log")


def _maybe_set_log_width_axis(ax: plt.Axes, widths: np.ndarray) -> None:
    finite = np.asarray(widths, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size >= 2 and float(np.nanmax(finite) / np.nanmin(finite)) >= 50.0:
        ax.set_yscale("log")


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": "#D1D5DB",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "axes.labelcolor": "#111827",
            "axes.titlecolor": "#111827",
            "legend.handlelength": 1.8,
        }
    )


def _method_label(method: str) -> str:
    return _METHOD_LABELS.get(_normalize_method(method), str(method).title())


def _format_number(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def _warn_if_missing(label: str, paths: Iterable[Path]) -> bool:
    missing = [str(path) for path in paths if not path.exists()]
    if not missing:
        return False
    warnings.warn(f"Skipping {label}: missing {', '.join(missing)}", RuntimeWarning, stacklevel=2)
    return True


if __name__ == "__main__":
    raise SystemExit(main())
