from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.protein import (  # noqa: E402
    load_alphafold_npz,
    prepare_protein_dataset,
    protein_truth,
    run_protein_experiment,
)


def _toy_arrays() -> dict[str, np.ndarray]:
    return {
        "Y": np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float),
        "Yhat": np.array([0.90, 0.20, 0.70, 0.10, 0.80, 0.40, 0.60, 0.30, 0.55, 0.45, 0.65, 0.35]),
        "phosphorylated": np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
        "ubiquitinated": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        "acetylated": np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    }


def _write_toy_npz(tmp_path: Path, **overrides: np.ndarray) -> Path:
    arrays = _toy_arrays()
    arrays.update(overrides)
    path = tmp_path / "toy_alphafold.npz"
    np.savez(path, **arrays)
    return path


def _write_config(tmp_path: Path, raw_file: Path) -> Path:
    config = {
        "experiment": "toy_protein",
        "seeds": [11, 12],
        "smoke_seeds": [11, 12],
        "confidence_level": 0.90,
        "baseline_effort": 0.8,
        "data": {"raw_file": str(raw_file), "estimand": "odds_ratio"},
        "incentive": {"w0": 0.64, "k": 1.0, "rho_grid": [0.10, 0.20]},
        "sampling": {"baseline_mix": {"tau": 0.5}},
        "budget_grid": {"smoke": [8.0], "full": [8.0, 10.0]},
        "output": {"tables_dir": str(tmp_path / "tables")},
    }
    path = tmp_path / "protein.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_load_alphafold_npz_validates_and_exposes_expected_arrays(tmp_path: Path):
    path = _write_toy_npz(tmp_path)

    loaded = load_alphafold_npz(path)

    assert loaded.y_idr.shape == (12,)
    assert loaded.ai_probability.shape == (12,)
    np.testing.assert_array_equal(loaded.ai_label, (loaded.ai_probability > 0.5).astype(float))
    np.testing.assert_array_equal(loaded.phosphorylated, _toy_arrays()["phosphorylated"].astype(float))

    missing_path = tmp_path / "missing.npz"
    np.savez(missing_path, Y=np.array([0, 1]), Yhat=np.array([0.1, 0.9]))
    with pytest.raises(ValueError, match="missing required keys"):
        load_alphafold_npz(missing_path)

    invalid_path = _write_toy_npz(tmp_path, Yhat=np.array([0.2, 1.2] * 6))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_alphafold_npz(invalid_path)


def test_protein_truth_uses_odds_ratio_and_rejects_edge_cases():
    y_idr = np.array([1.0, 0.0, 1.0, 0.0])
    phosphorylated = np.array([1.0, 1.0, 0.0, 0.0])

    assert protein_truth(y_idr, phosphorylated) == pytest.approx(1.0)
    assert protein_truth(
        np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
    ) == pytest.approx(4.0)

    with pytest.raises(ValueError, match="unphosphorylated"):
        protein_truth(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        protein_truth(np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        protein_truth(np.array([1.2, 0.0]), np.array([1.0, 0.0]))


def test_prepare_protein_dataset_computes_tau_and_truth(tmp_path: Path):
    raw_file = _write_toy_npz(tmp_path)
    config_path = _write_config(tmp_path, raw_file)

    dataset = prepare_protein_dataset(config_path)

    assert dataset.n == 12
    assert dataset.truth == pytest.approx(1.0)
    assert np.all(dataset.tau > 0.0)
    np.testing.assert_allclose(dataset.tau, np.maximum((dataset.y_idr - dataset.ai_probability) ** 2, 1e-6))
    assert dataset.metadata["target_parameter"] == "odds(IDR=1|phosphorylated=1)/odds(IDR=1|phosphorylated=0)"


def test_run_protein_experiment_writes_schema_and_finite_intervals(tmp_path: Path):
    raw_file = _write_toy_npz(tmp_path)
    config_path = _write_config(tmp_path, raw_file)

    trials, summary = run_protein_experiment(config_path, smoke=True)

    expected_prefix = ["method", "budget", "estimate", "lower", "upper", "width", "covered", "seed"]
    assert list(trials.columns[:8]) == expected_prefix
    assert list(summary.columns[:8]) == expected_prefix
    assert set(trials["method"]) == {"classical", "uniform", "active", "robust"}
    assert set(summary["method"]) == {"classical", "uniform", "active", "robust"}
    assert len(trials) == 8
    assert np.all(np.isfinite(trials[["estimate", "lower", "upper", "width"]].to_numpy()))
    assert np.all(trials["upper"] >= trials["lower"])
    assert np.all(summary["n_trials"] == 2)
    assert (tmp_path / "tables" / "protein_trials.csv").exists()
    assert (tmp_path / "tables" / "protein_summary.csv").exists()
