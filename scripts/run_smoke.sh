#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

run_cmd() {
  echo
  echo "+ $*"
  "$@"
}

echo "Running local experiment-code smoke checks from ${ROOT_DIR}"
echo "Pew-dependent smoke commands use --allow-missing-data, so missing pyreadstat or Pew .sav data are reported as skips instead of failing the smoke run."

run_cmd python -m pytest

run_cmd python -m experiments.pew --config configs/pew.yaml --outcome biden --smoke --allow-missing-data
run_cmd python -m experiments.pew --config configs/pew.yaml --outcome trump --smoke --allow-missing-data

run_cmd python -m experiments.protein --config configs/protein.yaml --smoke
run_cmd python -m experiments.acs --config configs/acs.yaml --smoke

run_cmd python -m experiments.robustness --config configs/robustness.yaml --which posterior --smoke --allow-missing-data
run_cmd python -m experiments.robustness --config configs/robustness.yaml --which misspecification --smoke --allow-missing-data

run_cmd python -m plotting.figures --all --output-dir outputs/figures

echo
echo "Smoke checks completed. Full experiment grids were not run."
