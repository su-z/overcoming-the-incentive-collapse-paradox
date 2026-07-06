# Overcoming the Incentive Collapse Paradox Code

Code for the ICML 2026 paper "Overcoming the Incentive Collapse Paradox."

## Paper

Qichuan Yin*, Ziwei Su*, and Shuangning Li (2026). "Overcoming the Incentive Collapse Paradox." ICML 2026. arXiv:2603.27049.

*Equal contribution.

- ICML poster: https://icml.cc/virtual/2026/poster/60666
- arXiv: https://arxiv.org/abs/2603.27049

## Setup

```bash
uv sync
```

## Data

The data files are not tracked in git. Create these paths and place the downloaded files there:

```text
data/pew/ATP W79.sav
data/alphafold/alphafold.npz
data/2019/1-Year/psam_p06.csv
```

- Pew ATP Wave 79: download from https://www.pewresearch.org/dataset/american-trends-panel-wave-79/
- Protein/AlphaFold arrays: place `alphafold.npz` at the path above. As an
  optional download helper, run
  `uvx --from ppi-python python -c "from ppi_py.datasets import load_dataset; load_dataset('data/alphafold', 'alphafold')"`
  in a separate temporary environment.
- ACS PUMS: download the 2019 1-year California person file from https://www.census.gov/programs-surveys/acs/microdata.html and save `psam_p06.csv`.

## Run

```bash
uv run python -m experiments.pew --config configs/pew.yaml --outcome biden
uv run python -m experiments.pew --config configs/pew.yaml --outcome trump
uv run python -m experiments.protein --config configs/protein.yaml
uv run python -m experiments.acs --config configs/acs.yaml
uv run python -m experiments.robustness --config configs/robustness.yaml --which posterior
uv run python -m experiments.robustness --config configs/robustness.yaml --which misspecification
uv run python -m plotting.figures --all --output-dir outputs/figures
```

Outputs are written to `outputs/tables/` and `outputs/figures/`.

## Smoke Check

```bash
uv run bash scripts/run_smoke.sh
```
