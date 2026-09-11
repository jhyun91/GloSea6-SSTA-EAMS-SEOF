# Seasonally evolving SST prediction in the East Asian marginal seas

Reproducible implementation of the SEOF-based seasonal SST prediction framework for the East Asian marginal seas, including model construction, forecast generation, and independent verification.

## Installation

```bash
git clone https://github.com/jhyun91/GloSea6-SSTA-EAMS-SEOF.git
cd GloSea6-SSTA-EAMS-SEOF
conda env create -f environment.yml
conda activate eams-seof
pip install -e .
```

## Required data

### NOAA OISST v2.1

Run `notebooks/00_Download_OISST.ipynb` to download monthly NOAA OISST v2.1.

### IHO Sea Areas

Download IHO Sea Areas Version 3 (2018) from Marine Regions and place the shapefile files under:

```text
data/external/World_Seas_IHO_v3/
```

## Workflow

Run the notebooks in order:

1. `00_Download_OISST.ipynb`
2. `01_Preprocess_OISST.ipynb`
3. `02_Generate_Region_Masks.ipynb`
4. `03_SEOF_Decomposition.ipynb`
5. `04_SEOF_Forecast_Model.ipynb`
6. `05_Generate_Forecasts.ipynb`
7. `06_Evaluate_Forecast_Skill.ipynb`

The workflow constructs the SEOF basis and forecast model from the training period, generates SST forecasts, and evaluates forecast skill over an independent verification period.

The primary forecast uses the leading three SEOF modes. A one-mode configuration using SEOF1 alone is included as a sensitivity experiment for evaluating how forecast skill changes when only the leading seasonal SST structure is retained.

## Analysis configuration

- Domain: 20–50°N, 120–140°E
- Training period: 1993–2012
- Verification period: 2013–2025
- Forecast-product anomaly reference: 1993–2016
- Verification anomaly reference: 1993–2012
- Retained SEOF modes: 3
- Forecast input: preceding 12 months
- Generated leads: 0–12 months
- Verification leads: 1–12 months
- Primary forecast: causally corrected forecast
- Primary verification metrics: uncentered ACC and MSSS
- Bootstrap resamples: 5,000
- Confidence interval: 95%
- Minimum verification samples: 10
- Multiple-testing control: Benjamini–Hochberg FDR, q = 0.10
- Random seed: 42

Core analysis settings are defined in `configs/manuscript.yml`.

## Forecast verification

Forecast skill is evaluated only over the independent 2013–2025 verification period.

The primary metrics are:

- **ACC**: uncentered anomaly correlation coefficient
- **MSSS**: mean squared skill score relative to a zero-anomaly climatological reference

Statistical support is assessed using bootstrap resampling and Benjamini–Hochberg false-discovery-rate control at `q = 0.10`.

A simple persistence forecast is evaluated over the same verification period as an additional benchmark.

## Outputs

Generated model files, forecasts, verification results, and figures are written under:

```text
outputs/
```

Main output subdirectories include:

```text
outputs/model/
outputs/forecasts/
outputs/verification/
outputs/png/
```

## Tests

Run the test suite with:

```bash
pytest -q
```
