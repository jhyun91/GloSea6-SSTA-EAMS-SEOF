# Seasonally evolving SST prediction in the East Asian marginal seas

Jupyter-based implementation of the SEOF seasonal SST prediction workflow for the East Asian marginal seas.

## Installation

```bash
git clone https://github.com/jhyun91/GloSea6-SSTA-EAMS-SEOF.git
cd GloSea6-SSTA-EAMS-SEOF

conda env create -f environment.yml
conda activate eams-seof
pip install -e .
```

## Data

### NOAA OISST v2.1

Notebook `00_Download_OISST.ipynb` downloads the monthly NOAA OISST v2.1 dataset. The equivalent command is:

```bash
mkdir -p data/raw/OISST
wget -c \
https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc \
-O data/raw/OISST/sst.mon.mean.nc
```

### IHO Sea Areas

Download **IHO Sea Areas — Version 3 (2018)** from the
[Marine Regions download page](https://www.marineregions.org/downloads.php).
Place the shapefile and companion files under:

```text
data/external/World_Seas_IHO_v3/
```

The IHO data are not redistributed with this repository.

## Reproduce the workflow

Run the notebooks in order:

1. `00_Download_OISST.ipynb`
2. `01_Preprocess_OISST.ipynb`
3. `02_Generate_Region_Masks.ipynb`
4. `03_SEOF_Decomposition.ipynb`
5. `04_SEOF_Forecast_Model.ipynb`
6. `05_Generate_Forecasts.ipynb`
7. `06_Evaluate_Forecast_Skill.ipynb`

The manuscript configuration is `configs/manuscript.yml`.
Notebook 05 also generates a mode-1 reference forecast used only to quantify the added ACC from SEOF modes 2–3.

## Model configuration

- Training: 1993–2012
- Verification: 2013–2025
- SST smoothing: 13 × 13 grid points
- Retained SEOFs: 3
- Monthly SEOF alignment: cosine-latitude-weighted orthogonal Procrustes
- Input sequence: 12 months
- Forecast leads: 0–12 months
- Lag-sequence polarity: lag-0 nonnegative convention
- Rolling climatological correction: 7 years
- Fixed verification anomaly base: 1993–2016
- Primary ACC: uncentered anomaly correlation relative to the fixed anomaly base

## Tests

```bash
pytest -q
```
