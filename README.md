# Seasonally evolving SST prediction in the East Asian marginal seas

Reproducible Jupyter implementation of the SEOF-based seasonal SST prediction workflow used for the manuscript.

## Frozen paper release

**Release:** `v1.0.0-paper`

The paper workflow is frozen as:

* training period: **1993–2012**
* verification period: **2013–2025**
* retained basis: **SEOF1–3**
* monthly projection: cosine-latitude-weighted least squares (`G A = b`)
* predictor history: preceding **12 monthly projection scores**
* lag weights: pooled + target-month-specific **lag-correlation templates**
* target-month shrinkage pseudocount: **24**
* lag sequence normalization: unit absolute sum
* lead 0: exact same-month reconstruction diagnostic; not used in skill assessment
* primary forecast: three-mode forecast with **causal rolling mean-state correction**
* mode sensitivity: independently constructed **one-mode** and **uncorrected three-mode** forecasts
* anomaly verification reference: fixed **1993–2012 monthly climatology**

The ridge distributed-lag experiment used during model development is **not part of the production release**.

## Installation

```bash
git clone https://github.com/jhyun91/GloSea6-SSTA-EAMS-SEOF.git
cd GloSea6-SSTA-EAMS-SEOF
conda env create -f environment.yml
conda activate eams-seof
pip install -e .
```

The environment is pinned to Python 3.10, Matplotlib 3.4.3, ProPlot 0.9.7, and NumPy < 2 for reproducible figure rendering.

## Workflow

Run in order:

1. `00_Download_OISST.ipynb`
2. `01_Preprocess_OISST.ipynb`
3. `02_Generate_Region_Masks.ipynb`
4. `03_SEOF_Decomposition.ipynb`
5. `04_SEOF_Forecast_Model.ipynb`
6. `05_Generate_Forecasts.ipynb`
7. `06_Evaluate_Forecast_Skill.ipynb`

## Statistical verification

* primary ACC: uncentered anomaly correlation
* centered ACC: sensitivity diagnostic
* MSSS reference: zero anomaly relative to the fixed 1993–2012 climatology
* ACC support: temporal-permutation Monte Carlo null
* MSSS support: paired squared-error-advantage sign-flip null
* Benjamini–Hochberg FDR: `q = 0.10`
* one-mode minus three-mode differences: descriptive retained-basis sensitivity; no configuration-difference inferential test

## Tests

```bash
pytest -q
```
