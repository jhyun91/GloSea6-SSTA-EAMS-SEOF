from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter

from .time import standardize_monthly_time, datetime_to_decimal_year


def load_oisst(
    path: str | Path,
    start: str = "1993-01-01",
    variable: str | None = None,
) -> xr.DataArray:
    'Load and standardize monthly NOAA OISST.'
    ds = xr.open_dataset(path)

    if variable is None:
        candidates = [v for v in ds.data_vars if "sst" in v.lower()]
        variable = candidates[0] if candidates else list(ds.data_vars)[0]

    sst = ds[variable].astype(np.float32).rename("sst")

    if "lon" not in sst.coords or "lat" not in sst.coords:
        raise ValueError("Expected OISST coordinates named 'lat' and 'lon'.")

    sst = sst.assign_coords(lon=(sst.lon % 360)).sortby("lon").sortby("lat")
    sst = sst.assign_coords(time=standardize_monthly_time(sst.time))
    sst = sst.sel(time=slice(start, None))
    return sst


def smooth_sst(
    sst: xr.DataArray,
    window_size: int = 13,
) -> xr.DataArray:
    'Apply a NaN-aware spatial box filter.'
    if sst.dims[-2:] != ("lat", "lon"):
        sst = sst.transpose("time", "lat", "lon")

    values = sst.values.astype(np.float64)
    valid = np.isfinite(values)

    filled = np.where(valid, values, 0.0)
    size = (1, window_size, window_size)

    # Normalize by the valid-data fraction.
    numerator = uniform_filter(
        filled,
        size=size,
        mode=("nearest", "constant", "wrap"),
        cval=0.0,
    )
    denominator = uniform_filter(
        valid.astype(np.float64),
        size=size,
        mode=("nearest", "constant", "wrap"),
        cval=0.0,
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = numerator / denominator

    smoothed[(denominator <= 0) | (~valid)] = np.nan

    out = xr.DataArray(
        smoothed.astype(np.float32),
        coords=sst.coords,
        dims=sst.dims,
        name="sst",
        attrs=sst.attrs,
    )
    out.attrs["spatial_smoothing"] = (
        f"NaN-aware {window_size}x{window_size} box filter; periodic longitude"
    )
    return out


def training_clim(
    sst: xr.DataArray,
    training_period: Tuple[str, str],
    sea_mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    'Compute the training-period monthly climatology.'
    x = sst.where(sea_mask) if sea_mask is not None else sst
    return (
        x.sel(time=slice(*training_period))
        .groupby("time.month")
        .mean("time")
        .astype(np.float32)
    )


def detrend_anomaly(
    sst: xr.DataArray,
    training_period: Tuple[str, str],
    sea_mask: Optional[xr.DataArray] = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, float]:
    'Remove the training climatology and linear trend.'
    x = sst.where(sea_mask) if sea_mask is not None else sst

    climatology = training_clim(
        x, training_period, sea_mask=None
    )
    deseasonalized = x.groupby("time.month") - climatology

    full_decimal_year = xr.DataArray(
        datetime_to_decimal_year(
            pd.DatetimeIndex(deseasonalized.time.values)
        ).astype(np.float32),
        coords={"time": deseasonalized.time},
        dims=("time",),
    )

    train_year = full_decimal_year.sel(time=slice(*training_period))
    reference_decimal_year = float(train_year.mean().values)

    y = deseasonalized.sel(time=slice(*training_period))
    xx = train_year - reference_decimal_year

    x_mean = xx.mean("time")
    y_mean = y.mean("time")
    xc = xx - x_mean
    yc = y - y_mean

    slope = (xc * yc).sum("time") / (xc**2).sum("time")
    intercept = y_mean - slope * x_mean

    trend_coefficients = xr.concat(
        [intercept.rename("intercept"), slope.rename("slope")],
        dim="coef",
    ).assign_coords(coef=["intercept", "slope"])
    trend_coefficients = trend_coefficients.transpose("coef", "lat", "lon")

    trend_full = (
        intercept + slope * (full_decimal_year - reference_decimal_year)
    ).transpose("time", "lat", "lon")

    anomaly = (deseasonalized - trend_full).astype(np.float32)
    anomaly.name = "sst_anomaly_detrended"

    return (
        anomaly,
        climatology.astype(np.float32),
        trend_coefficients.astype(np.float32),
        reference_decimal_year,
    )


def fixed_clim(
    sst: xr.DataArray,
    base_period: Tuple[str, str],
    sea_mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    'Compute the fixed monthly climatology base.'
    x = sst.where(sea_mask) if sea_mask is not None else sst
    return (
        x.sel(time=slice(*base_period))
        .groupby("time.month")
        .mean("time")
        .astype(np.float32)
    )
