from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def standardize_monthly_time(time: xr.DataArray) -> xr.DataArray:
    'Convert monthly timestamps to month-start.'
    idx = (
        pd.DatetimeIndex(pd.to_datetime(time.values))
        .to_period("M")
        .to_timestamp(how="start")
    )
    return xr.DataArray(
        idx.to_numpy(dtype="datetime64[ns]"),
        coords=time.coords,
        dims=time.dims,
    )


def datetime_to_decimal_year(index: pd.DatetimeIndex) -> np.ndarray:
    'Convert timestamps to decimal year.'
    year = index.year
    start = pd.to_datetime(year.astype(str) + "-01-01")
    end = pd.to_datetime((year + 1).astype(str) + "-01-01")

    decimal_year = year + (index - start).days / (end - start).days
    return np.asarray(decimal_year, dtype=float)


def month_start(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp(how="start")


def make_target_time_matrix(
    initialization_times: pd.DatetimeIndex,
    leads: np.ndarray,
) -> xr.DataArray:
    'Build target times from initialization times and leads.'
    values = np.empty(
        (len(initialization_times), len(leads)),
        dtype="datetime64[ns]",
    )
    for j, lead in enumerate(leads.astype(int)):
        values[:, j] = (
            initialization_times + pd.DateOffset(months=int(lead))
        ).values

    return xr.DataArray(
        values,
        coords={"init": initialization_times, "lead": leads},
        dims=("init", "lead"),
    )
