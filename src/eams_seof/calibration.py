from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from .time import make_target_time_matrix, month_start


def clim_correction(
    uncorrected_forecast_target: xr.DataArray,
    observed_sst: xr.DataArray,
    initialization_times: pd.DatetimeIndex,
    leads: np.ndarray,
    window_years: int = 7,
    apply_from: str = "2013-01-01",
    minimum_archive_months: int = 12,
    archive_start: Optional[str] = "2013-01-01",
) -> xr.DataArray:
    'Apply the causal rolling climatological correction.'
    target_time = make_target_time_matrix(
        initialization_times, leads
    )

    target_month = xr.DataArray(
        np.asarray(
            pd.DatetimeIndex(
                target_time.values.reshape(-1)
            ).month
        ).reshape(target_time.shape),
        coords=target_time.coords,
        dims=target_time.dims,
    )

    data_start = month_start(
        pd.DatetimeIndex(observed_sst.time.values).min()
    )
    floor = (
        data_start
        if archive_start is None
        else max(data_start, month_start(archive_start))
    )

    zero = xr.DataArray(
        np.zeros(
            (
                12,
                len(leads),
                uncorrected_forecast_target.sizes["lat"],
                uncorrected_forecast_target.sizes["lon"],
            ),
            dtype=np.float32,
        ),
        dims=("month", "lead", "lat", "lon"),
        coords={
            "month": np.arange(1, 13),
            "lead": leads,
            "lat": uncorrected_forecast_target.lat,
            "lon": uncorrected_forecast_target.lon,
        },
    )

    corrections = []

    for init in initialization_times:
        archive_end = month_start(
            init - pd.DateOffset(months=1)
        )
        rolling_start = month_start(
            archive_end
            - pd.DateOffset(years=window_years)
            + pd.DateOffset(months=1)
        )
        archive_begin = max(floor, rolling_start)

        obs_recent = observed_sst.sel(
            time=slice(
                archive_begin.strftime("%Y-%m-%d"),
                archive_end.strftime("%Y-%m-%d"),
            )
        )

        fcst_recent = uncorrected_forecast_target.sel(
            target_time=slice(
                archive_begin.strftime("%Y-%m-%d"),
                archive_end.strftime("%Y-%m-%d"),
            )
        )

        if (
            obs_recent.sizes.get("time", 0)
            >= minimum_archive_months
            and fcst_recent.sizes.get("target_time", 0)
            >= minimum_archive_months
        ):
            obs_clim = (
                obs_recent.groupby("time.month")
                .mean("time")
                .reindex(month=np.arange(1, 13))
                .astype(np.float32)
            )

            model_clim = (
                fcst_recent.groupby("target_time.month")
                .mean("target_time")
                .reindex(month=np.arange(1, 13))
                .transpose("month", "lead", "lat", "lon")
                .astype(np.float32)
            )

            correction = (
                obs_clim - model_clim
            ).transpose("month", "lead", "lat", "lon")
            correction = correction.fillna(0.0).astype(np.float32)
        else:
            correction = zero

        month_for_init = target_month.sel(init=init)
        correction_for_init = (
            correction.sel(month=month_for_init)
            .assign_coords(lead=leads)
            .transpose("lead", "lat", "lon")
            .reset_coords(drop=True)
        )

        corrections.append(
            correction_for_init.expand_dims(init=[init])
        )

    correction_init = (
        xr.concat(
            corrections,
            dim="init",
            join="outer",
            coords="minimal",
            compat="equals",
        )
        .transpose("init", "lead", "lat", "lon")
        .astype(np.float32)
    )

    gate = xr.where(
        target_time >= np.datetime64(month_start(apply_from)),
        1.0,
        0.0,
    ).astype(np.float32)

    return (correction_init * gate).astype(np.float32)
