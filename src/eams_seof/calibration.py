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
    return_diagnostics: bool = False,
):
    """Apply a causal rolling mean-state correction.

    Only observations and forecasts whose *target dates precede the current
    initialization* enter the archive.  For each target calendar month and
    lead, the recent observed mean minus recent forecast mean is added to the
    uncorrected forecast.  No correction is applied until both archives span
    at least ``minimum_archive_months`` and at least one matched historical
    sample exists for the requested target-month/lead combination.

    Parameters are intentionally identical to the original implementation;
    ``return_diagnostics=True`` additionally returns archive sample counts,
    target months, and an application flag for notebook 06.
    """
    uncorrected_forecast_target = uncorrected_forecast_target.transpose(
        "lead", "target_time", "lat", "lon"
    )
    initialization_times = pd.DatetimeIndex(initialization_times)
    leads = np.asarray(leads, dtype=int)

    target_time = make_target_time_matrix(initialization_times, leads)
    flat_target = pd.DatetimeIndex(target_time.values.reshape(-1))
    target_month = xr.DataArray(
        flat_target.month.to_numpy().reshape(target_time.shape).astype(np.int16),
        coords=target_time.coords,
        dims=target_time.dims,
        name="rolling_correction_target_month",
    )

    data_start = month_start(pd.DatetimeIndex(observed_sst.time.values).min())
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
    count_rows = []
    ready_rows = []

    obs_all_time = pd.DatetimeIndex(observed_sst.time.values)

    for init in initialization_times:
        archive_end = month_start(init - pd.DateOffset(months=1))
        rolling_start = month_start(
            archive_end - pd.DateOffset(years=window_years) + pd.DateOffset(months=1)
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

        archive_ready = (
            obs_recent.sizes.get("time", 0) >= minimum_archive_months
            and fcst_recent.sizes.get("target_time", 0) >= minimum_archive_months
        )

        count_month_lead = xr.DataArray(
            np.zeros((12, len(leads)), dtype=np.int32),
            dims=("month", "lead"),
            coords={"month": np.arange(1, 13), "lead": leads},
        )

        if archive_ready:
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
            correction = (obs_clim - model_clim).transpose(
                "month", "lead", "lat", "lon"
            )

            # Count matched target dates for each month and lead.  The count is
            # temporal (not grid-cell dependent) and is therefore transparent
            # for manuscript diagnostics.
            recent_target = pd.DatetimeIndex(fcst_recent.target_time.values)
            obs_index = pd.DatetimeIndex(obs_recent.time.values)
            obs_period = obs_index.to_period("M")
            obs_valid = np.isfinite(obs_recent).any(("lat", "lon")).values
            obs_valid_by_period = {p: bool(v) for p, v in zip(obs_period, obs_valid)}
            target_period = recent_target.to_period("M")

            for lead in leads:
                fcst_valid = np.isfinite(
                    fcst_recent.sel(lead=int(lead))
                ).any(("lat", "lon")).values
                matched_obs = np.array(
                    [obs_valid_by_period.get(p, False) for p in target_period],
                    dtype=bool,
                )
                for month in range(1, 13):
                    month_mask = recent_target.month == month
                    n_match = int(np.sum(month_mask & matched_obs & fcst_valid))
                    count_month_lead.loc[dict(month=month, lead=int(lead))] = n_match

            # Missing month/lead climatologies correspond to no usable archive
            # and must not silently create an offset.
            correction = correction.where(count_month_lead > 0, 0.0)
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
        count_for_init = (
            count_month_lead.sel(month=month_for_init)
            .assign_coords(lead=leads)
            .transpose("lead")
            .reset_coords(drop=True)
            .astype(np.int32)
        )

        corrections.append(correction_for_init.expand_dims(init=[init]))
        count_rows.append(count_for_init.expand_dims(init=[init]))
        ready_rows.append(
            xr.DataArray(
                np.full(len(leads), bool(archive_ready), dtype=bool),
                dims=("lead",),
                coords={"lead": leads},
            ).expand_dims(init=[init])
        )

    correction_init = (
        xr.concat(corrections, dim="init", join="outer", coords="minimal", compat="equals")
        .transpose("init", "lead", "lat", "lon")
        .astype(np.float32)
    )
    archive_count = (
        xr.concat(count_rows, dim="init", join="outer", coords="minimal", compat="equals")
        .transpose("init", "lead")
        .astype(np.int32)
        .rename("rolling_correction_archive_sample_count")
    )
    archive_ready_da = (
        xr.concat(ready_rows, dim="init", join="outer", coords="minimal", compat="equals")
        .transpose("init", "lead")
    )

    gate = xr.DataArray(
        target_time.values >= np.datetime64(month_start(apply_from)),
        coords=target_time.coords,
        dims=target_time.dims,
    )
    applied = (
        gate & archive_ready_da & (archive_count > 0)
    ).rename("rolling_correction_applied")

    correction_init = correction_init.where(applied, 0.0).astype(np.float32)
    correction_init.name = "rolling_mean_state_correction"

    if not return_diagnostics:
        return correction_init

    diagnostics = xr.Dataset(
        {
            "rolling_correction_archive_sample_count": archive_count,
            "rolling_correction_target_month": target_month.astype(np.int16),
            "rolling_correction_applied": applied.astype(bool),
        }
    )
    diagnostics.attrs.update(
        archive_start="none" if archive_start is None else str(archive_start),
        window_years=int(window_years),
        minimum_archive_months=int(minimum_archive_months),
        causality="only target dates preceding initialization are used",
    )
    return correction_init, diagnostics
