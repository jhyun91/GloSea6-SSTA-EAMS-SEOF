from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from .time import (
    make_target_time_matrix,
    datetime_to_decimal_year,
)
from .preprocessing import fixed_clim


def forecast_scores(
    monthly_projection_scores: xr.DataArray,
    lag_weights: xr.DataArray,
    input_sequence_months: int = 12,
) -> xr.DataArray:
    'Compute forecast component scores for each lead.'
    scores = monthly_projection_scores
    leads = lag_weights.lead.values.astype(int)
    modes = scores.mode.values

    out = xr.DataArray(
        np.full(
            (scores.sizes["time"], len(leads), len(modes)),
            np.nan,
            dtype=np.float32,
        ),
        coords={
            "time": scores.time,
            "lead": leads,
            "mode": modes,
        },
        dims=("time", "lead", "mode"),
        name="forecast_component_score",
    )

    init_month = pd.DatetimeIndex(scores.time.values).month
    values = scores.values

    for lead_index, lead in enumerate(leads):
        target_month = ((init_month + lead - 1) % 12) + 1

        for time_index in range(scores.sizes["time"]):
            if time_index - (input_sequence_months - 1) < 0:
                continue

            history = values[
                time_index - (input_sequence_months - 1):
                time_index + 1,
                :
            ][::-1, :]

            month = int(target_month[time_index])

            for mode_index, mode in enumerate(modes):
                x = history[:, mode_index]
                if not np.all(np.isfinite(x)):
                    continue

                w = lag_weights.sel(
                    mode=mode,
                    month=month,
                    lead=int(lead),
                ).values
                out[
                    time_index, lead_index, mode_index
                ] = float(np.dot(x, w))

    return out


def reconstruct_forecast(
    forecast_component_scores: xr.DataArray,
    aligned_patterns: xr.DataArray,
    seof_mean: xr.DataArray,
    sea_mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    'Reconstruct the target-month SST anomaly forecast.'
    scores = forecast_component_scores
    leads = scores.lead.values.astype(int)

    out = xr.DataArray(
        np.full(
            (
                len(leads),
                scores.sizes["time"],
                aligned_patterns.sizes["lat"],
                aligned_patterns.sizes["lon"],
            ),
            np.nan,
            dtype=np.float32,
        ),
        coords={
            "lead": leads,
            "time": scores.time,
            "lat": aligned_patterns.lat,
            "lon": aligned_patterns.lon,
        },
        dims=("lead", "time", "lat", "lon"),
        name="sst_anomaly_forecast_detrended",
    )

    init_month = pd.DatetimeIndex(scores.time.values).month
    pattern_values = aligned_patterns.values
    score_values = scores.values

    for lead_index, lead in enumerate(leads):
        target_month = ((init_month + lead - 1) % 12) + 1

        for time_index in range(scores.sizes["time"]):
            component = score_values[time_index, lead_index, :]
            if not np.any(np.isfinite(component)):
                continue

            month_index = int(target_month[time_index]) - 1
            field = np.einsum(
                "k,kij->ij",
                np.nan_to_num(
                    component,
                    nan=0.0,
                ),
                pattern_values[:, month_index],
            )
            
            field = (
                field
                + seof_mean.sel(
                    month=month_index + 1
                ).values
            )
            
            out[
                lead_index,
                time_index,
            ] = field.astype(np.float32)

    if sea_mask is not None:
        out = out.where(sea_mask)
    return out


def restore_state(
    detrended_anomaly_forecast: xr.DataArray,
    training_climatology: xr.DataArray,
    trend_coefficients: xr.DataArray,
    reference_decimal_year: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    'Restore the training climatology and trend.'
    x = detrended_anomaly_forecast.rename(
        time="init"
    ).transpose("init", "lead", "lat", "lon")

    initialization_times = pd.DatetimeIndex(x.init.values)
    leads = x.lead.values.astype(int)

    target_time = make_target_time_matrix(
        initialization_times, leads
    )

    flat = pd.DatetimeIndex(target_time.values.reshape(-1))
    target_month = xr.DataArray(
        flat.month.to_numpy().reshape(target_time.shape),
        coords=target_time.coords,
        dims=target_time.dims,
    )
    target_decimal_year = xr.DataArray(
        datetime_to_decimal_year(flat).astype(np.float32).reshape(
            target_time.shape
        ),
        coords=target_time.coords,
        dims=target_time.dims,
    )

    seasonal = training_climatology.sel(month=target_month)

    intercept = trend_coefficients.sel(coef="intercept")
    slope = trend_coefficients.sel(coef="slope")
    trend = (
        intercept
        + slope
        * (target_decimal_year - float(reference_decimal_year))
    )

    absolute = (x + seasonal + trend).astype(np.float32)
    absolute.name = "sst_forecast_absolute"
    return absolute.reset_coords(drop=True), target_time


def forecast_to_target(
    forecast_init: xr.DataArray,
) -> xr.DataArray:
    'Convert forecasts from initialization time to target time.'
    initialization_times = pd.DatetimeIndex(
        forecast_init.init.values
    )
    leads = forecast_init.lead.values.astype(int)

    pieces = []
    for lead in leads:
        target = initialization_times + pd.DateOffset(
            months=int(lead)
        )
        piece = (
            forecast_init.sel(lead=int(lead))
            .rename({"init": "target_time"})
            .assign_coords(target_time=target)
            .expand_dims(lead=[int(lead)])
        )
        pieces.append(piece)

    return (
        xr.concat(
            pieces,
            dim="lead",
            join="outer",
            coords="minimal",
            compat="equals",
        )
        .transpose("lead", "target_time", "lat", "lon")
        .sortby("target_time")
    )


def fixed_base_anomaly(
    absolute_forecast_target: xr.DataArray,
    observed_sst: xr.DataArray,
    fixed_base_period: tuple[str, str] | None = None,
    sea_mask: xr.DataArray | None = None,
    anomaly_reference_period: tuple[str, str] | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Convert forecasts and observations to one fixed climatology base.

    ``anomaly_reference_period`` is the preferred explicit name used by the
    revised verification notebook.  ``fixed_base_period`` is retained for
    backward compatibility; if both are supplied they must agree.
    """
    if anomaly_reference_period is None and fixed_base_period is None:
        raise ValueError("Provide anomaly_reference_period or fixed_base_period.")
    if anomaly_reference_period is None:
        anomaly_reference_period = fixed_base_period
    if fixed_base_period is not None and tuple(fixed_base_period) != tuple(anomaly_reference_period):
        raise ValueError("fixed_base_period and anomaly_reference_period disagree.")

    climatology = fixed_clim(
        observed_sst,
        tuple(anomaly_reference_period),
        sea_mask=sea_mask,
    )

    month = xr.DataArray(
        pd.DatetimeIndex(
            absolute_forecast_target.target_time.values
        ).month,
        coords={
            "target_time": absolute_forecast_target.target_time
        },
        dims=("target_time",),
    )

    base = climatology.sel(month=month).assign_coords(
        target_time=absolute_forecast_target.target_time
    )
    base = base.expand_dims(
        lead=absolute_forecast_target.lead
    ).transpose("lead", "target_time", "lat", "lon")

    forecast_anomaly = (
        absolute_forecast_target - base
    ).astype(np.float32)
    forecast_anomaly.name = "sst_forecast_fixed_base_anomaly"

    observed_anomaly = (
        observed_sst.groupby("time.month") - climatology
    ).astype(np.float32)
    observed_anomaly.name = "sst_observed_fixed_base_anomaly"

    if "month" in forecast_anomaly.coords:
        forecast_anomaly = forecast_anomaly.reset_coords("month", drop=True)
    
    if "month" in observed_anomaly.coords:
        observed_anomaly = observed_anomaly.reset_coords("month", drop=True)
    
    return forecast_anomaly, observed_anomaly
