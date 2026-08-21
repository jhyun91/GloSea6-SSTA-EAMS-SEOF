from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr


def _lead_with_nan(x: np.ndarray, lead: int) -> np.ndarray:
    y = np.full_like(x, np.nan, dtype=np.float32)
    if lead == 0:
        y[:] = x
    else:
        y[:-lead] = x[lead:]
    return y


def _lag_with_nan(x: np.ndarray, lag: int) -> np.ndarray:
    y = np.full_like(x, np.nan, dtype=np.float32)
    if lag == 0:
        y[:] = x
    else:
        y[lag:] = x[:-lag]
    return y


def estimate_lag_templates(
    monthly_scores: xr.DataArray,
    training_period: Tuple[str, str],
    maximum_lead_months: int,
    input_sequence_months: int,
) -> xr.Dataset:
    'Estimate pooled and target-month lag-correlation templates.'
    scores = monthly_scores.sel(
        time=slice(*training_period)
    ).dropna("time", how="all")

    time = pd.DatetimeIndex(scores.time.values)
    modes = scores.mode.values
    leads = np.arange(maximum_lead_months + 1)
    lags = np.arange(input_sequence_months)

    pooled = xr.DataArray(
        np.zeros(
            (len(modes), len(leads), len(lags)),
            dtype=np.float32,
        ),
        dims=("mode", "lead", "lag"),
        coords={"mode": modes, "lead": leads, "lag": lags},
        name="pooled_lag_correlation",
    )

    target = xr.DataArray(
        np.zeros(
            (len(modes), 12, len(leads), len(lags)),
            dtype=np.float32,
        ),
        dims=("mode", "month", "lead", "lag"),
        coords={
            "mode": modes,
            "month": np.arange(1, 13),
            "lead": leads,
            "lag": lags,
        },
        name="target_month_lag_correlation",
    )

    x_all = scores.values.astype(np.float32)

    for mode_index, mode in enumerate(modes):
        x = x_all[:, mode_index]

        for lead in leads:
            y = _lead_with_nan(x, int(lead))
            target_month = np.asarray(
                (time + pd.DateOffset(months=int(lead))).month
            )

            for lag in lags:
                xlag = _lag_with_nan(x, int(lag))
                valid = np.isfinite(xlag) & np.isfinite(y)
                if valid.sum() >= 3:
                    c = np.corrcoef(xlag[valid], y[valid])[0, 1]
                    pooled.loc[
                        dict(mode=mode, lead=int(lead), lag=int(lag))
                    ] = 0.0 if not np.isfinite(c) else float(c)

            for month in range(1, 13):
                month_selection = target_month == month

                for lag in lags:
                    xlag = _lag_with_nan(x, int(lag))
                    valid = (
                        month_selection
                        & np.isfinite(xlag)
                        & np.isfinite(y)
                    )
                    if valid.sum() >= 3:
                        c = np.corrcoef(xlag[valid], y[valid])[0, 1]
                        target.loc[
                            dict(
                                mode=mode,
                                month=month,
                                lead=int(lead),
                                lag=int(lag),
                            )
                        ] = 0.0 if not np.isfinite(c) else float(c)

    return xr.Dataset(
        {
            "pooled": pooled,
            "target_month": target,
        }
    )


def normalize_lag(
    template: xr.DataArray,
) -> xr.DataArray:
    'Normalize each lag sequence by its mean absolute value.'
    denominator = np.abs(template).mean("lag")
    return template / xr.where(denominator > 0, denominator, 1.0)


def regularize_lag(
    template: xr.DataArray,
    method: str = "lag0_nonnegative",
) -> xr.DataArray:
    'Apply the lag-sequence polarity convention.'
    if method == "raw_signed":
        return template.copy()

    if method != "lag0_nonnegative":
        raise ValueError(
            "method must be 'lag0_nonnegative' or 'raw_signed'"
        )

    sign = xr.where(template.sel(lag=0) < 0, -1.0, 1.0)
    return (template * sign).astype(np.float32)


def count_training_samples(
    monthly_scores: xr.DataArray,
    training_period: Tuple[str, str],
    maximum_lead_months: int,
    input_sequence_months: int,
) -> xr.DataArray:
    'Count valid training cases for shrinkage.'
    scores = monthly_scores.sel(
        time=slice(*training_period)
    ).dropna("time", how="all")
    time = pd.DatetimeIndex(scores.time.values)
    values = scores.values.astype(np.float32)

    leads = np.arange(maximum_lead_months + 1)
    modes = scores.mode.values

    count = xr.DataArray(
        np.zeros(
            (len(modes), 12, len(leads)),
            dtype=np.int32,
        ),
        dims=("mode", "month", "lead"),
        coords={
            "mode": modes,
            "month": np.arange(1, 13),
            "lead": leads,
        },
        name="training_sample_count",
    )

    for mode_index, mode in enumerate(modes):
        x = values[:, mode_index]

        for lead in leads:
            y = _lead_with_nan(x, int(lead))
            target_month = np.asarray(
                (time + pd.DateOffset(months=int(lead))).month
            )

            complete_history = np.isfinite(y)
            for lag in range(input_sequence_months):
                complete_history &= np.isfinite(
                    _lag_with_nan(x, lag)
                )

            for month in range(1, 13):
                count.loc[
                    dict(mode=mode, month=month, lead=int(lead))
                ] = int(
                    ((target_month == month) & complete_history).sum()
                )

    return count


def lag_weights(
    monthly_scores: xr.DataArray,
    training_period: Tuple[str, str],
    maximum_lead_months: int = 12,
    input_sequence_months: int = 12,
    shrinkage_pseudocount: int = 24,
    polarity_method: str = "lag0_nonnegative",
) -> xr.Dataset:
    'Build the final lag-weight kernels.'
    raw = estimate_lag_templates(
        monthly_scores=monthly_scores,
        training_period=training_period,
        maximum_lead_months=maximum_lead_months,
        input_sequence_months=input_sequence_months,
    )

    pooled = normalize_lag(
        raw["pooled"]
    )
    target = normalize_lag(
        raw["target_month"]
    )

    pooled = regularize_lag(
        pooled, method=polarity_method
    )
    target = regularize_lag(
        target, method=polarity_method
    )

    pooled_month = pooled.expand_dims(
        month=np.arange(1, 13)
    ).transpose("mode", "month", "lead", "lag")

    sample_count = count_training_samples(
        monthly_scores=monthly_scores,
        training_period=training_period,
        maximum_lead_months=maximum_lead_months,
        input_sequence_months=input_sequence_months,
    )

    shrinkage = (
        sample_count
        / (shrinkage_pseudocount + sample_count)
    ).astype(np.float32)
    shrinkage.name = "target_month_shrinkage_fraction"

    blended = (
        (1.0 - shrinkage) * pooled_month
        + shrinkage * target
    )

    l1_norm = np.abs(blended).sum("lag")
    final = blended / xr.where(l1_norm > 1e-6, l1_norm, 1.0)
    final.name = "lag_weight"

    return xr.Dataset(
        {
            "lag_weight": final.astype(np.float32),
            "pooled_lag_correlation": raw["pooled"],
            "target_month_lag_correlation": raw["target_month"],
            "training_sample_count": sample_count,
            "target_month_shrinkage_fraction": shrinkage,
        },
        attrs={
            "polarity_regularization": polarity_method,
            "shrinkage_pseudocount": int(shrinkage_pseudocount),
            "input_sequence_months": int(input_sequence_months),
            "maximum_lead_months": int(maximum_lead_months),
        },
    )
