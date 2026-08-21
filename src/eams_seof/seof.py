from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def _latitude_weights(lat: xr.DataArray) -> xr.DataArray:
    return np.cos(np.deg2rad(lat)).astype(np.float32)


def compute_seof(
    training_anomaly: xr.DataArray,
    retained_modes: int = 3,
    latitude_weighting: bool = True,
) -> xr.Dataset:
    'Compute the extended 12-month SEOF decomposition.'

    x = training_anomaly.transpose("time", "lat", "lon")
    n_years = x.sizes["time"] // 12

    if n_years < 3:
        raise ValueError("At least three complete years are required.")

    x = x.isel(time=slice(0, n_years * 12))
    first_year = pd.Timestamp(x.time.values[0]).year

    xm = xr.DataArray(
        x.values.reshape(
            n_years,
            12,
            x.sizes["lat"],
            x.sizes["lon"],
        ),
        dims=("year", "month", "lat", "lon"),
        coords={
            "year": np.arange(
                first_year,
                first_year + n_years,
            ),
            "month": np.arange(1, 13),
            "lat": x.lat,
            "lon": x.lon,
        },
    )

    # Center each month-grid feature across training years.

    seof_mean = (
        xm.mean("year", skipna=True)
        .astype(np.float32)
        .rename("seof_mean")
    )

    xm_centered = (
        xm - seof_mean
    ).astype(np.float32)

    # Latitude weighting.

    if latitude_weighting:
        sqrt_weight = np.sqrt(
            _latitude_weights(x.lat)
        ).broadcast_like(
            xm.isel(year=0, month=0)
        )
    else:
        sqrt_weight = xr.ones_like(
            xm.isel(year=0, month=0),
            dtype=np.float32,
        )

    # SEOF decomposition.

    weighted = xm_centered * sqrt_weight

    stacked = weighted.stack(
        feature=("month", "lat", "lon")
    )

    matrix = np.nan_to_num(
        stacked.values,
        nan=0.0,
    )

    u, singular_values, vt = np.linalg.svd(
        matrix,
        full_matrices=False,
    )

    # A centered matrix has rank <= n_years - 1.
    n_available = min(
        n_years - 1,
        vt.shape[0],
    )
    n_keep = min(
        retained_modes,
        n_available,
    )

    # SEOF patterns.

    vt_keep = vt[:n_keep]

    weighted_pattern = xr.DataArray(
        vt_keep,
        dims=("mode", "feature"),
        coords={
            "mode": np.arange(1, n_keep + 1),
            "feature": stacked.feature,
        },
    ).unstack("feature")

    eof_patterns = (
        weighted_pattern / sqrt_weight
    ).transpose(
        "mode",
        "month",
        "lat",
        "lon",
    )

    eof_patterns.name = "eof_patterns"

    # Annual PCs.

    annual_pc_values = (
        u[:, :n_keep]
        * singular_values[:n_keep][None, :]
    )

    annual_pc = xr.DataArray(
        annual_pc_values.astype(np.float32),
        dims=("year", "mode"),
        coords={
            "year": xm.year,
            "mode": np.arange(1, n_keep + 1),
        },
        name="annual_pc",
    )

    pc_mean = annual_pc.mean("year")
    pc_std = annual_pc.std(
        "year",
        ddof=1,
    )

    annual_pc_standardized = (
        (annual_pc - pc_mean)
        / xr.where(pc_std > 0, pc_std, 1.0)
    ).astype(np.float32)

    annual_pc_standardized.name = (
        "annual_pc_standardized"
    )

    # Monthly covariance patterns for visualization.

    covariance_patterns = []

    for mode in annual_pc.mode.values:

        pc = annual_pc_standardized.sel(
            mode=mode
        )

        cov = (
            xm_centered * pc
        ).mean(
            "year"
        ).expand_dims(
            mode=[mode]
        )

        covariance_patterns.append(cov)

    covariance_patterns = xr.concat(
        covariance_patterns,
        dim="mode",
    ).transpose(
        "mode",
        "month",
        "lat",
        "lon",
    )

    covariance_patterns.name = (
        "covariance_patterns"
    )

    # Explained variance.

    variance_all = singular_values**2

    explained = (
        variance_all[:n_keep]
        / variance_all.sum()
    ).astype(np.float32)

    explained_variance = xr.DataArray(
        explained,
        dims=("mode",),
        coords={
            "mode": np.arange(1, n_keep + 1)
        },
        name="explained_variance",
    )

    ds = xr.Dataset(
        {
            "eof_patterns": (
                eof_patterns.astype(np.float32)
            ),
            "covariance_patterns": (
                covariance_patterns.astype(np.float32)
            ),
            "annual_pc": annual_pc,
            "annual_pc_standardized": (
                annual_pc_standardized
            ),
            "explained_variance": (
                explained_variance
            ),
            "seof_mean": seof_mean,
        }
    )

    ds.attrs["decomposition"] = (
        "12-month extended SEOF of year-centered "
        "detrended anomalies"
    )

    ds.attrs["latitude_weighting"] = (
        "sqrt(cos(lat))"
        if latitude_weighting
        else "none"
    )

    return ds

def align_seof(
    eof_patterns: xr.DataArray,
    normalize_each_month: bool = True,
) -> xr.DataArray:
    'Align the retained monthly SEOF patterns.'
    e = eof_patterns.transpose("mode", "month", "lat", "lon")
    w = _latitude_weights(e.lat).broadcast_like(
        e.isel(mode=0, month=0)
    )
    sqrt_w = np.sqrt(w)

    def flatten_weighted(month_patterns: xr.DataArray) -> np.ndarray:
        return (
            (month_patterns * sqrt_w)
            .fillna(0.0)
            .values.reshape(month_patterns.sizes["mode"], -1)
        )

    def normalize(month_patterns: xr.DataArray) -> xr.DataArray:
        norm = np.sqrt(
            ((month_patterns**2) * w).sum(("lat", "lon"))
        )
        norm = xr.where(norm > 0, norm, 1.0)
        return month_patterns / norm

    def align(previous: xr.DataArray, current: xr.DataArray) -> xr.DataArray:
        a = flatten_weighted(previous)
        b = flatten_weighted(current)

        cross = a @ b.T
        u, _, vt = np.linalg.svd(cross, full_matrices=False)
        rotation = u @ vt

        current_raw = current.fillna(0.0).values.reshape(
            current.sizes["mode"], -1
        )
        aligned_raw = (
            rotation @ current_raw
        ).reshape(current.shape)

        aligned = xr.DataArray(
            aligned_raw,
            dims=current.dims,
            coords=current.coords,
        )

        # Retain a consistent component sign relative to the preceding month.
        dot = ((previous * aligned) * w).sum(("lat", "lon")).values
        sign = np.sign(dot)
        sign[sign == 0] = 1.0
        aligned = aligned * xr.DataArray(
            sign,
            dims=("mode",),
            coords={"mode": current.mode},
        )

        if normalize_each_month:
            aligned = normalize(aligned)
        return aligned.astype(np.float32)

    month_energy = (
        ((e**2) * w).sum(("lat", "lon")).sum("mode")
    )
    anchor_index = int(np.argmax(month_energy.values))

    aligned_months = [None] * 12
    anchor = e.isel(month=anchor_index).drop_vars("month")
    if normalize_each_month:
        anchor = normalize(anchor)
    aligned_months[anchor_index] = anchor.astype(np.float32)

    for idx in range(anchor_index - 1, -1, -1):
        aligned_months[idx] = align(
            aligned_months[idx + 1],
            e.isel(month=idx).drop_vars("month"),
        )

    for idx in range(anchor_index + 1, 12):
        aligned_months[idx] = align(
            aligned_months[idx - 1],
            e.isel(month=idx).drop_vars("month"),
        )

    aligned_patterns = xr.concat(
        aligned_months,
        dim=xr.DataArray(
            np.arange(1, 13),
            dims=("month",),
            name="month",
        ),
    ).transpose("mode", "month", "lat", "lon")

    aligned_patterns.name = "aligned_eof_patterns"
    aligned_patterns.attrs["anchor_month"] = anchor_index + 1
    aligned_patterns.attrs["alignment"] = (
        "sqrt(cos(lat))-weighted orthogonal Procrustes"
    )
    aligned_patterns.attrs["monthwise_unit_norm"] = int(
        normalize_each_month
    )
    return aligned_patterns.astype(np.float32)


def project_seof(
    detrended_sst_anomaly: xr.DataArray,
    aligned_patterns: xr.DataArray,
    seof_mean: xr.DataArray,
) -> xr.DataArray:
    'Project monthly SST anomalies onto the forecast basis.'

    x = detrended_sst_anomaly.transpose(
        "time", "lat", "lon"
    )

    patterns = aligned_patterns.transpose(
        "mode", "month", "lat", "lon"
    )

    mean = seof_mean.transpose(
        "month", "lat", "lon"
    )

    w = _latitude_weights(
        x.lat
    ).values[:, None]

    month_index = pd.DatetimeIndex(
        x.time.values
    ).month

    values = np.full(
        (
            x.sizes["time"],
            patterns.sizes["mode"],
        ),
        np.nan,
        dtype=np.float32,
    )

    xv = x.values
    ev = patterns.values
    meanv = mean.values

    for it in range(x.sizes["time"]):

        im = int(month_index[it]) - 1

        field = xv[it] - meanv[im]

        for ik in range(
            patterns.sizes["mode"]
        ):
            pattern = ev[ik, im]

            values[it, ik] = np.nansum(
                field * pattern * w
            )

    return xr.DataArray(
        values,
        coords={
            "time": x.time,
            "mode": patterns.mode,
        },
        dims=("time", "mode"),
        name="monthly_projection_score",
    )


def weighted_norms(
    aligned_patterns: xr.DataArray,
) -> xr.DataArray:
    w = _latitude_weights(aligned_patterns.lat).broadcast_like(
        aligned_patterns.isel(mode=0, month=0)
    )
    return np.sqrt(
        ((aligned_patterns**2) * w).sum(("lat", "lon"))
    ).transpose("month", "mode")


def weighted_gram(
    aligned_patterns: xr.DataArray,
) -> xr.DataArray:
    w = _latitude_weights(aligned_patterns.lat).broadcast_like(
        aligned_patterns.isel(mode=0, month=0)
    )

    mats = []
    for month in aligned_patterns.month.values:
        e = aligned_patterns.sel(month=month)
        vals = []
        for i in e.mode.values:
            row = []
            for j in e.mode.values:
                row.append(
                    float(
                        (
                            e.sel(mode=i)
                            * e.sel(mode=j)
                            * w
                        ).sum(("lat", "lon"))
                    )
                )
            vals.append(row)

        mats.append(
            xr.DataArray(
                np.asarray(vals, dtype=np.float32),
                dims=("mode_i", "mode_j"),
                coords={
                    "mode_i": e.mode.values,
                    "mode_j": e.mode.values,
                },
            ).expand_dims(month=[month])
        )

    return xr.concat(mats, dim="month")
