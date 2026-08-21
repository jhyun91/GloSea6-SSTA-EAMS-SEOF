from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BASE_FONT_SIZE = 10
SMALL_FONT = BASE_FONT_SIZE - 4
MID_FONT = BASE_FONT_SIZE - 2.5

SEASON_MONTHS = {
    "JFM": [1, 2, 3],
    "AMJ": [4, 5, 6],
    "JAS": [7, 8, 9],
    "OND": [10, 11, 12],
}

LEAD_GROUPS = {
    "1-3": [1, 2, 3],
    "4-6": [4, 5, 6],
    "7-12": [7, 8, 9, 10, 11, 12],
}

MONTH_NAMES = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]


def _valid(forecast, observed, min_samples=3):
    f = np.asarray(forecast, dtype=float)
    o = np.asarray(observed, dtype=float)
    ok = np.isfinite(f) & np.isfinite(o)
    if ok.sum() < min_samples:
        return None, None
    return f[ok], o[ok]


def _valid_three(first, second, observed, min_samples=3):
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    o = np.asarray(observed, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(o)
    if ok.sum() < min_samples:
        return None, None, None
    return a[ok], b[ok], o[ok]


def acc(forecast, observed, min_samples=3):
    """Uncentered anomaly correlation coefficient."""
    f, o = _valid(forecast, observed, min_samples)
    if f is None:
        return np.nan
    denom = np.sqrt(np.sum(f ** 2) * np.sum(o ** 2))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.sum(f * o) / denom)


def acc_centered(forecast, observed, min_samples=3):
    """Pearson correlation after removing verification-period means."""
    f, o = _valid(forecast, observed, min_samples)
    if f is None:
        return np.nan
    f = f - np.mean(f)
    o = o - np.mean(o)
    denom = np.sqrt(np.sum(f ** 2) * np.sum(o ** 2))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.sum(f * o) / denom)


def msss(forecast, observed, min_samples=3):
    """Mean squared skill score against the zero-anomaly reference."""
    f, o = _valid(forecast, observed, min_samples)
    if f is None:
        return np.nan
    mse_ref = np.mean(o ** 2)
    if not np.isfinite(mse_ref) or mse_ref <= 0:
        return np.nan
    return float(1.0 - np.mean((f - o) ** 2) / mse_ref)


def metrics(forecast, observed, min_samples=3):
    f, o = _valid(forecast, observed, min_samples)
    if f is None:
        return {
            "ACC": np.nan,
            "CENTERED_ACC": np.nan,
            "MSSS": np.nan,
            "BIAS": np.nan,
            "RMSE": np.nan,
            "STD_RATIO": np.nan,
            "N": np.nan,
        }

    obs_std = np.std(o, ddof=1)
    return {
        "ACC": acc(f, o, min_samples=min_samples),
        "CENTERED_ACC": acc_centered(f, o, min_samples=min_samples),
        "MSSS": msss(f, o, min_samples=min_samples),
        "BIAS": float(np.mean(f - o)),
        "RMSE": float(np.sqrt(np.mean((f - o) ** 2))),
        "STD_RATIO": float(np.std(f, ddof=1) / obs_std) if obs_std > 0 else np.nan,
        "N": int(f.size),
    }


def _bootstrap(forecast, observed, metric, n_resamples, rng):
    f = np.asarray(forecast, dtype=float)
    o = np.asarray(observed, dtype=float)
    n = f.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    fb = f[idx]
    ob = o[idx]

    if metric in {"ACC", "CENTERED_ACC"}:
        if metric == "CENTERED_ACC":
            fb = fb - np.mean(fb, axis=1, keepdims=True)
            ob = ob - np.mean(ob, axis=1, keepdims=True)
        denom = np.sqrt(np.sum(fb ** 2, axis=1) * np.sum(ob ** 2, axis=1))
        return np.divide(
            np.sum(fb * ob, axis=1),
            denom,
            out=np.full(n_resamples, np.nan, dtype=float),
            where=np.isfinite(denom) & (denom > 0),
        )

    mse_ref = np.mean(ob ** 2, axis=1)
    mse_fcst = np.mean((fb - ob) ** 2, axis=1)
    ratio = np.divide(
        mse_fcst,
        mse_ref,
        out=np.full(n_resamples, np.nan, dtype=float),
        where=np.isfinite(mse_ref) & (mse_ref > 0),
    )
    return 1.0 - ratio


def bootstrap(
    forecast,
    observed,
    metric="ACC",
    n_boot=5000,
    min_samples=10,
    min_boot_valid=500,
    ci_level=0.95,
    seed=42,
):
    """Return statistic, one-sided bootstrap p-value, percentile CI, and N."""
    f, o = _valid(forecast, observed, min_samples)
    if f is None:
        return np.nan, np.nan, np.nan, np.nan, 0

    metric = metric.upper()
    if metric not in {"ACC", "CENTERED_ACC", "MSSS"}:
        raise ValueError("metric must be 'ACC', 'CENTERED_ACC', or 'MSSS'.")

    if metric == "ACC":
        stat = acc(f, o, min_samples=min_samples)
    elif metric == "CENTERED_ACC":
        stat = acc_centered(f, o, min_samples=min_samples)
    else:
        stat = msss(f, o, min_samples=min_samples)
    if not np.isfinite(stat):
        return stat, np.nan, np.nan, np.nan, int(f.size)

    rng = np.random.default_rng(seed)
    boot = _bootstrap(f, o, metric, n_boot, rng)
    boot = boot[np.isfinite(boot)]
    if boot.size < min_boot_valid:
        return stat, np.nan, np.nan, np.nan, int(f.size)

    tail = (1.0 - ci_level) / 2.0
    ci_low, ci_high = np.quantile(boot, [tail, 1.0 - tail])
    p_boot = (1.0 + np.sum(boot <= 0.0)) / (1.0 + boot.size)
    return stat, float(p_boot), float(ci_low), float(ci_high), int(f.size)


def bh_fdr(pvalues, alpha=0.10):
    p = np.asarray(pvalues, dtype=float)
    reject = np.zeros(p.shape, dtype=bool)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return reject, adjusted

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = ranked.size
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    inv = np.empty_like(order)
    inv[order] = np.arange(m)
    adjusted[valid] = q[inv]
    reject[valid] = adjusted[valid] <= alpha
    return reject, adjusted


def _align_mask(mask: xr.DataArray, template: xr.DataArray) -> xr.DataArray:
    mask = mask.squeeze(drop=True)
    if not {"lat", "lon"}.issubset(mask.dims):
        raise ValueError("Region mask must contain 'lat' and 'lon' dimensions.")

    same_grid = (
        mask.sizes.get("lat") == template.sizes.get("lat")
        and mask.sizes.get("lon") == template.sizes.get("lon")
        and np.array_equal(mask.lat.values, template.lat.values)
        and np.array_equal(mask.lon.values, template.lon.values)
    )
    if same_grid:
        aligned = mask
    else:
        aligned = mask.astype(np.float32).interp(lat=template.lat, lon=template.lon, method="nearest")
    return aligned.fillna(0) > 0.5


def region_mean(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    template = da.isel({d: 0 for d in da.dims if d not in ("lat", "lon")}, drop=True)
    mask = _align_mask(mask, template)
    weights = xr.DataArray(np.cos(np.deg2rad(da.lat)), coords={"lat": da.lat}, dims=("lat",))
    return da.where(mask).weighted(weights).mean(("lat", "lon"))


def load_masks(mask_ds: xr.Dataset, template: xr.DataArray) -> dict[str, xr.DataArray]:
    """Return manuscript masks with EAMS reconstructed as YS | ECS | EJS."""
    subregions = {
        "YS": _align_mask(mask_ds["region_mask_ys"], template),
        "ECS": _align_mask(mask_ds["region_mask_ecs"], template),
        "EJS": _align_mask(mask_ds["region_mask_ejs"], template),
    }
    eams = subregions["YS"] | subregions["ECS"] | subregions["EJS"]

    if "region_mask_eams" in mask_ds:
        stored = _align_mask(mask_ds["region_mask_eams"], template)
        if not np.array_equal(eams.values, stored.values):
            raise ValueError("region_mask_eams does not equal YS | ECS | EJS.")

    return {"EAMS": eams, **subregions}


def _target_time(da: xr.DataArray) -> xr.DataArray:
    if "target_time" in da.dims:
        return da
    if "time" in da.dims:
        return da.rename(time="target_time")
    raise ValueError("DataArray must contain 'target_time' or 'time'.")


def _score_matrix(
    forecast: xr.DataArray,
    observed: xr.DataArray,
    masks: Mapping[str, xr.DataArray],
    grouping: str,
    min_samples: int,
    n_boot: int,
    min_boot_valid: int,
    ci_level: float,
    fdr_q: float,
    fdr_scope: str,
    seed: int,
) -> xr.Dataset:
    leads = forecast.lead.values.astype(int)
    regions = list(masks)
    groups = np.arange(1, 13) if grouping == "month" else np.array(list(SEASON_MONTHS))
    group_dim = "month" if grouping == "month" else "season"

    shape = (len(regions), len(leads), len(groups))
    coords = {"region": regions, "lead": leads, group_dim: groups}
    dims = ("region", "lead", group_dim)

    fields = {
        name: xr.DataArray(np.full(shape, np.nan, dtype=np.float32), coords=coords, dims=dims)
        for name in [
            "ACC", "CENTERED_ACC", "MSSS", "BIAS", "RMSE", "STD_RATIO", "N",
            "P_ACC_RAW", "P_CENTERED_ACC_RAW", "P_MSSS_RAW",
            "ACC_CI_LO", "ACC_CI_HI",
            "CENTERED_ACC_CI_LO", "CENTERED_ACC_CI_HI",
            "MSSS_CI_LO", "MSSS_CI_HI",
        ]
    }

    for ir, region in enumerate(regions):
        f_reg = region_mean(forecast, masks[region])
        o_reg = region_mean(observed, masks[region])

        for il, lead in enumerate(leads):
            f = f_reg.sel(lead=lead)
            f, o = xr.align(f, o_reg, join="inner")

            for ig, group in enumerate(groups):
                if grouping == "month":
                    ff = f.where(f.target_time.dt.month == int(group), drop=True)
                    oo = o.where(o.target_time.dt.month == int(group), drop=True)
                else:
                    months = SEASON_MONTHS[str(group)]
                    ff = f.where(f.target_time.dt.month.isin(months), drop=True)
                    oo = o.where(o.target_time.dt.month.isin(months), drop=True)
                    ff_count = ff.resample(target_time="YS").count()
                    oo_count = oo.resample(target_time="YS").count()
                    ff = ff.resample(target_time="YS").mean().where(ff_count >= 3)
                    oo = oo.resample(target_time="YS").mean().where(oo_count >= 3)

                ff, oo = xr.align(ff, oo, join="inner")
                valid = np.isfinite(ff.values) & np.isfinite(oo.values)
                fv = ff.values[valid]
                ov = oo.values[valid]
                if fv.size < min_samples:
                    continue

                loc = {"region": region, "lead": lead, group_dim: group}
                scores = metrics(fv, ov, min_samples=min_samples)
                for name in ["ACC", "CENTERED_ACC", "MSSS", "BIAS", "RMSE", "STD_RATIO", "N"]:
                    fields[name].loc[loc] = scores[name]

                acc = bootstrap(
                    fv, ov, metric="ACC", n_boot=n_boot, min_samples=min_samples,
                    min_boot_valid=min_boot_valid, ci_level=ci_level,
                    seed=seed + 10000 * ir + 100 * il + ig,
                )
                cacc = bootstrap(
                    fv, ov, metric="CENTERED_ACC", n_boot=n_boot, min_samples=min_samples,
                    min_boot_valid=min_boot_valid, ci_level=ci_level,
                    seed=seed + 50000 + 10000 * ir + 100 * il + ig,
                )
                ms = bootstrap(
                    fv, ov, metric="MSSS", n_boot=n_boot, min_samples=min_samples,
                    min_boot_valid=min_boot_valid, ci_level=ci_level,
                    seed=seed + 100000 + 10000 * ir + 100 * il + ig,
                )
                fields["P_ACC_RAW"].loc[loc] = acc[1]
                fields["ACC_CI_LO"].loc[loc] = acc[2]
                fields["ACC_CI_HI"].loc[loc] = acc[3]
                fields["P_CENTERED_ACC_RAW"].loc[loc] = cacc[1]
                fields["CENTERED_ACC_CI_LO"].loc[loc] = cacc[2]
                fields["CENTERED_ACC_CI_HI"].loc[loc] = cacc[3]
                fields["P_MSSS_RAW"].loc[loc] = ms[1]
                fields["MSSS_CI_LO"].loc[loc] = ms[2]
                fields["MSSS_CI_HI"].loc[loc] = ms[3]

    for metric in ["ACC", "CENTERED_ACC", "MSSS"]:
        raw = fields[f"P_{metric}_RAW"]
        adj = xr.full_like(raw, np.nan)
        sig = xr.zeros_like(raw, dtype=bool)

        if fdr_scope in {"region", "subregion"}:
            for ir in range(raw.sizes["region"]):
                reject, values = bh_fdr(raw.isel(region=ir).values.ravel(), alpha=fdr_q)
                adj.values[ir] = values.reshape(raw.isel(region=ir).shape)
                sig.values[ir] = reject.reshape(raw.isel(region=ir).shape)
        elif fdr_scope == "global":
            reject, values = bh_fdr(raw.values.ravel(), alpha=fdr_q)
            adj.values[:] = values.reshape(raw.shape)
            sig.values[:] = reject.reshape(raw.shape)
        else:
            raise ValueError("fdr_scope must be 'region', 'subregion', or 'global'.")

        fields[f"P_{metric}_FDR"] = adj.astype(np.float32)
        fields[f"SIG_{metric}"] = sig & np.isfinite(fields[metric]) & (fields[metric] > 0)

    return xr.Dataset(fields)


def evaluate_skill(
    forecast: xr.DataArray,
    observed: xr.DataArray,
    masks: Mapping[str, xr.DataArray],
    verification_period=("2013-01-01", "2025-12-31"),
    min_samples=10,
    n_boot=5000,
    min_boot_valid=500,
    ci_level=0.95,
    fdr_q=0.10,
    target_month_fdr_scope="region",
    seasonal_fdr_scope="subregion",
    seed=42,
) -> xr.Dataset:
    forecast = _target_time(forecast).sel(target_time=slice(*verification_period))
    observed = _target_time(observed).sel(target_time=slice(*verification_period))
    forecast, observed = xr.align(forecast, observed, join="inner")

    lead_records = []
    for region, mask in masks.items():
        f_reg = region_mean(forecast, mask)
        o_reg = region_mean(observed, mask)
        rows = []
        for lead in forecast.lead.values.astype(int):
            f, o = xr.align(f_reg.sel(lead=lead), o_reg, join="inner")
            rows.append(metrics(f.values, o.values, min_samples=3))
        lead_records.append(
            xr.Dataset(
                {
                    name: ("lead", np.array([row[name] for row in rows], dtype=np.float32))
                    for name in ["ACC", "CENTERED_ACC", "MSSS", "BIAS", "RMSE", "STD_RATIO", "N"]
                },
                coords={"lead": forecast.lead.values.astype(int)},
            ).expand_dims(region=[region])
        )
    lead_ds = xr.concat(lead_records, dim="region")

    target = _score_matrix(
        forecast, observed, masks, "month", min_samples, n_boot,
        min_boot_valid, ci_level, fdr_q, target_month_fdr_scope, seed,
    )
    seasonal = _score_matrix(
        forecast, observed, masks, "season", min_samples, n_boot,
        min_boot_valid, ci_level, fdr_q, seasonal_fdr_scope, seed + 200000,
    )

    output = xr.Dataset()
    for name, da in lead_ds.data_vars.items():
        output[f"LEAD_{name}"] = da

    for prefix, ds in [("TARGET", target), ("SEASON", seasonal)]:
        for name, da in ds.data_vars.items():
            output[f"{prefix}_{name}"] = da

    output.attrs.update(
        verification_start=str(verification_period[0]),
        verification_end=str(verification_period[1]),
        bootstrap_samples=int(n_boot),
        minimum_samples=int(min_samples),
        minimum_bootstrap_valid=int(min_boot_valid),
        confidence_level=float(ci_level),
        fdr_q=float(fdr_q),
        target_month_fdr_scope=str(target_month_fdr_scope),
        seasonal_fdr_scope=str(seasonal_fdr_scope),
    )
    return output


def evaluate_configs(
    forecasts: Mapping[str, xr.DataArray],
    observed: xr.DataArray,
    masks: Mapping[str, xr.DataArray],
    **kwargs,
) -> xr.Dataset:
    datasets = [
        evaluate_skill(fcst, observed, masks, **kwargs).expand_dims(configuration=[name])
        for name, fcst in forecasts.items()
    ]
    return xr.concat(datasets, dim="configuration")



def build_anomaly_persistence_forecast(
    observed_anomaly: xr.DataArray,
    forecast_template: xr.DataArray,
) -> xr.DataArray:
    """Construct a no-change anomaly-persistence forecast on a forecast target grid.

    For target time ``t`` and lead ``L`` (months), the persistence forecast is
    the observed SST anomaly at ``t - L``.  Lead 0 is therefore identical to
    the verifying anomaly and should normally be excluded when assessing
    forecast skill relative to persistence.
    """
    if "time" not in observed_anomaly.dims:
        if "target_time" in observed_anomaly.dims:
            observed_anomaly = observed_anomaly.rename(target_time="time")
        else:
            raise ValueError("observed_anomaly must contain 'time' or 'target_time'.")

    template = _target_time(forecast_template)
    if "lead" not in template.dims:
        raise ValueError("forecast_template must contain a 'lead' dimension.")

    target_times = pd.DatetimeIndex(template.target_time.values)
    pieces = []
    for lead in template.lead.values.astype(int):
        source_times = target_times - pd.DateOffset(months=int(lead))
        source = observed_anomaly.reindex(time=source_times)
        source = source.rename(time="target_time").assign_coords(
            target_time=template.target_time.values
        )
        pieces.append(source.expand_dims(lead=[int(lead)]))

    out = xr.concat(pieces, dim="lead").transpose(
        "lead", "target_time", "lat", "lon"
    )
    out.name = "sst_anomaly_persistence_forecast"
    out.attrs.update(
        definition="Persistence at target time t and lead L equals observed anomaly at t-L months.",
        lead0_note="Lead 0 equals the verifying anomaly and is not a meaningful persistence benchmark.",
    )
    return out.astype(np.float32)


def persistence_skill_score(model_forecast, persistence_forecast, observed, min_samples=3):
    """MSE skill score of the model forecast relative to anomaly persistence."""
    m, p, o = _valid_three(model_forecast, persistence_forecast, observed, min_samples)
    if m is None:
        return np.nan
    mse_persistence = np.mean((p - o) ** 2)
    if not np.isfinite(mse_persistence) or mse_persistence <= 0:
        return np.nan
    mse_model = np.mean((m - o) ** 2)
    return float(1.0 - mse_model / mse_persistence)


def _bootstrap_persistence_skill(model_forecast, persistence_forecast, observed, n_resamples, rng):
    m = np.asarray(model_forecast, dtype=float)
    p = np.asarray(persistence_forecast, dtype=float)
    o = np.asarray(observed, dtype=float)
    n = m.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    mb = m[idx]
    pb = p[idx]
    ob = o[idx]

    mse_model = np.mean((mb - ob) ** 2, axis=1)
    mse_persistence = np.mean((pb - ob) ** 2, axis=1)
    ratio = np.divide(
        mse_model,
        mse_persistence,
        out=np.full(n_resamples, np.nan, dtype=float),
        where=np.isfinite(mse_persistence) & (mse_persistence > 0),
    )
    return 1.0 - ratio


def bootstrap_persistence_skill(
    model_forecast,
    persistence_forecast,
    observed,
    n_boot=5000,
    min_samples=10,
    min_boot_valid=500,
    ci_level=0.95,
    seed=42,
):
    """Paired bootstrap for model MSE skill relative to anomaly persistence."""
    m, p, o = _valid_three(
        model_forecast, persistence_forecast, observed, min_samples
    )
    if m is None:
        return np.nan, np.nan, np.nan, np.nan, 0

    stat = persistence_skill_score(m, p, o, min_samples=min_samples)
    if not np.isfinite(stat):
        return stat, np.nan, np.nan, np.nan, int(m.size)

    rng = np.random.default_rng(seed)
    boot = _bootstrap_persistence_skill(m, p, o, n_boot, rng)
    boot = boot[np.isfinite(boot)]
    if boot.size < min_boot_valid:
        return stat, np.nan, np.nan, np.nan, int(m.size)

    tail = (1.0 - ci_level) / 2.0
    ci_low, ci_high = np.quantile(boot, [tail, 1.0 - tail])
    p_boot = (1.0 + np.sum(boot <= 0.0)) / (1.0 + boot.size)
    return stat, float(p_boot), float(ci_low), float(ci_high), int(m.size)


def _persistence_score_matrix(
    model_forecast: xr.DataArray,
    persistence_forecast: xr.DataArray,
    observed: xr.DataArray,
    masks: Mapping[str, xr.DataArray],
    grouping: str,
    min_samples: int,
    n_boot: int,
    min_boot_valid: int,
    ci_level: float,
    fdr_q: float,
    fdr_scope: str,
    seed: int,
) -> xr.Dataset:
    leads = model_forecast.lead.values.astype(int)
    regions = list(masks)
    groups = np.arange(1, 13) if grouping == "month" else np.array(list(SEASON_MONTHS))
    group_dim = "month" if grouping == "month" else "season"

    shape = (len(regions), len(leads), len(groups))
    coords = {"region": regions, "lead": leads, group_dim: groups}
    dims = ("region", "lead", group_dim)
    names = [
        "MODEL_ACC", "PERSISTENCE_ACC", "DELTA_ACC", "PSS",
        "MODEL_RMSE", "PERSISTENCE_RMSE", "N",
        "P_PSS_RAW", "PSS_CI_LO", "PSS_CI_HI",
    ]
    fields = {
        name: xr.DataArray(
            np.full(shape, np.nan, dtype=np.float32), coords=coords, dims=dims
        )
        for name in names
    }

    for ir, region in enumerate(regions):
        m_reg = region_mean(model_forecast, masks[region])
        p_reg = region_mean(persistence_forecast, masks[region])
        o_reg = region_mean(observed, masks[region])

        for il, lead in enumerate(leads):
            m = m_reg.sel(lead=lead)
            p = p_reg.sel(lead=lead)
            m, p, o = xr.align(m, p, o_reg, join="inner")

            for ig, group in enumerate(groups):
                if grouping == "month":
                    selector = m.target_time.dt.month == int(group)
                    mm = m.where(selector, drop=True)
                    pp = p.where(selector, drop=True)
                    oo = o.where(selector, drop=True)
                else:
                    months = SEASON_MONTHS[str(group)]
                    mm = m.where(m.target_time.dt.month.isin(months), drop=True)
                    pp = p.where(p.target_time.dt.month.isin(months), drop=True)
                    oo = o.where(o.target_time.dt.month.isin(months), drop=True)
                    mm_count = mm.resample(target_time="YS").count()
                    pp_count = pp.resample(target_time="YS").count()
                    oo_count = oo.resample(target_time="YS").count()
                    mm = mm.resample(target_time="YS").mean().where(mm_count >= 3)
                    pp = pp.resample(target_time="YS").mean().where(pp_count >= 3)
                    oo = oo.resample(target_time="YS").mean().where(oo_count >= 3)

                mm, pp, oo = xr.align(mm, pp, oo, join="inner")
                valid = (
                    np.isfinite(mm.values)
                    & np.isfinite(pp.values)
                    & np.isfinite(oo.values)
                )
                mv = mm.values[valid]
                pv = pp.values[valid]
                ov = oo.values[valid]
                if mv.size < min_samples:
                    continue

                loc = {"region": region, "lead": lead, group_dim: group}
                model_acc = acc(mv, ov, min_samples=min_samples)
                persistence_acc = acc(pv, ov, min_samples=min_samples)
                pss = persistence_skill_score(
                    mv, pv, ov, min_samples=min_samples
                )
                fields["MODEL_ACC"].loc[loc] = model_acc
                fields["PERSISTENCE_ACC"].loc[loc] = persistence_acc
                fields["DELTA_ACC"].loc[loc] = model_acc - persistence_acc
                fields["PSS"].loc[loc] = pss
                fields["MODEL_RMSE"].loc[loc] = np.sqrt(np.mean((mv - ov) ** 2))
                fields["PERSISTENCE_RMSE"].loc[loc] = np.sqrt(np.mean((pv - ov) ** 2))
                fields["N"].loc[loc] = mv.size

                boot = bootstrap_persistence_skill(
                    mv, pv, ov,
                    n_boot=n_boot,
                    min_samples=min_samples,
                    min_boot_valid=min_boot_valid,
                    ci_level=ci_level,
                    seed=seed + 10000 * ir + 100 * il + ig,
                )
                fields["P_PSS_RAW"].loc[loc] = boot[1]
                fields["PSS_CI_LO"].loc[loc] = boot[2]
                fields["PSS_CI_HI"].loc[loc] = boot[3]

    raw = fields["P_PSS_RAW"]
    adj = xr.full_like(raw, np.nan)
    sig = xr.zeros_like(raw, dtype=bool)
    if fdr_scope in {"region", "subregion"}:
        for ir in range(raw.sizes["region"]):
            reject, values = bh_fdr(
                raw.isel(region=ir).values.ravel(), alpha=fdr_q
            )
            adj.values[ir] = values.reshape(raw.isel(region=ir).shape)
            sig.values[ir] = reject.reshape(raw.isel(region=ir).shape)
    elif fdr_scope == "global":
        reject, values = bh_fdr(raw.values.ravel(), alpha=fdr_q)
        adj.values[:] = values.reshape(raw.shape)
        sig.values[:] = reject.reshape(raw.shape)
    else:
        raise ValueError("fdr_scope must be 'region', 'subregion', or 'global'.")

    fields["P_PSS_FDR"] = adj.astype(np.float32)
    fields["SIG_PSS"] = sig & np.isfinite(fields["PSS"]) & (fields["PSS"] > 0)
    return xr.Dataset(fields)


def evaluate_persistence_benchmark(
    model_forecast: xr.DataArray,
    persistence_forecast: xr.DataArray,
    observed: xr.DataArray,
    masks: Mapping[str, xr.DataArray],
    verification_period=("2013-01-01", "2025-12-31"),
    min_samples=10,
    n_boot=5000,
    min_boot_valid=500,
    ci_level=0.95,
    fdr_q=0.10,
    target_month_fdr_scope="region",
    seasonal_fdr_scope="subregion",
    seed=42,
) -> xr.Dataset:
    """Evaluate model skill relative to a no-change SST-anomaly persistence forecast."""
    model_forecast = _target_time(model_forecast).sel(
        target_time=slice(*verification_period)
    )
    persistence_forecast = _target_time(persistence_forecast).sel(
        target_time=slice(*verification_period)
    )
    observed = _target_time(observed).sel(target_time=slice(*verification_period))
    model_forecast, persistence_forecast, observed = xr.align(
        model_forecast, persistence_forecast, observed, join="inner"
    )

    target = _persistence_score_matrix(
        model_forecast, persistence_forecast, observed, masks, "month",
        min_samples, n_boot, min_boot_valid, ci_level, fdr_q,
        target_month_fdr_scope, seed,
    )
    seasonal = _persistence_score_matrix(
        model_forecast, persistence_forecast, observed, masks, "season",
        min_samples, n_boot, min_boot_valid, ci_level, fdr_q,
        seasonal_fdr_scope, seed + 200000,
    )

    output = xr.Dataset()
    for prefix, ds in [("TARGET", target), ("SEASON", seasonal)]:
        for name, da in ds.data_vars.items():
            output[f"{prefix}_{name}"] = da

    output.attrs.update(
        verification_start=str(verification_period[0]),
        verification_end=str(verification_period[1]),
        persistence_definition="forecast(t, L) = observed_anomaly(t - L)",
        lead0_note="PSS is undefined at lead 0 because persistence equals observation.",
        bootstrap_samples=int(n_boot),
        confidence_level=float(ci_level),
        fdr_q=float(fdr_q),
    )
    return output


def set_plot_style():
    import matplotlib.patheffects as path_effects
    import proplot as pplt

    pplt.rc.update({
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "savefig.transparent": False,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": SMALL_FONT,
        "axes.labelsize": SMALL_FONT,
        "xtick.labelsize": SMALL_FONT,
        "ytick.labelsize": SMALL_FONT,
        "legend.fontsize": SMALL_FONT,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        "tick.len": 2.5,
    })

    return {
        "halo_effects": [path_effects.withStroke(linewidth=3, foreground="w")],
        "text_kw": {
            "fontsize": SMALL_FONT - 2,
            "fontweight": "normal",
            "color": "black",
            "ha": "center",
            "va": "center",
        },
    }


def plot_monthly_skill(skill_ds, configuration="uncorrected", region="EAMS", savepath=None, exclude_lead0=True):
    import cmaps
    import matplotlib as mpl
    import matplotlib.ticker as mticker
    import proplot as pplt

    set_plot_style()

    ds = skill_ds.sel(configuration=configuration) if "configuration" in skill_ds.dims else skill_ds
    if exclude_lead0:
        ds = ds.sel(lead=ds.lead > 0)
    acc = ds["TARGET_ACC"].sel(region=region).transpose("lead", "month").values
    ms = ds["TARGET_MSSS"].sel(region=region).transpose("lead", "month").values
    sig_acc = ds["TARGET_SIG_ACC"].sel(region=region).transpose("lead", "month").values.astype(bool)
    sig_ms = ds["TARGET_SIG_MSSS"].sel(region=region).transpose("lead", "month").values.astype(bool)

    leads = ds.lead.values.astype(int)
    months = ds.month.values.astype(int)

    base_cmap = cmaps.sunshine_diff_12lev
    levels = [-0.2, 0.2, 0.4, 0.6, 0.8]
    colors = base_cmap(np.linspace(0, 1, 11))
    cmap = mpl.colors.ListedColormap([[1, 1, 1, 1], colors[6], colors[7], colors[8]])
    cmap.set_under(colors[0])
    cmap.set_over(colors[10])
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)

    x_edges = np.arange(0.5, 13.5, 1.0)
    y_edges = np.arange(leads.min() - 0.5, leads.max() + 1.5, 1.0)

    fig, axs = pplt.subplots(nrows=1, ncols=2, figwidth="14.0cm", share=False, wspace=4)
    axs.format(grid=False, ec="black", lw=1.0)

    meshes = []
    for ax, values, sig, title in zip(
        axs,
        [acc, ms],
        [sig_acc, sig_ms],
        [r"$\mathbf{(a)}$ Anomaly Correlation Coefficient", r"$\mathbf{(b)}$ Mean Squared Skill Score"],
    ):
        mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(values), cmap=cmap, norm=norm, extend="both", shading="flat")
        ii, jj = np.where(sig & np.isfinite(values))
        ax.scatter(months[jj], leads[ii], s=7, c="k", zorder=5)
        ax.format(title=title, titleloc="l")
        meshes.append(mesh)

    axs.format(
        xlim=(0.5, 12.5),
        ylim=(leads.min() - 0.5, leads.max() + 0.5),
        xticks=months,
        xticklabels=MONTH_NAMES,
        xrotation=45,
        yticks=leads[::2],
        xlabel="Target Month",
        ylabel="Lead Time (months)",
        xlabelsize=SMALL_FONT,
        ylabelsize=SMALL_FONT,
        ticklabelsize=SMALL_FONT,
    )

    for ax in axs:
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False)

    colorbars = [
        fig.colorbar(meshes[0], col=(1, 1), loc="b", label="ACC", ticklabelsize=SMALL_FONT, labelsize=SMALL_FONT, length=0.82, extendsize=0.5, pad=0.40, width=0.10),
        fig.colorbar(meshes[1], col=(2, 2), loc="b", label="MSSS", ticklabelsize=SMALL_FONT, labelsize=SMALL_FONT, length=0.82, extendsize=0.5, pad=0.40, width=0.10),
    ]
    for cbar in colorbars:
        cbar.ax.xaxis.set_minor_locator(mticker.NullLocator())
        cbar.ax.yaxis.set_minor_locator(mticker.NullLocator())
        cbar.ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.save(savepath, dpi=600)

    return fig, axs


def plot_regional_skill(skill_ds, configuration="uncorrected", regions=("EAMS", "YS", "ECS", "EJS"), savepath=None, exclude_lead0=True):
    import cmaps
    import matplotlib as mpl
    import matplotlib.patches as patches
    import matplotlib.ticker as mticker
    import proplot as pplt

    set_plot_style()
    ds = skill_ds.sel(configuration=configuration) if "configuration" in skill_ds.dims else skill_ds
    if exclude_lead0:
        ds = ds.sel(lead=ds.lead > 0)

    region_order = list(regions)
    season_order = ["JFM", "AMJ", "JAS", "OND"]
    leads = ds.lead.values.astype(int)

    base_cmap = cmaps.sunshine_diff_12lev
    levels = [-0.2, 0.2, 0.4, 0.6, 0.8]
    colors = base_cmap(np.linspace(0, 1, 11))
    cmap = mpl.colors.ListedColormap([[1, 1, 1, 1], colors[6], colors[7], colors[8]])
    cmap.set_under(colors[0])
    cmap.set_over(colors[10])
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)

    tri_verts = {
        "JFM": lambda x, y: [(x, y + 1), (x + 1, y + 1), (x + 0.5, y + 0.5)],
        "AMJ": lambda x, y: [(x + 1, y + 1), (x + 1, y), (x + 0.5, y + 0.5)],
        "JAS": lambda x, y: [(x, y), (x + 1, y), (x + 0.5, y + 0.5)],
        "OND": lambda x, y: [(x, y + 1), (x, y), (x + 0.5, y + 0.5)],
    }
    dot_pos = {
        "JFM": (0.50, 0.78),
        "AMJ": (0.78, 0.50),
        "JAS": (0.50, 0.22),
        "OND": (0.22, 0.50),
    }

    fig, axs = pplt.subplots(
        nrows=2,
        ncols=1,
        figwidth="14.5cm",
        figheight="10.0cm",
        share=False,
        hspace=2.3,
    )
    axs.format(grid=False, ec="black", lw=1.0)

    def draw_panel(ax, metric, title):
        values = ds[f"SEASON_{metric}"].sel(region=region_order)
        sig = ds[f"SEASON_SIG_{metric}"].sel(region=region_order)

        for ir, region in enumerate(region_order):
            y = len(region_order) - 1 - ir
            for il, lead in enumerate(leads):
                x = il
                for season in season_order:
                    value = float(values.sel(region=region, lead=lead, season=season))
                    color = cmap(norm(value)) if np.isfinite(value) else "white"
                    poly = patches.Polygon(
                        tri_verts[season](x, y),
                        closed=True,
                        fc=color,
                        ec="0.99",
                        lw=0.8,
                        zorder=1,
                    )
                    ax.add_patch(poly)
                    if bool(sig.sel(region=region, lead=lead, season=season)):
                        dx, dy = dot_pos[season]
                        ax.scatter(x + dx, y + dy, s=7, c="k", zorder=5)

                ax.add_patch(
                    patches.Rectangle((x, y), 1, 1, fill=False, ec="0.35", lw=0.6, zorder=2)
                )

        ax.format(
            title=title,
            titleloc="l",
            xlim=(0, len(leads)),
            ylim=(0, len(region_order)),
            xticks=np.arange(len(leads)) + 0.5,
            xticklabels=[str(v) for v in leads],
            yticks=np.arange(len(region_order)) + 0.5,
            yticklabels=region_order[::-1],
            xlabel="Lead (Months)",
            ticklabelsize=SMALL_FONT,
            grid=False,
        )
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False)

    draw_panel(axs[0], "ACC", r"$\mathbf{(a)}$ ACC")
    draw_panel(axs[1], "MSSS", r"$\mathbf{(b)}$ MSSS")

    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = fig.colorbar(
        scalar,
        loc="r",
        rows=(1, 2),
        ticks=levels,
        label="Skill Score",
        length=0.72,
        width=0.13,
        extendsize=0.6,
        extend="both",
    )
    cbar.ax.xaxis.set_minor_locator(mticker.NullLocator())
    cbar.ax.yaxis.set_minor_locator(mticker.NullLocator())
    cbar.ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)

    fig.canvas.draw()
    cb = cbar.ax.get_position()
    key = fig.add_axes([cb.x0 - 0.01, cb.y1 + 0.015, 0.065, 0.065])
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.set_aspect("equal")
    key.set_xticks([])
    key.set_yticks([])
    key.plot([0, 1], [0, 1], color="0.25", lw=0.8)
    key.plot([0, 1], [1, 0], color="0.25", lw=0.8)
    key.text(0.50, 0.78, "JFM", ha="center", va="center", fontsize=SMALL_FONT)
    key.text(0.80, 0.50, "AMJ", ha="center", va="center", fontsize=SMALL_FONT)
    key.text(0.50, 0.20, "JAS", ha="center", va="center", fontsize=SMALL_FONT)
    key.text(0.20, 0.50, "OND", ha="center", va="center", fontsize=SMALL_FONT)
    for spine in key.spines.values():
        spine.set_color("0.25")
        spine.set_linewidth(0.8)

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.save(savepath, dpi=600)

    return fig, axs



def _monthly_heatmap_axes(ax, values, leads, months, cmap, norm, title, sig=None):
    import matplotlib.ticker as mticker

    x_edges = np.arange(0.5, 13.5, 1.0)
    y_edges = np.arange(leads.min() - 0.5, leads.max() + 1.5, 1.0)
    mesh = ax.pcolormesh(
        x_edges, y_edges, np.ma.masked_invalid(values),
        cmap=cmap, norm=norm, shading="flat", extend="both",
    )
    if sig is not None:
        ii, jj = np.where(sig & np.isfinite(values))
        ax.scatter(months[jj], leads[ii], s=7, c="k", zorder=5)
    ax.format(
        title=title,
        titleloc="l",
        xlim=(0.5, 12.5),
        ylim=(leads.min() - 0.5, leads.max() + 0.5),
        xticks=months,
        xticklabels=MONTH_NAMES,
        xrotation=45,
        yticks=leads[::2],
        xlabel="Target Month",
        ylabel="Lead Time (months)",
        ticklabelsize=SMALL_FONT,
        grid=False,
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    return mesh


def plot_persistence_benchmark(
    benchmark_ds,
    region="EAMS",
    savepath=None,
    exclude_lead0=True,
):
    """Plot target-month persistence ACC, model-minus-persistence ACC, and PSS."""
    import matplotlib as mpl
    import proplot as pplt

    set_plot_style()
    ds = benchmark_ds
    if exclude_lead0:
        ds = ds.sel(lead=ds.lead > 0)

    leads = ds.lead.values.astype(int)
    months = ds.month.values.astype(int)
    persistence_acc = ds["TARGET_PERSISTENCE_ACC"].sel(region=region).transpose("lead", "month").values
    delta_acc = ds["TARGET_DELTA_ACC"].sel(region=region).transpose("lead", "month").values
    pss = ds["TARGET_PSS"].sel(region=region).transpose("lead", "month").values
    sig_pss = ds["TARGET_SIG_PSS"].sel(region=region).transpose("lead", "month").values.astype(bool)

    seq_cmap = mpl.colormaps["YlOrRd"]
    seq_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    div_cmap = mpl.colormaps["RdBu_r"]
    delta_lim = max(0.2, float(np.nanpercentile(np.abs(delta_acc), 95)))
    pss_lim = max(0.5, float(np.nanpercentile(np.abs(pss), 95)))
    delta_norm = mpl.colors.TwoSlopeNorm(vmin=-delta_lim, vcenter=0.0, vmax=delta_lim)
    pss_norm = mpl.colors.TwoSlopeNorm(vmin=-pss_lim, vcenter=0.0, vmax=pss_lim)

    fig, axs = pplt.subplots(nrows=1, ncols=3, figwidth="19cm", share=False, wspace=3)
    axs.format(grid=False, ec="black", lw=1.0)
    meshes = [
        _monthly_heatmap_axes(
            axs[0], persistence_acc, leads, months, seq_cmap, seq_norm,
            r"$\mathbf{(a)}$ Persistence ACC",
        ),
        _monthly_heatmap_axes(
            axs[1], delta_acc, leads, months, div_cmap, delta_norm,
            r"$\mathbf{(b)}$ $\Delta$ACC (SEOF $-$ persistence)",
        ),
        _monthly_heatmap_axes(
            axs[2], pss, leads, months, div_cmap, pss_norm,
            r"$\mathbf{(c)}$ Persistence skill score", sig=sig_pss,
        ),
    ]

    labels = ["Persistence ACC", r"$\Delta$ACC", "PSS"]
    for i, (mesh, label) in enumerate(zip(meshes, labels)):
        cbar = fig.colorbar(mesh, col=(i + 1, i + 1), loc="b", label=label, length=0.80, width=0.10)
        cbar.ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.save(savepath, dpi=600)
    return fig, axs


def plot_centered_acc_sensitivity(
    skill_ds,
    configuration="corrected",
    region="EAMS",
    savepath=None,
):
    """Compare uncentered and centered target-month ACC."""
    import matplotlib as mpl
    import proplot as pplt

    set_plot_style()
    ds = skill_ds.sel(configuration=configuration) if "configuration" in skill_ds.dims else skill_ds
    if exclude_lead0:
        ds = ds.sel(lead=ds.lead > 0)
    leads = ds.lead.values.astype(int)
    months = ds.month.values.astype(int)
    unc = ds["TARGET_ACC"].sel(region=region).transpose("lead", "month").values
    cen = ds["TARGET_CENTERED_ACC"].sel(region=region).transpose("lead", "month").values
    diff = cen - unc
    sig_cen = None
    if "TARGET_SIG_CENTERED_ACC" in ds:
        sig_cen = ds["TARGET_SIG_CENTERED_ACC"].sel(region=region).transpose("lead", "month").values.astype(bool)

    seq_cmap = mpl.colormaps["YlOrRd"]
    seq_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    div_cmap = mpl.colormaps["RdBu_r"]
    diff_lim = max(0.2, float(np.nanpercentile(np.abs(diff), 95)))
    diff_norm = mpl.colors.TwoSlopeNorm(vmin=-diff_lim, vcenter=0.0, vmax=diff_lim)

    fig, axs = pplt.subplots(nrows=1, ncols=3, figwidth="19cm", share=False, wspace=3)
    axs.format(grid=False, ec="black", lw=1.0)
    meshes = [
        _monthly_heatmap_axes(
            axs[0], unc, leads, months, seq_cmap, seq_norm,
            r"$\mathbf{(a)}$ Uncentered ACC",
        ),
        _monthly_heatmap_axes(
            axs[1], cen, leads, months, seq_cmap, seq_norm,
            r"$\mathbf{(b)}$ Centered ACC", sig=sig_cen,
        ),
        _monthly_heatmap_axes(
            axs[2], diff, leads, months, div_cmap, diff_norm,
            r"$\mathbf{(c)}$ Centered $-$ uncentered ACC",
        ),
    ]
    labels = ["ACC", "Centered ACC", r"$\Delta$ACC"]
    for i, (mesh, label) in enumerate(zip(meshes, labels)):
        cbar = fig.colorbar(mesh, col=(i + 1, i + 1), loc="b", label=label, length=0.80, width=0.10)
        cbar.ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.save(savepath, dpi=600)
    return fig, axs


def plot_anomaly_base_sensitivity(
    base_skill_ds,
    reference_base,
    alternate_base,
    region="EAMS",
    savepath=None,
):
    """Plot alternate-minus-reference changes in target-month ACC and MSSS."""
    import matplotlib as mpl
    import proplot as pplt

    set_plot_style()
    ref = base_skill_ds.sel(anomaly_base=reference_base)
    alt = base_skill_ds.sel(anomaly_base=alternate_base)
    if exclude_lead0:
        ref = ref.sel(lead=ref.lead > 0)
        alt = alt.sel(lead=alt.lead > 0)
    leads = ref.lead.values.astype(int)
    months = ref.month.values.astype(int)

    delta_acc = (
        alt["TARGET_ACC"].sel(region=region)
        - ref["TARGET_ACC"].sel(region=region)
    ).transpose("lead", "month").values
    delta_msss = (
        alt["TARGET_MSSS"].sel(region=region)
        - ref["TARGET_MSSS"].sel(region=region)
    ).transpose("lead", "month").values

    cmap = mpl.colormaps["RdBu_r"]
    acc_lim = max(0.1, float(np.nanpercentile(np.abs(delta_acc), 95)))
    msss_lim = max(0.1, float(np.nanpercentile(np.abs(delta_msss), 95)))
    norms = [
        mpl.colors.TwoSlopeNorm(vmin=-acc_lim, vcenter=0.0, vmax=acc_lim),
        mpl.colors.TwoSlopeNorm(vmin=-msss_lim, vcenter=0.0, vmax=msss_lim),
    ]

    fig, axs = pplt.subplots(nrows=1, ncols=2, figwidth="14cm", share=False, wspace=3)
    axs.format(grid=False, ec="black", lw=1.0)
    meshes = [
        _monthly_heatmap_axes(
            axs[0], delta_acc, leads, months, cmap, norms[0],
            rf"$\mathbf{{(a)}}$ $\Delta$ACC ({alternate_base} $-$ {reference_base})",
        ),
        _monthly_heatmap_axes(
            axs[1], delta_msss, leads, months, cmap, norms[1],
            rf"$\mathbf{{(b)}}$ $\Delta$MSSS ({alternate_base} $-$ {reference_base})",
        ),
    ]
    for i, (mesh, label) in enumerate(zip(meshes, [r"$\Delta$ACC", r"$\Delta$MSSS"])):
        cbar = fig.colorbar(mesh, col=(i + 1, i + 1), loc="b", label=label, length=0.82, width=0.10)
        cbar.ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.save(savepath, dpi=600)
    return fig, axs


def _lead_label(group):
    return {"1-3": "1–3", "4-6": "4–6", "7-12": "7–12"}.get(group, group)


def _cosine_weights_2d(lat):
    return xr.DataArray(np.cos(np.deg2rad(lat)), coords={"lat": lat}, dims=("lat",))


def _area_weighted_mean_da(da: xr.DataArray, mask: xr.DataArray) -> float:
    weights = _cosine_weights_2d(da.lat)
    out = da.where(mask).weighted(weights).mean(("lat", "lon"))
    return float(out) if np.isfinite(out) else np.nan


def _area_weighted_fraction_da(mask_true: xr.DataArray, domain_mask: xr.DataArray) -> float:
    weights = _cosine_weights_2d(mask_true.lat)
    numerator = mask_true.astype(float).where(domain_mask).weighted(weights).mean(("lat", "lon"))
    return float(numerator) if np.isfinite(numerator) else np.nan


def _nanmean_or_nan(values):
    values = np.asarray(values, dtype=float)
    if np.isfinite(values).any():
        return float(np.nanmean(values))
    return np.nan


def _grid_acc(forecast: xr.DataArray, observed: xr.DataArray, months, lead: int) -> xr.DataArray:
    f = _target_time(forecast).sel(lead=lead)
    o = _target_time(observed)
    f = f.where(f.target_time.dt.month.isin(months), drop=True)
    o = o.where(o.target_time.dt.month.isin(months), drop=True)
    f, o = xr.align(f, o, join="inner")
    numer = (f * o).sum("target_time", skipna=True)
    denom = np.sqrt((f ** 2).sum("target_time", skipna=True) * (o ** 2).sum("target_time", skipna=True))
    return numer / denom


def _refinement_threshold(cache: dict, q: float = 0.25) -> float:
    values = []
    for item in cache.values():
        delta = item["delta_total"].values.ravel()
        delta = delta[np.isfinite(delta) & (delta > 0)]
        if delta.size:
            values.append(delta)
    if not values:
        return 0.0
    merged = np.concatenate(values)
    return float(np.quantile(merged, q))


def _build_acc_delta(mode1_forecast: xr.DataArray, full_model: xr.DataArray, observed: xr.DataArray) -> dict:
    cache = {}
    mode1_forecast, full_model, observed = xr.align(
        _target_time(mode1_forecast),
        _target_time(full_model),
        _target_time(observed),
        join="inner",
    )
    for season_name, months in SEASON_MONTHS.items():
        for lead in full_model.lead.values.astype(int):
            acc_mode1 = _grid_acc(mode1_forecast, observed, months, int(lead))
            acc_full = _grid_acc(full_model, observed, months, int(lead))
            cache[(season_name, int(lead))] = {
                "delta_total": (acc_full - acc_mode1).astype(np.float32)
            }
    return cache


def _summarize_refinement(cache: dict, region_masks: Mapping[str, xr.DataArray], helped_min_delta: float) -> xr.Dataset:
    region_order = ["EAMS", "YS", "ECS", "EJS"]
    season_names = list(SEASON_MONTHS)
    group_names = list(LEAD_GROUPS)

    mean_helped_delta_acc = np.full((len(region_order), len(season_names), len(group_names)), np.nan, dtype=np.float32)
    helped_area_fraction = np.full_like(mean_helped_delta_acc, np.nan)

    for ir, region in enumerate(region_order):
        region_mask = region_masks[region]
        for isea, season_name in enumerate(season_names):
            for igrp, (_, leads) in enumerate(LEAD_GROUPS.items()):
                lead_means = []
                lead_fracs = []
                for lead in leads:
                    result = cache.get((season_name, int(lead)))
                    if result is None:
                        continue
                    delta = result["delta_total"]
                    region_domain = _align_mask(region_mask, delta)
                    helped = region_domain & np.isfinite(delta) & (delta > float(helped_min_delta))
                    if int(helped.fillna(False).astype(np.int8).sum()) > 0:
                        lead_means.append(_area_weighted_mean_da(delta, helped))
                        lead_fracs.append(_area_weighted_fraction_da(helped, region_domain))
                    else:
                        lead_means.append(np.nan)
                        lead_fracs.append(0.0)
                mean_helped_delta_acc[ir, isea, igrp] = _nanmean_or_nan(lead_means)
                helped_area_fraction[ir, isea, igrp] = _nanmean_or_nan(lead_fracs)

    summary = xr.Dataset(
        {
            "mean_helped_delta_acc": (("region", "season", "lead_group"), mean_helped_delta_acc),
            "helped_area_fraction": (("region", "season", "lead_group"), helped_area_fraction),
        },
        coords={"region": region_order, "season": season_names, "lead_group": group_names},
    )
    summary.attrs["helped_threshold"] = float(helped_min_delta)
    return summary


def build_acc_refinement(
    mode1_path,
    full_model_path,
    region_masks: Mapping[str, xr.DataArray],
    summary_path=None,
    verification_period=("2013-01-01", "2025-12-31"),
    threshold_quantile=0.25,
):
    with xr.open_dataset(mode1_path) as mode1_ds, xr.open_dataset(full_model_path) as full_ds:
        mode1_forecast = mode1_ds["sst_forecast_fixed_base_anomaly"].load()
        full_model = full_ds["sst_forecast_fixed_base_anomaly"].load()
        observed = full_ds["sst_observed_fixed_base_anomaly"].load()

    mode1_forecast = _target_time(mode1_forecast).sel(target_time=slice(*verification_period))
    full_model = _target_time(full_model).sel(target_time=slice(*verification_period))
    observed = _target_time(observed).sel(target_time=slice(*verification_period))
    mode1_forecast, full_model, observed = xr.align(mode1_forecast, full_model, observed, join="inner")
    mode1_forecast = mode1_forecast.sel(lead=mode1_forecast.lead > 0)
    full_model = full_model.sel(lead=full_model.lead > 0)

    cache = _build_acc_delta(mode1_forecast, full_model, observed)
    helped_min_delta = _refinement_threshold(cache, q=threshold_quantile)
    summary = _summarize_refinement(cache, region_masks, helped_min_delta=helped_min_delta)
    summary.attrs["threshold_quantile"] = float(threshold_quantile)

    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_netcdf(summary_path)

    return summary

def plot_acc_refinement(summary_ds, savepath="ACC_Refinement.png"):
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import proplot as pplt

    set_plot_style()

    fig, axs = pplt.subplots(
        nrows=1,
        ncols=4,
        figwidth="18cm",
        figheight="7cm",
        share=False,
        hspace=4,
    )

    axs.format(
        suptitle="Local positive ACC refinements in the three-component forecast",
        suptitle_kw={"weight": "normal", "size": "med-large"},
        suptitlepad="1.2em",
        ec="black",
        lw=1.2,
        grid=False,
    )

    region_order = ["EAMS", "YS", "ECS", "EJS"]
    seasons = summary_ds.season.values.tolist()
    groups = summary_ds.lead_group.values.tolist()
    display_groups = [_lead_label(group) for group in groups]

    bounds_skill = [0.02, 0.05, 0.10, 0.15, 0.20]
    base_cmap = plt.get_cmap("ylorrd")
    raw_cols = base_cmap(np.linspace(0, 1, 10))
    inner_colors = [raw_cols[1], raw_cols[3], raw_cols[6], raw_cols[8]]
    cmap_skill = mpl.colors.ListedColormap(inner_colors)
    cmap_skill.set_under("white")
    cmap_skill.set_over("#7f0000")
    cmap_skill.set_bad("white")
    norm_skill = mcolors.BoundaryNorm(bounds_skill, cmap_skill.N)

    text_effect = [path_effects.withStroke(linewidth=1.3, foreground="white")]
    im_s = None

    for r_idx, region in enumerate(region_order):
        values = summary_ds["mean_helped_delta_acc"].sel(region=region).values
        ax = axs[r_idx]
        im_s = ax.imshow(np.ma.masked_invalid(values), cmap=cmap_skill, norm=norm_skill, aspect="auto")

        for i in range(len(seasons)):
            for j in range(len(groups)):
                value = float(values[i, j])
                if np.isfinite(value):
                    ax.text(
                        j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=6, color="black", path_effects=text_effect,
                    )

        ax.format(
            yticklabels=seasons,
            yticks=np.arange(len(seasons)),
            xticks=np.arange(len(groups)),
            xticklabels=display_groups,
            grid=False,
            title=rf"$\mathbf{{({chr(97 + r_idx)})}}$ {region}",
            titleloc="l",
            titleweight="normal",
            titlesize=MID_FONT,
            titlepad=3,
            ticklabelsize=SMALL_FONT,
        )
        ax.tick_params(axis="x", labelbottom=True, pad=2)

    for ax in axs:
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False)

    scalar = mpl.cm.ScalarMappable(norm=norm_skill, cmap=cmap_skill)
    scalar.set_array([])
    cbar = fig.colorbar(
        scalar,
        loc="b",
        col=(2, 3),
        ticks=bounds_skill,
        boundaries=bounds_skill,
        label=r"Mean $\Delta$ACC",
        length=0.85,
        width=0.15,
        spacing="proportional",
    )
    cbar.ax.xaxis.set_minor_locator(mticker.NullLocator())
    cbar.ax.yaxis.set_minor_locator(mticker.NullLocator())
    cbar.ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)

    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.save(str(savepath), dpi=600)
    jpg_path = savepath.with_suffix(".jpg")
    fig.save(str(jpg_path), dpi=600)

    return fig, axs