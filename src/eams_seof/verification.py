from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .forecast import fixed_base_anomaly
from .time import datetime_to_decimal_year

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
    f, o = _valid(forecast, observed, min_samples)
    if f is None:
        return np.nan
    denom = np.sqrt(np.sum(f**2) * np.sum(o**2))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.sum(f * o) / denom)


def acc_centered(forecast, observed, min_samples=3):
    f, o = _valid(forecast, observed, min_samples)
    if f is None:
        return np.nan
    f = f - np.mean(f)
    o = o - np.mean(o)
    denom = np.sqrt(np.sum(f**2) * np.sum(o**2))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.sum(f * o) / denom)


def msss(forecast, observed, min_samples=3):
    f, o = _valid(forecast, observed, min_samples)
    if f is None:
        return np.nan
    mse_ref = np.mean(o**2)
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


def _bootstrap_values(f, o, metric, n_resamples, rng):
    n = f.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    fb = f[idx]
    ob = o[idx]

    if metric in {"ACC", "CENTERED_ACC"}:
        if metric == "CENTERED_ACC":
            fb = fb - np.mean(fb, axis=1, keepdims=True)
            ob = ob - np.mean(ob, axis=1, keepdims=True)
        denom = np.sqrt(np.sum(fb**2, axis=1) * np.sum(ob**2, axis=1))
        return np.divide(
            np.sum(fb * ob, axis=1),
            denom,
            out=np.full(n_resamples, np.nan, dtype=float),
            where=np.isfinite(denom) & (denom > 0),
        )

    mse_ref = np.mean(ob**2, axis=1)
    mse_fcst = np.mean((fb - ob) ** 2, axis=1)
    ratio = np.divide(
        mse_fcst,
        mse_ref,
        out=np.full(n_resamples, np.nan, dtype=float),
        where=np.isfinite(mse_ref) & (mse_ref > 0),
    )
    return 1.0 - ratio


def bootstrap_ci(
    forecast,
    observed,
    metric="ACC",
    n_boot=5000,
    min_samples=10,
    min_boot_valid=500,
    ci_level=0.95,
    seed=42,
):
    f, o = _valid(forecast, observed, min_samples)
    if f is None:
        return np.nan, np.nan, np.nan, 0

    metric = metric.upper()
    if metric == "ACC":
        stat = acc(f, o, min_samples=min_samples)
    elif metric == "CENTERED_ACC":
        stat = acc_centered(f, o, min_samples=min_samples)
    elif metric == "MSSS":
        stat = msss(f, o, min_samples=min_samples)
    else:
        raise ValueError("metric must be ACC, CENTERED_ACC, or MSSS")

    if not np.isfinite(stat):
        return stat, np.nan, np.nan, int(f.size)

    if n_boot <= 0:
        return stat, np.nan, np.nan, int(f.size)

    rng = np.random.default_rng(seed)
    boot = _bootstrap_values(f, o, metric, int(n_boot), rng)
    boot = boot[np.isfinite(boot)]
    if boot.size < min_boot_valid:
        return stat, np.nan, np.nan, int(f.size)

    tail = (1.0 - ci_level) / 2.0
    ci_low, ci_high = np.quantile(boot, [tail, 1.0 - tail])
    return stat, float(ci_low), float(ci_high), int(f.size)


def null_pvalue(
    forecast,
    observed,
    metric="ACC",
    n_null=5000,
    min_samples=10,
    seed=42,
):
    f, o = _valid(forecast, observed, min_samples)
    if f is None or n_null <= 0:
        return np.nan

    metric = metric.upper()
    rng = np.random.default_rng(seed)

    if metric in {"ACC", "CENTERED_ACC"}:
        stat = acc(f, o, min_samples=min_samples) if metric == "ACC" else acc_centered(f, o, min_samples=min_samples)
        if not np.isfinite(stat):
            return np.nan
        idx = np.vstack([rng.permutation(f.size) for _ in range(int(n_null))])
        fb = np.broadcast_to(f, idx.shape)
        ob = o[idx]
        if metric == "CENTERED_ACC":
            fb = fb - np.mean(fb, axis=1, keepdims=True)
            ob = ob - np.mean(ob, axis=1, keepdims=True)
        denom = np.sqrt(np.sum(fb**2, axis=1) * np.sum(ob**2, axis=1))
        null = np.divide(
            np.sum(fb * ob, axis=1),
            denom,
            out=np.full(int(n_null), np.nan, dtype=float),
            where=np.isfinite(denom) & (denom > 0),
        )
        null = null[np.isfinite(null)]
        return float((1.0 + np.sum(null >= stat)) / (1.0 + null.size)) if null.size else np.nan

    if metric == "MSSS":
        mse_ref = np.mean(o**2)
        if not np.isfinite(mse_ref) or mse_ref <= 0:
            return np.nan
        benefit = o**2 - (f - o) ** 2
        stat = float(np.mean(benefit))
        signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_null), f.size))
        null = np.mean(signs * benefit[None, :], axis=1)
        return float((1.0 + np.sum(null >= stat)) / (1.0 + null.size))

    raise ValueError("metric must be ACC, CENTERED_ACC, or MSSS")


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
        raise ValueError("Region mask must contain lat and lon dimensions")

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
    subregions = {
        "YS": _align_mask(mask_ds["region_mask_ys"], template),
        "ECS": _align_mask(mask_ds["region_mask_ecs"], template),
        "EJS": _align_mask(mask_ds["region_mask_ejs"], template),
    }
    eams = subregions["YS"] | subregions["ECS"] | subregions["EJS"]

    if "region_mask_eams" in mask_ds:
        stored = _align_mask(mask_ds["region_mask_eams"], template)
        if not np.array_equal(eams.values, stored.values):
            raise ValueError("region_mask_eams does not equal YS | ECS | EJS")

    return {"EAMS": eams, **subregions}


def _target_time(da: xr.DataArray) -> xr.DataArray:
    if "target_time" in da.dims:
        return da
    if "time" in da.dims:
        return da.rename(time="target_time")
    raise ValueError("DataArray must contain target_time or time")


def _season_mean(da: xr.DataArray, months: Sequence[int]) -> xr.DataArray:
    da = _target_time(da)
    da = da.where(da.target_time.dt.month.isin(list(months)), drop=True)
    count = da.resample(target_time="YS").count()
    return da.resample(target_time="YS").mean().where(count >= len(months))


def _fdr(raw: xr.DataArray, score: xr.DataArray, alpha: float, scope: str):
    adj = xr.full_like(raw, np.nan)
    sig = xr.zeros_like(raw, dtype=bool)

    if scope == "region":
        for ir in range(raw.sizes["region"]):
            reject, values = bh_fdr(raw.isel(region=ir).values.ravel(), alpha=alpha)
            adj.values[ir] = values.reshape(raw.isel(region=ir).shape)
            sig.values[ir] = reject.reshape(raw.isel(region=ir).shape)
    elif scope == "global":
        reject, values = bh_fdr(raw.values.ravel(), alpha=alpha)
        adj.values[:] = values.reshape(raw.shape)
        sig.values[:] = reject.reshape(raw.shape)
    else:
        raise ValueError("FDR scope must be region or global")

    return adj.astype(np.float32), sig & np.isfinite(score) & (score > 0)


def _score_matrix(
    forecast: xr.DataArray,
    observed: xr.DataArray,
    masks: Mapping[str, xr.DataArray],
    grouping: str,
    min_samples: int,
    n_boot: int,
    min_boot_valid: int,
    ci_level: float,
    n_null: int,
    fdr_q: float,
    fdr_scope: str,
    seed: int,
    run_significance: bool,
) -> xr.Dataset:
    leads = forecast.lead.values.astype(int)
    regions = list(masks)
    groups = np.arange(1, 13) if grouping == "month" else np.array(list(SEASON_MONTHS))
    group_dim = "month" if grouping == "month" else "season"
    shape = (len(regions), len(leads), len(groups))
    coords = {"region": regions, "lead": leads, group_dim: groups}
    dims = ("region", "lead", group_dim)

    names = [
        "ACC", "CENTERED_ACC", "MSSS", "BIAS", "RMSE", "STD_RATIO", "N",
        "P_ACC_RAW", "P_CENTERED_ACC_RAW", "P_MSSS_RAW",
        "ACC_CI_LO", "ACC_CI_HI", "CENTERED_ACC_CI_LO", "CENTERED_ACC_CI_HI",
        "MSSS_CI_LO", "MSSS_CI_HI",
    ]
    fields = {
        name: xr.DataArray(np.full(shape, np.nan, dtype=np.float32), coords=coords, dims=dims)
        for name in names
    }

    for ir, region in enumerate(regions):
        f_reg = region_mean(forecast, masks[region]).load()
        o_reg = region_mean(observed, masks[region]).load()

        for il, lead in enumerate(leads):
            f, o = xr.align(f_reg.sel(lead=lead), o_reg, join="inner")

            for ig, group in enumerate(groups):
                if grouping == "month":
                    ff = f.where(f.target_time.dt.month == int(group), drop=True)
                    oo = o.where(o.target_time.dt.month == int(group), drop=True)
                else:
                    ff = _season_mean(f, SEASON_MONTHS[str(group)])
                    oo = _season_mean(o, SEASON_MONTHS[str(group)])

                ff, oo = xr.align(ff, oo, join="inner")
                valid = np.isfinite(ff.values) & np.isfinite(oo.values)
                fv = np.asarray(ff.values[valid], dtype=float)
                ov = np.asarray(oo.values[valid], dtype=float)
                if fv.size < min_samples:
                    continue

                loc = {"region": region, "lead": lead, group_dim: group}
                score = metrics(fv, ov, min_samples=min_samples)
                for name in ["ACC", "CENTERED_ACC", "MSSS", "BIAS", "RMSE", "STD_RATIO", "N"]:
                    fields[name].loc[loc] = score[name]

                if not run_significance:
                    continue

                for imetric, metric in enumerate(["ACC", "CENTERED_ACC", "MSSS"]):
                    s = seed + 100000 * imetric + 10000 * ir + 100 * il + ig
                    stat, lo, hi, _ = bootstrap_ci(
                        fv,
                        ov,
                        metric=metric,
                        n_boot=n_boot,
                        min_samples=min_samples,
                        min_boot_valid=min_boot_valid,
                        ci_level=ci_level,
                        seed=s,
                    )
                    p = null_pvalue(
                        fv,
                        ov,
                        metric=metric,
                        n_null=n_null,
                        min_samples=min_samples,
                        seed=s + 500000,
                    )
                    fields[f"P_{metric}_RAW"].loc[loc] = p
                    fields[f"{metric}_CI_LO"].loc[loc] = lo
                    fields[f"{metric}_CI_HI"].loc[loc] = hi

    for metric in ["ACC", "CENTERED_ACC", "MSSS"]:
        raw = fields[f"P_{metric}_RAW"]
        if run_significance:
            adj, sig = _fdr(raw, fields[metric], fdr_q, fdr_scope)
        else:
            adj = xr.full_like(raw, np.nan)
            sig = xr.zeros_like(raw, dtype=bool)
        fields[f"P_{metric}_FDR"] = adj
        fields[f"SIG_{metric}"] = sig

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
    n_null=5000,
    fdr_q=0.10,
    target_month_fdr_scope="region",
    seasonal_fdr_scope="region",
    seed=42,
    run_significance=True,
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
        forecast,
        observed,
        masks,
        "month",
        min_samples,
        n_boot,
        min_boot_valid,
        ci_level,
        n_null,
        fdr_q,
        target_month_fdr_scope,
        seed,
        run_significance,
    )
    seasonal = _score_matrix(
        forecast,
        observed,
        masks,
        "season",
        min_samples,
        n_boot,
        min_boot_valid,
        ci_level,
        n_null,
        fdr_q,
        seasonal_fdr_scope,
        seed + 200000,
        run_significance,
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
        null_samples=int(n_null),
        minimum_samples=int(min_samples),
        minimum_bootstrap_valid=int(min_boot_valid),
        confidence_level=float(ci_level),
        fdr_q=float(fdr_q),
        target_month_fdr_scope=str(target_month_fdr_scope),
        seasonal_fdr_scope=str(seasonal_fdr_scope),
        acc_null_test="Monte Carlo permutation of forecast-observation pairing",
        centered_acc_null_test="Monte Carlo permutation of forecast-observation pairing",
        msss_null_test="Monte Carlo sign-flip test of paired squared-error advantage over zero anomaly",
        significance_alternative="positive skill",
    )
    return output


def evaluate_configs(
    forecasts: Mapping[str, xr.DataArray],
    observed: xr.DataArray,
    masks: Mapping[str, xr.DataArray],
    significance_configurations: Sequence[str] | None = None,
    **kwargs,
) -> xr.Dataset:
    selected = set(forecasts) if significance_configurations is None else set(significance_configurations)
    datasets = []
    for name, fcst in forecasts.items():
        ds = evaluate_skill(
            fcst,
            observed,
            masks,
            run_significance=name in selected,
            **kwargs,
        ).expand_dims(configuration=[name])
        datasets.append(ds)
    return xr.concat(datasets, dim="configuration")


def build_anomaly_persistence_forecast(
    observed_anomaly: xr.DataArray,
    forecast_template: xr.DataArray,
) -> xr.DataArray:
    if "time" not in observed_anomaly.dims:
        if "target_time" in observed_anomaly.dims:
            observed_anomaly = observed_anomaly.rename(target_time="time")
        else:
            raise ValueError("observed_anomaly must contain time or target_time")

    template = _target_time(forecast_template)
    if "lead" not in template.dims:
        raise ValueError("forecast_template must contain lead")

    target_times = pd.DatetimeIndex(template.target_time.values)
    pieces = []
    for lead in template.lead.values.astype(int):
        source_times = target_times - pd.DateOffset(months=int(lead))
        source = observed_anomaly.reindex(time=source_times)
        source = source.rename(time="target_time").assign_coords(target_time=template.target_time.values)
        pieces.append(source.expand_dims(lead=[int(lead)]))

    out = xr.concat(pieces, dim="lead").transpose("lead", "target_time", "lat", "lon")
    out.name = "sst_anomaly_persistence_forecast"
    return out.astype(np.float32)


def build_trend_adjusted_persistence_forecast(
    detrended_sst: xr.DataArray,
    forecast_template: xr.DataArray,
    training_climatology: xr.DataArray,
    trend_coefficients: xr.DataArray,
    reference_decimal_year: float,
    observed_sst: xr.DataArray,
    anomaly_reference_period: tuple[str, str],
    sea_mask: xr.DataArray | None = None,
):
    persistence = build_anomaly_persistence_forecast(detrended_sst, forecast_template)
    target_index = pd.DatetimeIndex(persistence.target_time.values)
    target_month = xr.DataArray(
        target_index.month.astype(np.int16),
        coords={"target_time": persistence.target_time},
        dims=("target_time",),
    )
    target_year = xr.DataArray(
        datetime_to_decimal_year(target_index).astype(np.float32),
        coords={"target_time": persistence.target_time},
        dims=("target_time",),
    )
    seasonal = training_climatology.sel(month=target_month).reset_coords(drop=True)
    linear = (
        trend_coefficients.sel(coef="intercept")
        + trend_coefficients.sel(coef="slope") * (target_year - float(reference_decimal_year))
    ).reset_coords(drop=True)
    absolute = (persistence + seasonal + linear).astype(np.float32)
    return fixed_base_anomaly(
        absolute_forecast_target=absolute,
        observed_sst=observed_sst,
        anomaly_reference_period=anomaly_reference_period,
        sea_mask=sea_mask,
    )


def persistence_skill_score(model_forecast, persistence_forecast, observed, min_samples=3):
    m, p, o = _valid_three(model_forecast, persistence_forecast, observed, min_samples)
    if m is None:
        return np.nan
    mse_persistence = np.mean((p - o) ** 2)
    if not np.isfinite(mse_persistence) or mse_persistence <= 0:
        return np.nan
    mse_model = np.mean((m - o) ** 2)
    return float(1.0 - mse_model / mse_persistence)


def _paired_bootstrap(m, p, o, n_boot, rng):
    n = m.size
    idx = rng.integers(0, n, size=(n_boot, n))
    mb = m[idx]
    pb = p[idx]
    ob = o[idx]
    model_acc = _row_acc(mb, ob)
    persistence_acc = _row_acc(pb, ob)
    mse_model = np.mean((mb - ob) ** 2, axis=1)
    mse_persistence = np.mean((pb - ob) ** 2, axis=1)
    pss = 1.0 - np.divide(
        mse_model,
        mse_persistence,
        out=np.full(n_boot, np.nan, dtype=float),
        where=np.isfinite(mse_persistence) & (mse_persistence > 0),
    )
    return model_acc - persistence_acc, pss


def _row_acc(a, b):
    denom = np.sqrt(np.sum(a**2, axis=1) * np.sum(b**2, axis=1))
    return np.divide(
        np.sum(a * b, axis=1),
        denom,
        out=np.full(a.shape[0], np.nan, dtype=float),
        where=np.isfinite(denom) & (denom > 0),
    )


def paired_persistence_stats(
    model_forecast,
    persistence_forecast,
    observed,
    n_boot=5000,
    n_null=5000,
    min_samples=10,
    min_boot_valid=500,
    ci_level=0.95,
    seed=42,
):
    m, p, o = _valid_three(model_forecast, persistence_forecast, observed, min_samples)
    if m is None:
        return {
            "DELTA_ACC": np.nan,
            "PSS": np.nan,
            "DELTA_ACC_P": np.nan,
            "PSS_P": np.nan,
            "DELTA_ACC_CI_LO": np.nan,
            "DELTA_ACC_CI_HI": np.nan,
            "PSS_CI_LO": np.nan,
            "PSS_CI_HI": np.nan,
            "N": 0,
        }

    delta_acc = acc(m, o, min_samples) - acc(p, o, min_samples)
    pss = persistence_skill_score(m, p, o, min_samples)
    rng = np.random.default_rng(seed)
    db, pb = _paired_bootstrap(m, p, o, int(n_boot), rng) if n_boot > 0 else (np.array([]), np.array([]))
    db = db[np.isfinite(db)]
    pb = pb[np.isfinite(pb)]
    tail = (1.0 - ci_level) / 2.0
    d_lo = d_hi = p_lo = p_hi = np.nan
    if db.size >= min_boot_valid:
        d_lo, d_hi = np.quantile(db, [tail, 1.0 - tail])
    if pb.size >= min_boot_valid:
        p_lo, p_hi = np.quantile(pb, [tail, 1.0 - tail])

    rng = np.random.default_rng(seed + 500000)
    swap = rng.integers(0, 2, size=(int(n_null), m.size)).astype(bool)
    mb = np.where(swap, p[None, :], m[None, :])
    pb_null = np.where(swap, m[None, :], p[None, :])
    ob = np.broadcast_to(o, mb.shape)
    d_null = _row_acc(mb, ob) - _row_acc(pb_null, ob)
    d_null = d_null[np.isfinite(d_null)]
    d_p = float((1.0 + np.sum(d_null >= delta_acc)) / (1.0 + d_null.size)) if d_null.size else np.nan

    benefit = (p - o) ** 2 - (m - o) ** 2
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_null), m.size))
    p_null = np.mean(signs * benefit[None, :], axis=1)
    p_stat = float(np.mean(benefit))
    p_p = float((1.0 + np.sum(p_null >= p_stat)) / (1.0 + p_null.size))

    return {
        "DELTA_ACC": float(delta_acc),
        "PSS": float(pss),
        "DELTA_ACC_P": d_p,
        "PSS_P": p_p,
        "DELTA_ACC_CI_LO": float(d_lo),
        "DELTA_ACC_CI_HI": float(d_hi),
        "PSS_CI_LO": float(p_lo),
        "PSS_CI_HI": float(p_hi),
        "N": int(m.size),
    }


def _persistence_score_matrix(
    model_forecast,
    persistence_forecast,
    observed,
    masks,
    grouping,
    min_samples,
    n_boot,
    min_boot_valid,
    ci_level,
    n_null,
    fdr_q,
    fdr_scope,
    seed,
):
    leads = model_forecast.lead.values.astype(int)
    regions = list(masks)
    groups = np.arange(1, 13) if grouping == "month" else np.array(list(SEASON_MONTHS))
    group_dim = "month" if grouping == "month" else "season"
    shape = (len(regions), len(leads), len(groups))
    coords = {"region": regions, "lead": leads, group_dim: groups}
    dims = ("region", "lead", group_dim)

    names = [
        "MODEL_ACC", "PERSISTENCE_ACC", "DELTA_ACC", "PSS", "MODEL_RMSE", "PERSISTENCE_RMSE", "N",
        "P_DELTA_ACC_RAW", "P_PSS_RAW", "DELTA_ACC_CI_LO", "DELTA_ACC_CI_HI", "PSS_CI_LO", "PSS_CI_HI",
    ]
    fields = {
        name: xr.DataArray(np.full(shape, np.nan, dtype=np.float32), coords=coords, dims=dims)
        for name in names
    }

    for ir, region in enumerate(regions):
        m_reg = region_mean(model_forecast, masks[region]).load()
        p_reg = region_mean(persistence_forecast, masks[region]).load()
        o_reg = region_mean(observed, masks[region]).load()

        for il, lead in enumerate(leads):
            m, p, o = xr.align(m_reg.sel(lead=lead), p_reg.sel(lead=lead), o_reg, join="inner")

            for ig, group in enumerate(groups):
                if grouping == "month":
                    selector = m.target_time.dt.month == int(group)
                    mm = m.where(selector, drop=True)
                    pp = p.where(selector, drop=True)
                    oo = o.where(selector, drop=True)
                else:
                    mm = _season_mean(m, SEASON_MONTHS[str(group)])
                    pp = _season_mean(p, SEASON_MONTHS[str(group)])
                    oo = _season_mean(o, SEASON_MONTHS[str(group)])

                mm, pp, oo = xr.align(mm, pp, oo, join="inner")
                valid = np.isfinite(mm.values) & np.isfinite(pp.values) & np.isfinite(oo.values)
                mv = np.asarray(mm.values[valid], dtype=float)
                pv = np.asarray(pp.values[valid], dtype=float)
                ov = np.asarray(oo.values[valid], dtype=float)
                if mv.size < min_samples:
                    continue

                loc = {"region": region, "lead": lead, group_dim: group}
                stats = paired_persistence_stats(
                    mv,
                    pv,
                    ov,
                    n_boot=n_boot,
                    n_null=n_null,
                    min_samples=min_samples,
                    min_boot_valid=min_boot_valid,
                    ci_level=ci_level,
                    seed=seed + 10000 * ir + 100 * il + ig,
                )
                fields["MODEL_ACC"].loc[loc] = acc(mv, ov, min_samples)
                fields["PERSISTENCE_ACC"].loc[loc] = acc(pv, ov, min_samples)
                fields["DELTA_ACC"].loc[loc] = stats["DELTA_ACC"]
                fields["PSS"].loc[loc] = stats["PSS"]
                fields["MODEL_RMSE"].loc[loc] = np.sqrt(np.mean((mv - ov) ** 2))
                fields["PERSISTENCE_RMSE"].loc[loc] = np.sqrt(np.mean((pv - ov) ** 2))
                fields["N"].loc[loc] = stats["N"]
                fields["P_DELTA_ACC_RAW"].loc[loc] = stats["DELTA_ACC_P"]
                fields["P_PSS_RAW"].loc[loc] = stats["PSS_P"]
                fields["DELTA_ACC_CI_LO"].loc[loc] = stats["DELTA_ACC_CI_LO"]
                fields["DELTA_ACC_CI_HI"].loc[loc] = stats["DELTA_ACC_CI_HI"]
                fields["PSS_CI_LO"].loc[loc] = stats["PSS_CI_LO"]
                fields["PSS_CI_HI"].loc[loc] = stats["PSS_CI_HI"]

    for metric in ["DELTA_ACC", "PSS"]:
        raw = fields[f"P_{metric}_RAW"]
        adj, sig = _fdr(raw, fields[metric], fdr_q, fdr_scope)
        fields[f"P_{metric}_FDR"] = adj
        fields[f"SIG_{metric}"] = sig

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
    n_null=5000,
    fdr_q=0.10,
    target_month_fdr_scope="region",
    seasonal_fdr_scope="region",
    seed=42,
) -> xr.Dataset:
    model_forecast = _target_time(model_forecast).sel(target_time=slice(*verification_period))
    persistence_forecast = _target_time(persistence_forecast).sel(target_time=slice(*verification_period))
    observed = _target_time(observed).sel(target_time=slice(*verification_period))
    model_forecast, persistence_forecast, observed = xr.align(
        model_forecast,
        persistence_forecast,
        observed,
        join="inner",
    )

    target = _persistence_score_matrix(
        model_forecast,
        persistence_forecast,
        observed,
        masks,
        "month",
        min_samples,
        n_boot,
        min_boot_valid,
        ci_level,
        n_null,
        fdr_q,
        target_month_fdr_scope,
        seed,
    )
    seasonal = _persistence_score_matrix(
        model_forecast,
        persistence_forecast,
        observed,
        masks,
        "season",
        min_samples,
        n_boot,
        min_boot_valid,
        ci_level,
        n_null,
        fdr_q,
        seasonal_fdr_scope,
        seed + 200000,
    )

    output = xr.Dataset()
    for prefix, ds in [("TARGET", target), ("SEASON", seasonal)]:
        for name, da in ds.data_vars.items():
            output[f"{prefix}_{name}"] = da

    output.attrs.update(
        verification_start=str(verification_period[0]),
        verification_end=str(verification_period[1]),
        persistence_definition="forecast(t,L)=detrended anomaly at t-L with target climatology and trend restored",
        bootstrap_samples=int(n_boot),
        null_samples=int(n_null),
        confidence_level=float(ci_level),
        fdr_q=float(fdr_q),
        delta_acc_null_test="paired label-swap randomization",
        pss_null_test="paired sign-flip test of squared-error advantage over persistence",
        significance_alternative="model higher than persistence",
    )
    return output


def evaluate_verification(
    forecasts: Mapping[str, xr.DataArray],
    observed: xr.DataArray,
    masks: Mapping[str, xr.DataArray],
    primary_configuration: str,
    persistence_configuration: str | None = None,
    significance_configurations: Sequence[str] | None = None,
    **kwargs,
):
    skill = evaluate_configs(
        forecasts,
        observed,
        masks,
        significance_configurations=significance_configurations,
        **kwargs,
    )
    result = {"skill": skill}

    if persistence_configuration is not None:
        if primary_configuration not in forecasts:
            raise KeyError(primary_configuration)
        if persistence_configuration not in forecasts:
            raise KeyError(persistence_configuration)
        result["persistence_benchmark"] = evaluate_persistence_benchmark(
            forecasts[primary_configuration],
            forecasts[persistence_configuration],
            observed,
            masks,
            **kwargs,
        )

    return result


def configuration_summary(skill: xr.Dataset, first: str, second: str) -> pd.DataFrame:
    rows = []
    for region in skill.region.values:
        for metric in ["ACC", "CENTERED_ACC", "MSSS", "RMSE", "BIAS"]:
            a = float(skill[f"TARGET_{metric}"].sel(configuration=first, region=region).mean(skipna=True))
            b = float(skill[f"TARGET_{metric}"].sel(configuration=second, region=region).mean(skipna=True))
            rows.append(
                {
                    "region": str(region),
                    "metric": metric,
                    first: a,
                    second: b,
                    f"{first}_minus_{second}": a - b,
                }
            )
    return pd.DataFrame(rows)


def seasonal_lead_group_comparison(
    skill: xr.Dataset,
    first: str,
    second: str,
    regions: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for region in regions:
        for season in SEASON_MONTHS:
            for group, leads in LEAD_GROUPS.items():
                a_acc = float(
                    skill["SEASON_ACC"]
                    .sel(configuration=first, region=region, season=season, lead=leads)
                    .mean("lead", skipna=True)
                )
                b_acc = float(
                    skill["SEASON_ACC"]
                    .sel(configuration=second, region=region, season=season, lead=leads)
                    .mean("lead", skipna=True)
                )
                a_msss = float(
                    skill["SEASON_MSSS"]
                    .sel(configuration=first, region=region, season=season, lead=leads)
                    .mean("lead", skipna=True)
                )
                b_msss = float(
                    skill["SEASON_MSSS"]
                    .sel(configuration=second, region=region, season=season, lead=leads)
                    .mean("lead", skipna=True)
                )
                rows.append(
                    {
                        "region": region,
                        "season": season,
                        "lead_group": group,
                        "ACC_ONE": a_acc,
                        "ACC_THREE": b_acc,
                        "DACC": a_acc - b_acc,
                        "MSSS_ONE": a_msss,
                        "MSSS_THREE": b_msss,
                        "DMSSS": a_msss - b_msss,
                    }
                )
    return pd.DataFrame(rows)
