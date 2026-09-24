import numpy as np
import pandas as pd
import xarray as xr

from eams_seof.forecast import fixed_base_anomaly
from eams_seof.verification import (
    evaluate_verification,
    seasonal_lead_group_comparison,
)


def _tiny_field(time, seed=0):
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(size=(len(time), 2, 2)).astype(np.float32),
        dims=("target_time", "lat", "lon"),
        coords={"target_time": time, "lat": [30.0, 40.0], "lon": [125.0, 135.0]},
    )


def test_fixed_base_anomaly_accepts_explicit_reference_name():
    time = pd.date_range("1993-01-01", periods=24, freq="MS")
    sst = xr.DataArray(
        np.arange(24, dtype=np.float32)[:, None, None],
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [35.0], "lon": [130.0]},
    )
    target = sst.rename(time="target_time").expand_dims(lead=[1])
    a, o = fixed_base_anomaly(
        target,
        sst,
        anomaly_reference_period=("1993-01-01", "1994-12-31"),
    )
    assert a.dims == ("lead", "target_time", "lat", "lon")
    assert o.dims == ("time", "lat", "lon")


def test_revised_verification_api_and_mode_comparison():
    time = pd.date_range("2013-01-01", periods=12 * 13, freq="MS")
    obs = _tiny_field(time, seed=1)
    leads = np.arange(1, 3)
    model = xr.concat([(0.8 * obs).expand_dims(lead=[int(L)]) for L in leads], dim="lead")
    persistence = xr.concat([(0.5 * obs).expand_dims(lead=[int(L)]) for L in leads], dim="lead")
    mask = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("lat", "lon"),
        coords={"lat": obs.lat, "lon": obs.lon},
    )
    masks = {"EAMS": mask, "YS": mask, "ECS": mask, "EJS": mask}
    forecasts = {
        "corrected": model,
        "uncorrected": model * 0.98,
        "one_mode": model * 0.95,
        "persistence": persistence,
    }
    result = evaluate_verification(
        forecasts=forecasts,
        observed=obs,
        masks=masks,
        primary_configuration="corrected",
        persistence_configuration="persistence",
        significance_configurations=["corrected", "uncorrected", "persistence"],
        verification_period=("2013-01-01", "2025-12-31"),
        min_samples=5,
        n_null=20,
        fdr_q=0.10,
    )
    skill = result["skill"]
    assert "TARGET_SIG_ACC" in skill
    assert "TARGET_SIG_MSSS" in skill
    assert "TARGET_PSS" in result["persistence_benchmark"]

    table = seasonal_lead_group_comparison(
        skill,
        first="one_mode",
        second="uncorrected",
        regions=["EAMS"],
    )
    assert set(["season", "lead_group", "DACC", "DMSSS"]).issubset(table.columns)
    assert len(table) == 12
