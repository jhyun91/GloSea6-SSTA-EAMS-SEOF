import numpy as np
import xarray as xr

from eams_seof.seof import (
    align_seof,
    weighted_norms,
)


def test_monthly_alignment_unit_norm():
    rng = np.random.default_rng(0)

    patterns = xr.DataArray(
        rng.normal(size=(3, 12, 8, 10)),
        dims=("mode", "month", "lat", "lon"),
        coords={
            "mode": [1, 2, 3],
            "month": np.arange(1, 13),
            "lat": np.linspace(20, 50, 8),
            "lon": np.linspace(120, 140, 10),
        },
    )

    aligned = align_seof(patterns)
    norms = weighted_norms(aligned)

    assert float(np.abs(norms - 1.0).max()) < 1e-5


def test_dual_projection_round_trip():
    import pandas as pd

    from eams_seof.seof import project_seof

    lat = xr.DataArray([25.0, 35.0], dims=("lat",), name="lat")
    lon = xr.DataArray([125.0, 135.0], dims=("lon",), name="lon")
    month = xr.DataArray(np.arange(1, 13), dims=("month",), name="month")
    mode = xr.DataArray([1, 2], dims=("mode",), name="mode")

    p1 = np.array([[1.0, 0.0], [0.0, 0.0]])
    p2 = np.array([[0.8, 0.6], [0.0, 0.0]])
    patterns = np.stack([p1, p2])
    patterns = np.repeat(patterns[:, None, :, :], 12, axis=1)
    patterns = xr.DataArray(
        patterns,
        dims=("mode", "month", "lat", "lon"),
        coords={"mode": mode, "month": month, "lat": lat, "lon": lon},
    )

    w = np.cos(np.deg2rad(lat.values))[:, None]
    norms = np.sqrt(np.sum(patterns.values**2 * w[None, None, :, :], axis=(2, 3)))
    patterns = patterns / xr.DataArray(
        norms,
        dims=("mode", "month"),
        coords={"mode": mode, "month": month},
    )

    coeff = np.array([0.7, -0.2])
    time = pd.date_range("2000-01-01", periods=12, freq="MS")
    fields = np.stack([
        coeff[0] * patterns.sel(month=t.month, mode=1).values
        + coeff[1] * patterns.sel(month=t.month, mode=2).values
        for t in time
    ])
    x = xr.DataArray(
        fields,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
    )
    mean = xr.zeros_like(patterns.isel(mode=0, drop=True))

    scores = project_seof(x, patterns, mean)
    np.testing.assert_allclose(scores.values, np.tile(coeff, (12, 1)), atol=1e-6)
