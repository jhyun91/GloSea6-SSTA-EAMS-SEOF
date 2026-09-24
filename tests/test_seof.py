import numpy as np
import pandas as pd
import xarray as xr

from eams_seof.seof import compute_seof


def test_seof_year_centering():
    rng = np.random.default_rng(0)

    n_years = 6
    n_lat = 4
    n_lon = 5

    time = pd.date_range(
        "2000-01-01",
        periods=n_years * 12,
        freq="MS",
    )

    values = rng.normal(
        size=(n_years, 12, n_lat, n_lon)
    )

    monthly_offset = rng.normal(
        size=(12, n_lat, n_lon)
    )

    base = xr.DataArray(
        values.reshape(
            n_years * 12,
            n_lat,
            n_lon,
        ),
        dims=("time", "lat", "lon"),
        coords={
            "time": time,
            "lat": np.linspace(20, 50, n_lat),
            "lon": np.linspace(120, 140, n_lon),
        },
    )

    shifted = xr.DataArray(
        (
            values
            + monthly_offset[None, ...]
        ).reshape(
            n_years * 12,
            n_lat,
            n_lon,
        ),
        dims=base.dims,
        coords=base.coords,
    )

    ds0 = compute_seof(
        base,
        retained_modes=3,
    )

    ds1 = compute_seof(
        shifted,
        retained_modes=3,
    )

    np.testing.assert_allclose(
        ds0["explained_variance"],
        ds1["explained_variance"],
        rtol=1e-5,
        atol=1e-6,
    )

    assert float(
        np.abs(
            ds0["annual_pc"].mean("year")
        ).max()
    ) < 1e-5