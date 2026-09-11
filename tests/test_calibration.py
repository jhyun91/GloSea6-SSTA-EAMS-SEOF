import numpy as np
import pandas as pd
import xarray as xr

from eams_seof.calibration import clim_correction


def test_correction_returns_archive_counts():
    time = pd.date_range("2013-01-01", periods=36, freq="MS")
    lat = [30.0, 31.0]
    lon = [125.0, 126.0]
    observed = xr.DataArray(
        np.ones((36, 2, 2), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
    )
    forecast = xr.DataArray(
        np.ones((2, 36, 2, 2), dtype=np.float32),
        dims=("lead", "target_time", "lat", "lon"),
        coords={"lead": [0, 1], "target_time": time, "lat": lat, "lon": lon},
    )
    init = pd.date_range("2014-01-01", periods=12, freq="MS")

    result = clim_correction(
        forecast,
        observed,
        initialization_times=init,
        leads=np.array([0, 1]),
        window_years=7,
        apply_from="2013-01-01",
        minimum_archive_months=12,
        archive_start="2013-01-01",
        return_diagnostics=True,
    )

    assert set(result.data_vars) == {"correction", "archive_sample_count", "applied", "target_month"}
    assert result["archive_sample_count"].dims == ("init", "lead")
    assert result["target_month"].dims == ("init", "lead")
    assert int(result["archive_sample_count"].max()) >= 1
