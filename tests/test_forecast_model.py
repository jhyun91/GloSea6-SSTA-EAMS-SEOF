import numpy as np
import pandas as pd
import xarray as xr

from eams_seof.seof import project_seof
from eams_seof.lag_weights import lag_weights


def test_gram_projection_recovers_coefficients_for_nonorthogonal_basis():
    time = pd.date_range("2000-01-01", periods=12, freq="MS")
    lat = np.array([25.0, 35.0])
    lon = np.array([125.0, 130.0])
    p1 = np.array([[1.0, 0.2], [0.4, -0.1]])
    p2 = np.array([[0.6, 0.1], [0.3, 0.7]])
    patterns = np.stack([p1, p2])
    patterns = np.repeat(patterns[:, None, :, :], 12, axis=1)
    patterns = xr.DataArray(
        patterns, dims=("mode", "month", "lat", "lon"),
        coords={"mode":[1,2], "month":np.arange(1,13), "lat":lat, "lon":lon},
    )
    mean = xr.zeros_like(patterns.isel(mode=0)).rename("seof_mean")
    true = np.column_stack([np.linspace(-1,1,12), np.linspace(0.8,-0.4,12)])
    field = np.einsum("tm,tmij->tij", true, patterns.transpose("month","mode","lat","lon").values)
    sst = xr.DataArray(field, dims=("time","lat","lon"), coords={"time":time,"lat":lat,"lon":lon})
    estimated = project_seof(sst, patterns, mean)
    np.testing.assert_allclose(estimated.values, true, atol=1e-5, rtol=1e-5)


def test_correlation_lag_model_has_identity_lead0_and_l1_normalized_forecast_leads():
    rng = np.random.default_rng(7)
    time = pd.date_range("1993-01-01", periods=20*12, freq="MS")
    x = np.zeros(time.size, dtype=float)
    x[0] = rng.normal()
    for i in range(1, time.size):
        x[i] = 0.65*x[i-1] + rng.normal(scale=0.5)
    scores = xr.DataArray(x[:,None], dims=("time","mode"), coords={"time":time,"mode":[1]})
    model = lag_weights(
        scores, training_period=("1993-01-01","2012-12-31"),
        maximum_lead_months=2, input_sequence_months=4,
        shrinkage_pseudocount=24, polarity_method="lag0_nonnegative",
    )
    w = model["lag_weight"]
    np.testing.assert_allclose(w.sel(lead=0, lag=0), 1.0)
    np.testing.assert_allclose(w.sel(lead=0, lag=slice(1,None)), 0.0)
    l1 = np.abs(w.sel(lead=[1,2])).sum("lag")
    np.testing.assert_allclose(l1, 1.0, atol=1e-6)
