import numpy as np
import xarray as xr

from eams_seof.lag_weights import regularize_lag
from eams_seof.verification import acc, acc_centered


def test_lag0_nonnegative_flips_the_full_sequence():
    raw = xr.DataArray(
        [-0.4, 0.2, -0.1],
        dims=("lag",),
        coords={"lag": [0, 1, 2]},
    )

    regularized = regularize_lag(raw, method="lag0_nonnegative")

    np.testing.assert_allclose(regularized, -raw)
    assert float(regularized.sel(lag=0)) >= 0.0


def test_raw_signed_keeps_the_original_sequence():
    raw = xr.DataArray(
        [-0.4, 0.2, -0.1],
        dims=("lag",),
        coords={"lag": [0, 1, 2]},
    )

    np.testing.assert_allclose(
        regularize_lag(raw, method="raw_signed"),
        raw,
    )


def test_primary_acc_is_uncentered_anomaly_correlation():
    forecast = np.array([1.0, 2.0, 3.0])
    observed = np.array([2.0, 2.0, 4.0])

    expected = np.sum(forecast * observed) / np.sqrt(
        np.sum(forecast ** 2) * np.sum(observed ** 2)
    )

    assert np.isclose(acc(forecast, observed), expected)
    assert not np.isclose(acc(forecast, observed), acc_centered(forecast, observed))
