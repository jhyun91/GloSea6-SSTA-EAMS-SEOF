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
