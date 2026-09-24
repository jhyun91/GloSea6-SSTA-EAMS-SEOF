"""SEOF-based seasonal SST prediction for the East Asian marginal seas."""

from .config import load_config
from .preprocessing import (
    load_oisst,
    smooth_sst,
    detrend_anomaly,
)
from .regions import build_region_masks
from .seof import (
    compute_seof,
    align_seof,
    project_seof,
    weighted_norms,
    weighted_gram,
)
from .lag_weights import lag_weights
from .forecast import (
    forecast_scores,
    reconstruct_forecast,
    restore_state,
    forecast_to_target,
    fixed_base_anomaly,
)
from .calibration import clim_correction
