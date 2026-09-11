from __future__ import annotations

from pathlib import Path
import json
import xarray as xr


def save_dataset(
    dataset: xr.Dataset,
    path: str | Path,
    attrs: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = dataset.copy()
    if attrs:
        out.attrs.update(
            {
                k: (
                    json.dumps(v)
                    if isinstance(v, (dict, list, tuple))
                    else v
                )
                for k, v in attrs.items()
            }
        )

    out.to_netcdf(path)
    return path
