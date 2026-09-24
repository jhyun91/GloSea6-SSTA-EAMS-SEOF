from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
import xarray as xr


SEA_NAME_CANDIDATES = {
    "ys": ["Yellow Sea"],
    "ecs": ["Eastern China Sea", "East China Sea"],
    "ejs": ["Japan Sea", "East/Japan Sea", "East Sea"],
    "phs": ["Philippine Sea"],
}


def _union_named_geometry(gdf: gpd.GeoDataFrame, candidates: list[str]):
    for name in candidates:
        subset = gdf[gdf["NAME"] == name]
        if len(subset) > 0:
            geometry = subset.geometry
            return (
                geometry.union_all()
                if hasattr(geometry, "union_all")
                else geometry.unary_union
            )
    raise ValueError(
        "None of the requested sea names were found: "
        + ", ".join(candidates)
    )


def build_region_masks(
    lon: xr.DataArray,
    lat: xr.DataArray,
    shapefile: str | Path,
) -> xr.Dataset:
    'Build IHO-based regional masks on the SST grid.'
    shapefile = Path(shapefile)
    if not shapefile.exists():
        raise FileNotFoundError(
            f"IHO shapefile not found: {shapefile}\n"
            "Place World_Seas_IHO_v3.shp and companion files under "
            "data/external/World_Seas_IHO_v3/."
        )

    gdf = gpd.read_file(shapefile)
    if "NAME" not in gdf.columns:
        raise ValueError("Expected a 'NAME' field in the IHO shapefile.")

    lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    points = shapely.points(lon2d.ravel(), lat2d.ravel())

    masks = {}
    for short_name, candidates in SEA_NAME_CANDIDATES.items():
        geom = _union_named_geometry(gdf, candidates)
        covered = shapely.covers(geom, points)
        masks[short_name] = covered.reshape(
            lat.size, lon.size
        ).astype(np.uint8)

    eams = (
        (masks["ys"] == 1)
        | (masks["ecs"] == 1)
        | (masks["ejs"] == 1)
    ).astype(np.uint8)

    categorical = np.zeros((lat.size, lon.size), dtype=np.uint8)
    categorical[masks["ys"] == 1] = 1
    categorical[masks["ejs"] == 1] = 2
    categorical[masks["ecs"] == 1] = 3
    categorical[masks["phs"] == 1] = 4

    ds = xr.Dataset(
        data_vars={
            "region_mask": (("lat", "lon"), categorical),
            "region_mask_ys": (("lat", "lon"), masks["ys"]),
            "region_mask_ejs": (("lat", "lon"), masks["ejs"]),
            "region_mask_ecs": (("lat", "lon"), masks["ecs"]),
            "region_mask_phs": (("lat", "lon"), masks["phs"]),
            "region_mask_eams": (("lat", "lon"), eams),
        },
        coords={"lat": lat, "lon": lon},
    )

    ds["region_mask"].attrs["description"] = (
        "0: outside; 1: YS; 2: EJS; 3: ECS; 4: PHS"
    )
    ds["region_mask_eams"].attrs["description"] = (
        "Union of YS, ECS and EJS; PHS excluded"
    )
    ds.attrs["region_definition"] = "IHO World Seas v3 polygons"
    return ds
