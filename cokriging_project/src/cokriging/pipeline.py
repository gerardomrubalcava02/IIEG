from __future__ import annotations

import logging
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio

from .config import AppConfig
from .interpolation import build_grid, read_raster_grid, regression_kriging_to_grid, sample_raster_at_points

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas: {missing}. Disponibles: {list(df.columns)}")


def hist_plot(values: np.ndarray, out_png: Path, title: str, bins: int = 30) -> None:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    fig, ax = plt.subplots()
    ax.hist(v, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frecuencia")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def discover_numeric_fields(sub: pd.DataFrame, base_fields: list[str], exclude_cols: set[str], min_points: int) -> tuple[list[str], list[str]]:
    fields = list(base_fields)
    extra = []
    for c in sub.columns:
        if c in exclude_cols or c in fields or c == "geometry":
            continue
        s = pd.to_numeric(sub[c], errors="coerce")
        if int(np.isfinite(s.to_numpy(dtype=float)).sum()) >= min_points:
            extra.append(c)
    if extra:
        fields.extend(extra)
    return fields, extra


def _load_coast_distances(stations_gdf: gpd.GeoDataFrame, cfg: AppConfig, logger: logging.Logger):
    logger.info("Cargando costa y calculando distancias...")
    coast = gpd.read_file(cfg.paths.coast_gpkg, layer=cfg.paths.coast_layer) if cfg.paths.coast_layer else gpd.read_file(cfg.paths.coast_gpkg)
    coast = coast.to_crs(stations_gdf.crs)

    class_col = cfg.coast.class_column
    if class_col not in coast.columns:
        raise KeyError(f"En costa falta la columna '{class_col}'")

    coast["_CLASE_N"] = coast[class_col].astype(str).str.strip().str.upper()
    west = coast[coast["_CLASE_N"].isin(cfg.coast.west_tags)]
    east = coast[coast["_CLASE_N"].isin(cfg.coast.east_tags)]

    if west.empty and east.empty:
        geom_all = coast.geometry.union_all()
        d_all = stations_gdf.geometry.distance(geom_all).to_numpy(dtype=float)
        return d_all, None, None

    geom_w = west.geometry.union_all() if not west.empty else None
    geom_e = east.geometry.union_all() if not east.empty else None
    d_w = stations_gdf.geometry.distance(geom_w).to_numpy(dtype=float) if geom_w is not None else None
    d_e = stations_gdf.geometry.distance(geom_e).to_numpy(dtype=float) if geom_e is not None else None
    return None, d_w, d_e


def run_pipeline(cfg: AppConfig, logger: logging.Logger, plotdir: Path, histdir: Path) -> None:
    t0 = time.time()
    gdf = gpd.read_file(cfg.paths.stations_gpkg, layer=cfg.paths.stations_layer)
    require_columns(gdf, [cfg.filters.col_situation, cfg.filters.col_year, cfg.filters.col_index])

    gdf["X"] = gdf.geometry.x
    gdf["Y"] = gdf.geometry.y

    gridx, gridy = build_grid(cfg.domain.xmin, cfg.domain.xmax, cfg.domain.ymin, cfg.domain.ymax, cfg.domain.cellsize)
    gdf["DEM_Z"] = sample_raster_at_points(cfg.paths.dem_raster, gdf["X"].to_numpy(), gdf["Y"].to_numpy())
    dem_grid = read_raster_grid(cfg.paths.dem_raster, gridx, gridy)

    d_all, d_w, d_e = _load_coast_distances(gdf, cfg, logger)
    if d_all is not None:
        gdf["D_COAST"] = d_all
    if d_w is not None:
        gdf["D_WEST"] = d_w
    if d_e is not None:
        gdf["D_EAST"] = d_e

    Xg, Yg = np.meshgrid(gridx, gridy)
    grid_gs = gpd.GeoSeries(gpd.points_from_xy(Xg.ravel(), Yg.ravel()), crs=gdf.crs)
    coast_grid = {}
    if "D_COAST" in gdf.columns:
        coast = gpd.read_file(cfg.paths.coast_gpkg, layer=cfg.paths.coast_layer) if cfg.paths.coast_layer else gpd.read_file(cfg.paths.coast_gpkg)
        coast_grid["D_COAST"] = grid_gs.distance(coast.to_crs(gdf.crs).geometry.union_all()).to_numpy(dtype=float)
    else:
        coast = gpd.read_file(cfg.paths.coast_gpkg, layer=cfg.paths.coast_layer) if cfg.paths.coast_layer else gpd.read_file(cfg.paths.coast_gpkg)
        coast = coast.to_crs(gdf.crs)
        coast["_CLASE_N"] = coast[cfg.coast.class_column].astype(str).str.strip().str.upper()
        for key, tags in (("D_WEST", cfg.coast.west_tags), ("D_EAST", cfg.coast.east_tags)):
            side = coast[coast["_CLASE_N"].isin(tags)]
            if not side.empty:
                coast_grid[key] = grid_gs.distance(side.geometry.union_all()).to_numpy(dtype=float)

    gdf["_SITUACION_STR"] = gdf[cfg.filters.col_situation].astype(str).str.strip().str.upper()
    gdf["_INDICE_STR"] = gdf[cfg.filters.col_index].astype(str).str.strip()
    gdf["_YEAR_NUM"] = pd.to_numeric(gdf[cfg.filters.col_year], errors="coerce")

    def process_index(idx_name: str, base_fields: list[str], prefix: str, is_rain: bool):
        mask = (
            (gdf["_SITUACION_STR"] == cfg.filters.base_situation)
            & (gdf["_YEAR_NUM"] == cfg.filters.base_year)
            & (gdf["_INDICE_STR"] == idx_name)
        )
        sub = gdf.loc[mask].copy()
        if sub.empty:
            return

        exclude_cols = {
            "X", "Y", "DEM_Z", "geometry", cfg.filters.col_situation, cfg.filters.col_year, cfg.filters.col_index,
            "_SITUACION_STR", "_INDICE_STR", "_YEAR_NUM", "D_COAST", "D_WEST", "D_EAST", "Lat", "Lon", "Altura", "MESES", "PROM",
        }
        fields, _ = discover_numeric_fields(sub, base_fields, exclude_cols, cfg.model.min_points)
        coast_cols = [c for c in ("D_COAST", "D_WEST", "D_EAST") if is_rain and c in sub.columns]

        for fld in fields:
            if fld not in sub.columns:
                continue
            work = sub[["X", "Y", "DEM_Z", fld] + coast_cols].copy()
            # `to_numpy` puede devolver una vista read-only (dependiendo de versión/CoW);
            # pedimos copia explícita para permitir asignaciones in-place (e.g. z[z < 0] = np.nan).
            z = pd.to_numeric(work[fld], errors="coerce").to_numpy(dtype=float, copy=True)
            if is_rain:
                z[z < 0] = np.nan
            if np.isfinite(z).sum() < cfg.model.min_points:
                continue

            hist_plot(z, histdir / f"{prefix}_{fld}_obs_hist.png", f"Histograma obs - {prefix} {fld}")

            cov_pts = [work["DEM_Z"].to_numpy(dtype=float)]
            cov_grid = [dem_grid.ravel().astype(float)]
            for col in coast_cols:
                cov_pts.append(pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float))
                cov_grid.append(coast_grid[col].astype(float))

            out_tif = str(Path(cfg.paths.outdir) / f"{prefix}_{fld}_{cfg.filters.base_year}_rk.tif")
            target = np.log1p(z) if is_rain else z
            regression_kriging_to_grid(
                x=work["X"].to_numpy(dtype=float),
                y=work["Y"].to_numpy(dtype=float),
                z=target,
                cov_pts=np.vstack(cov_pts).T,
                cov_grid=np.vstack(cov_grid).T,
                gridx=gridx,
                gridy=gridy,
                out_tif=out_tif,
                xmin=cfg.domain.xmin,
                ymax=cfg.domain.ymax,
                cellsize=cfg.domain.cellsize,
                maxdist_frac=cfg.model.maxdist_frac,
                bin_no=cfg.model.bin_no,
                min_points=cfg.model.min_points,
                crs_epsg=gdf.crs.to_epsg(),
            )

            with rasterio.open(out_tif) as src:
                arr = src.read(1).astype(float)
                prof = src.profile
            nod = prof.get("nodata", -9999.0)
            m = arr != nod

            if is_rain:
                arr[m] = np.maximum(np.expm1(arr[m]), 0.0)
                cap = float(np.nanpercentile(z, 99.0) * cfg.model.rain_max_factor)
                arr[m] = np.minimum(arr[m], cap)
            else:
                arr[m] = np.clip(arr[m], np.nanmin(z) - cfg.model.temp_clip_buffer, np.nanmax(z) + cfg.model.temp_clip_buffer)

            with rasterio.open(out_tif, "w", **prof) as dst:
                dst.write(arr.astype("float32"), 1)

    process_index(cfg.filters.temp_index, list(cfg.months) + ["PROM"], "temp_media", is_rain=False)
    process_index(cfg.filters.rain_index, list(cfg.months) + ["ACUM"], "lluvia_total", is_rain=True)

    logger.info("Tiempo total: %.1fs", time.time() - t0)
