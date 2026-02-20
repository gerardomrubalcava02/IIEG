from __future__ import annotations

import logging
import time
from pathlib import Path

import geopandas as gpd
import gstools as gs
import matplotlib
import numpy as np
import pandas as pd
import rasterio

from .config import AppConfig
from .interpolation import (
    build_grid,
    fit_variogram_model,
    ols_fit,
    read_raster_grid,
    regression_kriging_to_grid,
    sample_raster_at_points,
    standardize,
)

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


def plot_variogram(bin_center: np.ndarray, gamma: np.ndarray, model, out_png: Path, title: str) -> None:
    sill = float(getattr(model, "var", np.nan) + getattr(model, "nugget", 0.0))
    xs = np.linspace(np.nanmin(bin_center), np.nanmax(bin_center), 240)

    fig, ax = plt.subplots()
    ax.plot(bin_center, gamma, marker="o", linestyle="none", label="Empírico")
    ax.plot(xs, model.variogram(xs), label=f"Ajuste: {model.__class__.__name__}")
    if np.isfinite(sill):
        ax.axhline(sill, linestyle="--", linewidth=1.2, label=f"Meseta(sill)≈{sill:.3g}")
    ax.set_xlabel("Distancia")
    ax.set_ylabel("Semivarianza")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
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


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mse = float(np.nanmean(err ** 2))
    var_y = float(np.nanvar(y_true))
    r2 = float(1.0 - (mse / var_y)) if np.isfinite(var_y) and var_y > 0 else float("nan")
    return {
        "n": int(y_true.size),
        "ME": float(np.nanmean(err)),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.nanmean(np.abs(err))),
        "R2": r2,
    }


def cv_regression_kriging(
    x: np.ndarray,
    y: np.ndarray,
    z_obs: np.ndarray,
    cov_pts: np.ndarray,
    *,
    is_rain: bool,
    rain_max_factor: float,
    min_points: int,
    n_splits: int,
    seed: int,
    maxdist_frac: float,
    bin_no: int,
) -> dict[str, float] | None:
    n = z_obs.size
    if n < max(min_points, n_splits * 2):
        return None

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for test_idx in folds:
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=False)

        xtr, ytr = x[train_idx], y[train_idx]
        xte, yte = x[test_idx], y[test_idx]
        ztr_obs, zte_obs = z_obs[train_idx], z_obs[test_idx]
        Ctr, Cte = cov_pts[train_idx, :], cov_pts[test_idx, :]

        mtr = np.isfinite(ztr_obs) & np.isfinite(xtr) & np.isfinite(ytr) & np.all(np.isfinite(Ctr), axis=1)
        mte = np.isfinite(zte_obs) & np.isfinite(xte) & np.isfinite(yte) & np.all(np.isfinite(Cte), axis=1)

        xtr, ytr, ztr_obs, Ctr = xtr[mtr], ytr[mtr], ztr_obs[mtr], Ctr[mtr]
        xte, yte, zte_obs, Cte = xte[mte], yte[mte], zte_obs[mte], Cte[mte]

        if ztr_obs.size < min_points or zte_obs.size < 3:
            continue

        Ctr_s = np.zeros_like(Ctr, dtype=float)
        Cte_s = np.zeros_like(Cte, dtype=float)
        for j in range(Ctr.shape[1]):
            Ctr_s[:, j], mu, sd = standardize(Ctr[:, j])
            Cte_s[:, j] = (Cte[:, j] - mu) / sd

        ztr_model = np.log1p(ztr_obs) if is_rain else ztr_obs
        Xtr = np.column_stack([np.ones(Ctr_s.shape[0]), Ctr_s])
        Xte = np.column_stack([np.ones(Cte_s.shape[0]), Cte_s])
        beta, _, resid = ols_fit(ztr_model, Xtr)

        model, _, _, _, _ = fit_variogram_model(xtr, ytr, resid, maxdist_frac=maxdist_frac, bin_no=bin_no)
        ok = gs.krige.Ordinary(model, (xtr, ytr), resid)

        r_pred, _ = ok((xte, yte))
        z_pred_model = (Xte @ beta) + r_pred

        if is_rain:
            z_pred = np.maximum(np.expm1(z_pred_model), 0.0)
            cap = float(np.nanpercentile(ztr_obs, 99.0) * rain_max_factor)
            z_pred = np.minimum(z_pred, cap)
        else:
            z_pred = z_pred_model

        y_true_all.append(zte_obs)
        y_pred_all.append(z_pred)

    if not y_true_all:
        return None

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return _compute_metrics(y_true, y_pred)


def _save_cv_metrics_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    ordered_cols = ["variable", "indice", "campo", "n", "ME", "RMSE", "MAE", "R2"]
    cols = [c for c in ordered_cols if c in df.columns] + [c for c in df.columns if c not in ordered_cols]
    df = df[cols]
    df.to_csv(out_csv, index=False, encoding="utf-8")


def run_pipeline(cfg: AppConfig, logger: logging.Logger, plotdir: Path, histdir: Path) -> None:
    t0 = time.time()
    cv_rows: list[dict[str, object]] = []

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
            "X",
            "Y",
            "DEM_Z",
            "geometry",
            cfg.filters.col_situation,
            cfg.filters.col_year,
            cfg.filters.col_index,
            "_SITUACION_STR",
            "_INDICE_STR",
            "_YEAR_NUM",
            "D_COAST",
            "D_WEST",
            "D_EAST",
            "Lat",
            "Lon",
            "Altura",
            "MESES",
            "PROM",
        }
        fields, _ = discover_numeric_fields(sub, base_fields, exclude_cols, cfg.model.min_points)
        coast_cols = [c for c in ("D_COAST", "D_WEST", "D_EAST") if is_rain and c in sub.columns]

        for fld in fields:
            if fld not in sub.columns:
                continue

            work = sub[["X", "Y", "DEM_Z", fld] + coast_cols].copy()
            z = pd.to_numeric(work[fld], errors="coerce").to_numpy(dtype=float, copy=True)
            if is_rain:
                z[z < 0] = np.nan
            if np.isfinite(z).sum() < cfg.model.min_points:
                logger.warning("%s_%s: muy pocos datos válidos", prefix, fld)
                continue

            hist_plot(z, histdir / f"{prefix}_{fld}_obs_hist.png", f"Histograma obs - {prefix} {fld}")

            cov_pts = [work["DEM_Z"].to_numpy(dtype=float)]
            cov_grid = [dem_grid.ravel().astype(float)]
            for col in coast_cols:
                cov_pts.append(pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float))
                cov_grid.append(coast_grid[col].astype(float))

            cov_pts_np = np.vstack(cov_pts).T
            cov_grid_np = np.vstack(cov_grid).T
            x = work["X"].to_numpy(dtype=float)
            y = work["Y"].to_numpy(dtype=float)
            target = np.log1p(z) if is_rain else z

            valid = np.isfinite(target) & np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(cov_pts_np), axis=1)
            if int(valid.sum()) < cfg.model.min_points:
                logger.warning("%s_%s: muy pocos datos válidos tras máscara de covariables", prefix, fld)
                continue

            x_v = x[valid]
            y_v = y[valid]
            t_v = target[valid]
            C_v = cov_pts_np[valid, :]

            C_s = np.zeros_like(C_v, dtype=float)
            for j in range(C_v.shape[1]):
                C_s[:, j], _, _ = standardize(C_v[:, j])
            beta, _, resid = ols_fit(t_v, np.column_stack([np.ones(C_s.shape[0]), C_s]))
            model, _, bin_center, gamma, _ = fit_variogram_model(
                x_v,
                y_v,
                resid,
                maxdist_frac=cfg.model.maxdist_frac,
                bin_no=cfg.model.bin_no,
            )
            variogram_png = plotdir / f"{prefix}_{fld}_{cfg.filters.base_year}_variograma.png"
            plot_variogram(
                bin_center,
                gamma,
                model,
                variogram_png,
                title=f"Semivariograma residuales - {prefix} {fld} {cfg.filters.base_year}",
            )
            logger.info("Semivariograma guardado: %s", variogram_png)

            out_tif = str(Path(cfg.paths.outdir) / f"{prefix}_{fld}_{cfg.filters.base_year}_rk.tif")
            regression_kriging_to_grid(
                x=x,
                y=y,
                z=target,
                cov_pts=cov_pts_np,
                cov_grid=cov_grid_np,
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

            cv = cv_regression_kriging(
                x=x,
                y=y,
                z_obs=z,
                cov_pts=cov_pts_np,
                is_rain=is_rain,
                rain_max_factor=cfg.model.rain_max_factor,
                min_points=cfg.model.min_points,
                n_splits=cfg.model.cv_splits,
                seed=cfg.model.cv_seed,
                maxdist_frac=cfg.model.maxdist_frac,
                bin_no=cfg.model.bin_no,
            )

            if cv is None:
                logger.warning("CV no disponible para %s_%s", prefix, fld)
                continue

            cv_rows.append(
                {
                    "variable": prefix,
                    "indice": idx_name,
                    "campo": fld,
                    "n": cv["n"],
                    "ME": cv["ME"],
                    "RMSE": cv["RMSE"],
                    "MAE": cv["MAE"],
                    "R2": cv["R2"],
                }
            )
            logger.info(
                "[%s_%s] CV -> n=%s ME=%.4g RMSE=%.4g MAE=%.4g R2=%.4g",
                prefix,
                fld,
                cv["n"],
                cv["ME"],
                cv["RMSE"],
                cv["MAE"],
                cv["R2"],
            )

    process_index(cfg.filters.temp_index, list(cfg.months) + ["PROM"], "temp_media", is_rain=False)
    process_index(cfg.filters.rain_index, list(cfg.months) + ["ACUM"], "lluvia_total", is_rain=True)

    metrics_csv = plotdir / f"metricas_cv_{cfg.filters.base_year}.csv"
    _save_cv_metrics_csv(cv_rows, metrics_csv)
    if cv_rows:
        logger.info("CSV de métricas guardado: %s", metrics_csv)

    logger.info("Tiempo total: %.1fs", time.time() - t0)
