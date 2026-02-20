#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import gstools as gs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------
# CONFIG
# ---------------------------
GPKG  = "/home/serviciosocial/iieg_2026/02_clima/insumos/estaciones.gpkg"
LAYER = "estaciones"

COSTA_GPKG  = "/home/serviciosocial/iieg_2026/02_clima/insumos/costa.gpkg"
COSTA_LAYER = None  # None = primera capa
COSTA_COL_CLASE = "CLASE"  # <- AJUSTA si tu campo se llama distinto
COSTA_OESTE_TAGS = {"OESTE", "PACIFICO", "PACÍFICO", "WEST"}
COSTA_ESTE_TAGS  = {"ESTE", "GOLFO", "CARIBE", "EAST"}

OUTDIR = "/home/serviciosocial/iieg_2026/02_clima/resultados/interpolaciones_ked2"
os.makedirs(OUTDIR, exist_ok=True)

DEM = "/home/serviciosocial/iieg_2026/00_sig/CEM_mex/CEM_15m_UTM13N_clip_aoi.tif"

# Grid objetivo
XMIN = 294250.4020848634
XMAX = 994250.4020848634
YMIN = 1927486.9180106297
YMAX = 2627486.9180106297
CELLSIZE = 1000

MESES = ("ENE","FEB","MAR","ABR","MAY","JUN","JUL","AGO","SEP","OCT","NOV","DIC")

BASE_SITUACION = "OPERANDO"
BASE_YEAR = 2025

TEMP_IDX = "TEMPERATURA MEDIA MENSUAL"
RAIN_IDX = "LLUVIA TOTAL MENSUAL"

COL_SITUACION = "Situación"
COL_YEAR      = "AÑO"
COL_INDICE    = "Índice"

# Diagnóstico
BIN_NO = 20
CV_SPLITS = 5
CV_SEED = 42

# Variograma: control de distancia
MAXDIST_FRAC = 0.5   # 0.35–0.60 recomendado; ajusta si tu AOI es raro
MIN_PTS = 10

# Recortes/transformaciones
TEMP_CLIP_BUFFER = 3.0   # °C
RAIN_MAX_FACTOR  = 1.5   # cap respecto p99 observado (back-transform)

# ---------------------------
# LOGGER
# ---------------------------
LOGDIR = Path(OUTDIR) / "_logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

PLOTDIR = Path(OUTDIR) / "_diagnosticos"
PLOTDIR.mkdir(parents=True, exist_ok=True)

HISTDIR = Path(OUTDIR) / "_hist"
HISTDIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("rk_kriging")
logger.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

_fh = logging.FileHandler(LOGDIR / "interpolacion_rk_2025.log", mode="w", encoding="utf-8")
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

# ---------------------------
# UTILIDADES
# ---------------------------
def build_grid():
    xs = np.arange(XMIN, XMAX + CELLSIZE, CELLSIZE)
    ys = np.arange(YMAX, YMIN - CELLSIZE, -CELLSIZE)
    return xs, ys

def sample_raster_at_points(raster_path, xs, ys):
    with rasterio.open(raster_path) as src:
        vals = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
        nod = src.nodata
    if nod is not None:
        vals[vals == nod] = np.nan
    return vals

def read_raster_grid(raster_path, grid_x, grid_y):
    X, Y = np.meshgrid(grid_x, grid_y)
    z = sample_raster_at_points(raster_path, X.ravel(), Y.ravel())
    return z.reshape(Y.shape)

def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas: {missing}. Disponibles: {list(df.columns)}")

def hist_plot(values, out_png, title, bins=30):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(v, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frecuencia")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def _domain_maxdist(x, y, frac=0.5):
    dx = float(np.nanmax(x) - np.nanmin(x))
    dy = float(np.nanmax(y) - np.nanmin(y))
    diam = np.sqrt(dx*dx + dy*dy)
    return frac * diam

def plot_variogram(bin_center, gamma, model, out_png, title="Semivariograma"):
    sill = float(getattr(model, "var", np.nan) + getattr(model, "nugget", 0.0))
    xs = np.linspace(np.nanmin(bin_center), np.nanmax(bin_center), 240)

    fig = plt.figure()
    ax = fig.add_subplot(111)
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
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def fit_variogram_model(x, y, v, max_dist=None, bin_no=20):
    coords = np.vstack([x, y]).T

    if max_dist is None:
        max_dist = _domain_maxdist(x, y, frac=MAXDIST_FRAC)

    # Empírico + bin counts (para pesos)
    # gstools.vario_estimate puede devolver counts si pides return_counts=True (según versión).
    # Lo intentamos; si no, seguimos sin pesos.
    try:
        bin_center, gamma, counts = gs.vario_estimate(coords.T, v, max_dist=max_dist, bin_no=bin_no, return_counts=True)
        w = counts.astype(float)
        w[w <= 0] = 1.0
    except TypeError:
        bin_center, gamma = gs.vario_estimate(coords.T, v, max_dist=max_dist, bin_no=bin_no)
        w = None

    candidates = [
        gs.Spherical(dim=2),
        gs.Exponential(dim=2),
        gs.Gaussian(dim=2),
    ]

    best = None
    best_rmse = np.inf

    for model in candidates:
        try:
            # Ponderado por #pares si está disponible
            if w is not None:
                model.fit_variogram(bin_center, gamma, nugget=True, weights=w)
            else:
                model.fit_variogram(bin_center, gamma, nugget=True)

            pred = model.variogram(bin_center)
            rmse = float(np.sqrt(np.nanmean((pred - gamma) ** 2)))
            if np.isfinite(rmse) and rmse < best_rmse:
                best_rmse = rmse
                best = model
        except Exception:
            continue

    if best is None:
        best = gs.Exponential(dim=2, var=float(np.nanvar(v)), len_scale=max_dist / 6, nugget=0.0)
        best_rmse = float("nan")

    return best, best_rmse, bin_center, gamma, max_dist

def standardize(arr, mu=None, sd=None):
    a = np.asarray(arr, dtype=float)
    if mu is None:
        mu = float(np.nanmean(a))
    if sd is None:
        sd = float(np.nanstd(a))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return (a - mu) / sd, mu, sd

def build_design_matrix(dem, d_w, d_e, add_intercept=True):
    cols = []
    if add_intercept:
        cols.append(np.ones_like(dem, dtype=float))
    cols.append(np.asarray(dem, dtype=float))
    if d_w is not None:
        cols.append(np.asarray(d_w, dtype=float))
    if d_e is not None:
        cols.append(np.asarray(d_e, dtype=float))
    X = np.vstack(cols).T
    return X

def ols_fit(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    return beta, yhat, resid

def cv_regression_kriging(x, y, z, cov_pts, cov_grid, gridx, gridy, diag_id, n_splits=5, seed=42):
    n = z.size
    if n < max(MIN_PTS, n_splits * 2):
        return None

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)

    y_true_all = []
    y_pred_all = []

    for test_idx in folds:
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=False)

        xtr, ytr, ztr = x[train_idx], y[train_idx], z[train_idx]
        xte, yte, zte = x[test_idx], y[test_idx], z[test_idx]

        Ctr = cov_pts[train_idx, :]
        Cte = cov_pts[test_idx, :]

        mtr = (
        np.isfinite(ztr) &
        np.isfinite(xtr) & np.isfinite(ytr) &
        np.all(np.isfinite(Ctr), axis=1)
        )
        mte = (
            np.isfinite(zte) &
            np.isfinite(xte) & np.isfinite(yte) &
            np.all(np.isfinite(Cte), axis=1)
        )

        xtr, ytr, ztr, Ctr = xtr[mtr], ytr[mtr], ztr[mtr], Ctr[mtr]
        xte, yte, zte, Cte = xte[mte], yte[mte], zte[mte], Cte[mte]

        # Si el fold queda mal condicionado, se omite
        if ztr.size < MIN_PTS or zte.size < 3:
            continue
        # Estandariza covariables con TRAIN (evita leakage)
        Ctr_s = np.zeros_like(Ctr, dtype=float)
        Cte_s = np.zeros_like(Cte, dtype=float)
        mus = []
        sds = []
        for j in range(Ctr.shape[1]):
            cj_s, mu, sd = standardize(Ctr[:, j])
            Ctr_s[:, j] = cj_s
            Cte_s[:, j] = (Cte[:, j] - mu) / sd
            mus.append(mu); sds.append(sd)

        # Regresión tendencia
        Xtr = np.column_stack([np.ones(Ctr_s.shape[0]), Ctr_s])
        Xte = np.column_stack([np.ones(Cte_s.shape[0]), Cte_s])
        beta, _, rtr = ols_fit(ztr, Xtr)

        # Variograma y kriging de residuales
        model, _, _, _, _ = fit_variogram_model(xtr, ytr, rtr, bin_no=BIN_NO)
        ok = gs.krige.Ordinary(model, (xtr, ytr), rtr)

        r_pred, _ = ok((xte, yte))
        z_pred = (Xte @ beta) + r_pred

        y_true_all.append(zte)
        y_pred_all.append(z_pred)

    if len(y_true_all) == 0:
        return None
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    err = y_pred - y_true

    mse = float(np.nanmean(err ** 2))
    var_y = float(np.nanvar(y_true))

    if np.isfinite(var_y) and var_y > 0:
        r2 = 1.0 - (mse / var_y)
    else:
        r2 = np.nan

    return {
        "n": int(y_true.size),
        "ME": float(np.nanmean(err)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.nanmean(np.abs(err))),
        "R2": float(r2),
    }

def regression_kriging_to_grid(x, y, z, cov_pts, cov_grid, gridx, gridy, out_tif, crs_epsg=None, diag_id=None):
    # Máscara válida
    m = np.isfinite(z) & np.all(np.isfinite(cov_pts), axis=1) & np.isfinite(x) & np.isfinite(y)
    x, y, z = x[m], y[m], z[m]
    C = cov_pts[m, :]

    if z.size < MIN_PTS:
        raise ValueError(f"Muy pocos puntos válidos ({z.size})")

    # Estandariza covariables (para estabilidad numérica)
    C_s = np.zeros_like(C, dtype=float)
    Cg_s = np.zeros_like(cov_grid, dtype=float)
    mus = []
    sds = []
    for j in range(C.shape[1]):
        cj_s, mu, sd = standardize(C[:, j])
        C_s[:, j] = cj_s
        Cg_s[:, j] = (cov_grid[:, j] - mu) / sd
        mus.append(mu); sds.append(sd)

    # Tendencia por OLS
    Xtr = np.column_stack([np.ones(C_s.shape[0]), C_s])
    beta, zhat, resid = ols_fit(z, Xtr)

    # Variograma en residuales
    model, v_rmse, bin_center, gamma, max_dist = fit_variogram_model(x, y, resid, bin_no=BIN_NO)

    sill = float(getattr(model, "var", np.nan) + getattr(model, "nugget", 0.0))
    logger.info(
        f"[{diag_id}] Variograma residuales {model.__class__.__name__}: "
        f"nugget={getattr(model,'nugget',np.nan):.4g} var={getattr(model,'var',np.nan):.4g} "
        f"sill≈{sill:.4g} len_scale={getattr(model,'len_scale',np.nan)} "
        f"max_dist={max_dist:.1f} RMSE_fit={v_rmse:.4g}"
    )

    if diag_id:
        try:
            plot_variogram(bin_center, gamma, model, str(PLOTDIR / f"{diag_id}_variograma.png"),
                           title=f"Semivariograma residuales - {diag_id}")
        except Exception as e:
            logger.warning(f"[{diag_id}] No se pudo graficar variograma: {e}")

    # OK sobre residuales
    ok = gs.krige.Ordinary(model, (x, y), resid)

    ny = len(gridy); nx = len(gridx)
    Xg, Yg = np.meshgrid(gridx, gridy)
    r_pred, _ = ok((Xg.ravel(), Yg.ravel()))
    r_field = r_pred.reshape((ny, nx))

    # Tendencia en grid
    Xg_tr = np.column_stack([np.ones(Cg_s.shape[0]), Cg_s])
    trend = (Xg_tr @ beta).reshape((ny, nx))

    field = trend + r_field

    # Escribir GeoTIFF
    transform = from_origin(XMIN, YMAX, CELLSIZE, CELLSIZE)
    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 1,
        "dtype": "float32",
        "crs": CRS.from_epsg(crs_epsg) if crs_epsg else None,
        "transform": transform,
        "nodata": -9999.0,
        "compress": "DEFLATE",
        "tiled": True,
    }

    out = np.where(np.isfinite(field), field.astype("float32"), profile["nodata"])
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(out, 1)

def discover_numeric_fields(sub, base_fields, exclude_cols):
    fields = list(base_fields)
    extra = []
    for c in sub.columns:
        if c in exclude_cols or c in fields or c == "geometry":
            continue
        s = pd.to_numeric(sub[c], errors="coerce")
        if int(np.isfinite(s.to_numpy(dtype=float)).sum()) >= MIN_PTS:
            extra.append(c)
    if extra:
        fields.extend(extra)
    return fields, extra

def load_costa_distances(stations_gdf):
    logger.info("Cargando costa y calculando distancias euclidianas por clase...")
    costa = gpd.read_file(COSTA_GPKG, layer=COSTA_LAYER) if COSTA_LAYER else gpd.read_file(COSTA_GPKG)

    if costa.crs is None:
        raise ValueError("La capa costa no tiene CRS definido.")
    if stations_gdf.crs is None:
        raise ValueError("La capa estaciones no tiene CRS definido.")

    costa = costa.to_crs(stations_gdf.crs)

    if COSTA_COL_CLASE not in costa.columns:
        raise KeyError(f"En costa falta la columna '{COSTA_COL_CLASE}'")

    # Normaliza clase
    cls = costa[COSTA_COL_CLASE].astype(str).str.strip().str.upper()
    costa["_CLASE_N"] = cls

    oeste = costa[costa["_CLASE_N"].isin(COSTA_OESTE_TAGS)]
    este  = costa[costa["_CLASE_N"].isin(COSTA_ESTE_TAGS)]

    if oeste.empty and este.empty:
        logger.warning("No se detectaron clases oeste/este; se calculará distancia a toda la costa como un solo predictor.")
        geom_all = costa.geometry.union_all()
        d_all = stations_gdf.geometry.distance(geom_all).to_numpy(dtype=float)
        return d_all, None, None  # dem, d_w, d_e (aquí d_all lo usas como d_w)
    else:
        geom_w = oeste.geometry.union_all() if not oeste.empty else None
        geom_e = este.geometry.union_all()  if not este.empty  else None
        d_w = stations_gdf.geometry.distance(geom_w).to_numpy(dtype=float) if geom_w is not None else None
        d_e = stations_gdf.geometry.distance(geom_e).to_numpy(dtype=float) if geom_e is not None else None
        return None, d_w, d_e

# ---------------------------
# MAIN
# ---------------------------
def main():
    t0 = time.time()
    logger.info("Inicio interpolación 2025 con Regression-Kriging (DEM + costa opcional).")

    gdf = gpd.read_file(GPKG, layer=LAYER)
    if gdf.crs is None:
        raise ValueError("Estaciones sin CRS definido.")

    require_columns(gdf, [COL_SITUACION, COL_YEAR, COL_INDICE])

    # Coordenadas
    gdf["X"] = gdf.geometry.x
    gdf["Y"] = gdf.geometry.y

    # Grid
    gridx, gridy = build_grid()
    logger.info(f"Grid: nx={len(gridx)} ny={len(gridy)} cell={CELLSIZE}")

    # DEM en puntos y grid
    logger.info("Muestreando DEM en puntos y grid...")
    gdf["DEM_Z"] = sample_raster_at_points(DEM, gdf["X"].to_numpy(), gdf["Y"].to_numpy())
    dem_grid = read_raster_grid(DEM, gridx, gridy)

    # Costa: distancias por clase
    d_all, d_w, d_e = load_costa_distances(gdf)

    # Prepara covariables en puntos (solo para precip; para temp puedes usar también)
    # Nota: si d_all no es None, lo tratamos como "distancia costa única".
    if d_all is not None:
        gdf["D_COAST"] = d_all
    if d_w is not None:
        gdf["D_WEST"] = d_w
    if d_e is not None:
        gdf["D_EAST"] = d_e

    # Prepara covariables en grid (distancias): calculamos distancia desde cada celda a costa(es)
    # OJO: esto puede ser pesado; con CELLSIZE=1000 suele ser razonable.
    logger.info("Calculando distancias a costa en grid (puede tardar)...")
    Xg, Yg = np.meshgrid(gridx, gridy)  # (ny, nx)
    grid_gs = gpd.GeoSeries(gpd.points_from_xy(Xg.ravel(), Yg.ravel()), crs=gdf.crs)

    # Para precip: usar costa(s) por clase; si no hay clases, usar costa total
    if d_all is not None:
        costa = gpd.read_file(COSTA_GPKG, layer=COSTA_LAYER) if COSTA_LAYER else gpd.read_file(COSTA_GPKG)
        costa = costa.to_crs(gdf.crs)
        geom_all = costa.geometry.union_all()  # <- en lugar de.union_all()
        dco_grid = grid_gs.distance(geom_all).to_numpy(dtype=float)
        cov_grid_coast = {"D_COAST": dco_grid}
    else:
        cov_grid_coast = {}
        costa = gpd.read_file(COSTA_GPKG, layer=COSTA_LAYER) if COSTA_LAYER else gpd.read_file(COSTA_GPKG)
        costa = costa.to_crs(gdf.crs)
        cls = costa[COSTA_COL_CLASE].astype(str).str.strip().str.upper()
        costa["_CLASE_N"] = cls
        oeste = costa[costa["_CLASE_N"].isin(COSTA_OESTE_TAGS)]
        este  = costa[costa["_CLASE_N"].isin(COSTA_ESTE_TAGS)]

        if not oeste.empty:
            geom_w = oeste.geometry.union_all()
            cov_grid_coast["D_WEST"] = grid_gs.distance(geom_w).to_numpy(dtype=float)
        if not este.empty:
            geom_e = este.geometry.union_all()
            cov_grid_coast["D_EAST"] = grid_gs.distance(geom_e).to_numpy(dtype=float)

    # Filtros base
    gdf["_SITUACION_STR"] = gdf[COL_SITUACION].astype(str).str.strip().str.upper()
    gdf["_INDICE_STR"] = gdf[COL_INDICE].astype(str).str.strip()
    gdf["_YEAR_NUM"] = pd.to_numeric(gdf[COL_YEAR], errors="coerce")

    epsg = gdf.crs.to_epsg()

    def process_index(idx_name, base_fields, out_prefix, is_rain=False):
        logger.info(f"INICIO índice='{idx_name}' lluvia={is_rain}")

        mask = (
            (gdf["_SITUACION_STR"] == BASE_SITUACION) &
            (gdf["_YEAR_NUM"] == BASE_YEAR) &
            (gdf["_INDICE_STR"] == idx_name)
        )
        sub = gdf.loc[mask].copy()
        logger.info(f"Filtrado base: n={len(sub)}")
        if sub.empty:
            return

        exclude_cols = {
            "X","Y","DEM_Z","geometry",
            COL_SITUACION,COL_YEAR,COL_INDICE,
            "_SITUACION_STR","_INDICE_STR","_YEAR_NUM",
            "D_COAST","D_WEST","D_EAST"
        }
        exclude_cols.update({"Lat", "Lon", "Altura", "MESES", "PROM"})
        
        fields, extra = discover_numeric_fields(sub, base_fields, exclude_cols)
        if extra:
            logger.info(f"Campos extra detectados: {extra}")

        # Define covariables a usar
        # Temperatura: DEM suele bastar; Precip: DEM + costa(s) suele aportar.
        use_coast = is_rain  # si también quieres para temp, cámbialo a True
        coast_cols = []
        if use_coast:
            if "D_COAST" in sub.columns:
                coast_cols = ["D_COAST"]
            else:
                for c in ("D_WEST","D_EAST"):
                    if c in sub.columns:
                        coast_cols.append(c)

        for fld in fields:
            if fld not in sub.columns:
                continue

            work = sub[["X","Y","DEM_Z", fld] + coast_cols].copy()
            work[fld] = pd.to_numeric(work[fld], errors="coerce")

            x = work["X"].to_numpy(dtype=float, copy=True)
            y = work["Y"].to_numpy(dtype=float, copy=True)
            z = work[fld].to_numpy(dtype=float, copy=True)
            dem_pts = work["DEM_Z"].to_numpy(dtype=float, copy=True)

            # Plausibilidad/outliers
            if is_rain:
                z[z < 0] = np.nan
            if np.isfinite(z).sum() >= MIN_PTS:
                p_lo, p_hi = np.nanpercentile(z, [0.5, 99.5])
                z = np.where((z >= p_lo) & (z <= p_hi), z, np.nan)

            n_ok = int(np.isfinite(z).sum())
            if n_ok < MIN_PTS:
                logger.warning(f"{fld}: muy pocos válidos ({n_ok})")
                continue

            # Histograma (observaciones)
            try:
                hist_plot(z, str(HISTDIR / f"{out_prefix}_{fld}_2025_obs_hist.png"),
                          title=f"Histograma obs - {out_prefix} {fld} 2025")
            except Exception as e:
                logger.warning(f"No se pudo hist {fld}: {e}")

            # Preparar covariables en puntos
            cov_list = [dem_pts]
            cov_names = ["DEM_Z"]
            for c in coast_cols:
                cov_list.append(pd.to_numeric(work[c], errors="coerce").to_numpy(float))
                cov_names.append(c)

            cov_pts = np.vstack(cov_list).T  # (n, p)

            # Preparar covariables en grid (mismo orden)
            dem_grid_flat = dem_grid.ravel().astype(float)
            covg_list = [dem_grid_flat]
            for c in coast_cols:
                covg_list.append(cov_grid_coast[c].astype(float))
            cov_grid = np.vstack(covg_list).T  # (ncell, p)

            diag_id = f"{out_prefix}_{fld}_2025_{'rain' if is_rain else 'temp'}"

            out_tif = os.path.join(OUTDIR, f"{out_prefix}_{fld}_2025_rk.tif")
            logger.info(f"{diag_id} -> {out_tif} | cov={cov_names}")

            # Transformaciones por variable
            if is_rain:
                # log1p en y
                z_k = np.log1p(z)

                # Interp (RK)
                regression_kriging_to_grid(x, y, z_k, cov_pts, cov_grid, gridx, gridy,
                                           out_tif, crs_epsg=epsg, diag_id=diag_id+"_log1p")

                # CV en escala log1p (reporta tal cual para consistencia)
                cv = cv_regression_kriging(x, y, z_k, cov_pts, cov_grid, gridx, gridy, diag_id=diag_id+"_log1p",
                                           n_splits=CV_SPLITS, seed=CV_SEED)
                if cv:
                    logger.info(
                        f"[{diag_id}_log1p] CV: n={cv['n']} "
                        f"ME={cv['ME']:.4g} RMSE={cv['RMSE']:.4g} "
                        f"MAE={cv['MAE']:.4g} R2={cv['R2']:.4g}"
                    )

                # Back-transform y cap
                with rasterio.open(out_tif) as src:
                    arr = src.read(1).astype(float)
                    prof = src.profile
                nod = prof.get("nodata", -9999.0)
                m = arr != nod

                arr_bt = arr.copy()
                arr_bt[m] = np.expm1(arr_bt[m])
                arr_bt[m] = np.maximum(arr_bt[m], 0.0)

                obs_p99 = np.nanpercentile(z, 99.0)
                cap = float(obs_p99 * RAIN_MAX_FACTOR) if np.isfinite(obs_p99) else None
                if cap is not None:
                    arr_bt[m] = np.minimum(arr_bt[m], cap)

                with rasterio.open(out_tif, "w", **prof) as dst:
                    dst.write(arr_bt.astype("float32"), 1)

            else:
                # Temperatura: sin detrend manual; RK ya modela tendencia con DEM
                regression_kriging_to_grid(x, y, z, cov_pts, cov_grid, gridx, gridy,
                                           out_tif, crs_epsg=epsg, diag_id=diag_id)

                cv = cv_regression_kriging(x, y, z, cov_pts, cov_grid, gridx, gridy, diag_id=diag_id,
                                           n_splits=CV_SPLITS, seed=CV_SEED)
                if cv:
                    logger.info(
                        f"[{diag_id}] CV: n={cv['n']} "
                        f"ME={cv['ME']:.4g} RMSE={cv['RMSE']:.4g} "
                        f"MAE={cv['MAE']:.4g} R2={cv['R2']:.4g}"
                    )


                # Clip suave al rango obs ± buffer
                with rasterio.open(out_tif) as src:
                    arr = src.read(1).astype(float)
                    prof = src.profile
                nod = prof.get("nodata", -9999.0)
                m = arr != nod
                z_min = float(np.nanmin(z))
                z_max = float(np.nanmax(z))
                arr[m] = np.clip(arr[m], z_min - TEMP_CLIP_BUFFER, z_max + TEMP_CLIP_BUFFER)
                with rasterio.open(out_tif, "w", **prof) as dst:
                    dst.write(arr.astype("float32"), 1)

        logger.info(f"FIN índice='{idx_name}'")

    process_index(TEMP_IDX, list(MESES) + ["PROM"], "temp_media", is_rain=False)
    process_index(RAIN_IDX, list(MESES) + ["ACUM"], "lluvia_total", is_rain=True)

    logger.info(f"Listo. OUTDIR={OUTDIR}")
    logger.info(f"Diagnósticos={PLOTDIR} | Hist={HISTDIR}")
    logger.info(f"Tiempo total: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
