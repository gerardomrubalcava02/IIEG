from __future__ import annotations

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
import gstools as gs


def build_grid(xmin: float, xmax: float, ymin: float, ymax: float, cellsize: float) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(xmin, xmax + cellsize, cellsize)
    ys = np.arange(ymax, ymin - cellsize, -cellsize)
    return xs, ys


def sample_raster_at_points(raster_path: str, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        values = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
        nodata = src.nodata
    if nodata is not None:
        values[values == nodata] = np.nan
    return values


def read_raster_grid(raster_path: str, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    X, Y = np.meshgrid(grid_x, grid_y)
    z = sample_raster_at_points(raster_path, X.ravel(), Y.ravel())
    return z.reshape(Y.shape)


def standardize(arr: np.ndarray, mu: float | None = None, sd: float | None = None) -> tuple[np.ndarray, float, float]:
    a = np.asarray(arr, dtype=float)
    mu = float(np.nanmean(a)) if mu is None else mu
    sd = float(np.nanstd(a)) if sd is None else sd
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return (a - mu) / sd, mu, sd


def ols_fit(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    return beta, yhat, y - yhat


def _domain_maxdist(x: np.ndarray, y: np.ndarray, frac: float) -> float:
    dx = float(np.nanmax(x) - np.nanmin(x))
    dy = float(np.nanmax(y) - np.nanmin(y))
    return frac * np.sqrt(dx * dx + dy * dy)


def fit_variogram_model(x: np.ndarray, y: np.ndarray, v: np.ndarray, maxdist_frac: float, max_dist: float | None = None, bin_no: int = 20):
    coords = np.vstack([x, y]).T
    if max_dist is None:
        max_dist = _domain_maxdist(x, y, frac=maxdist_frac)

    try:
        bin_center, gamma, counts = gs.vario_estimate(coords.T, v, max_dist=max_dist, bin_no=bin_no, return_counts=True)
        weights = np.maximum(counts.astype(float), 1.0)
    except TypeError:
        bin_center, gamma = gs.vario_estimate(coords.T, v, max_dist=max_dist, bin_no=bin_no)
        weights = None

    candidates = [gs.Spherical(dim=2), gs.Exponential(dim=2), gs.Gaussian(dim=2)]
    best_model = None
    best_rmse = np.inf

    for model in candidates:
        try:
            if weights is not None:
                model.fit_variogram(bin_center, gamma, nugget=True, weights=weights)
            else:
                model.fit_variogram(bin_center, gamma, nugget=True)
            pred = model.variogram(bin_center)
            rmse = float(np.sqrt(np.nanmean((pred - gamma) ** 2)))
            if np.isfinite(rmse) and rmse < best_rmse:
                best_model, best_rmse = model, rmse
        except Exception:
            continue

    if best_model is None:
        best_model = gs.Exponential(dim=2, var=float(np.nanvar(v)), len_scale=max_dist / 6, nugget=0.0)
        best_rmse = float("nan")

    return best_model, best_rmse, bin_center, gamma, max_dist


def regression_kriging_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cov_pts: np.ndarray,
    cov_grid: np.ndarray,
    gridx: np.ndarray,
    gridy: np.ndarray,
    out_tif: str,
    xmin: float,
    ymax: float,
    cellsize: float,
    maxdist_frac: float,
    bin_no: int,
    min_points: int,
    crs_epsg: int | None = None,
):
    m = np.isfinite(z) & np.all(np.isfinite(cov_pts), axis=1) & np.isfinite(x) & np.isfinite(y)
    x, y, z = x[m], y[m], z[m]
    C = cov_pts[m, :]
    if z.size < min_points:
        raise ValueError(f"Muy pocos puntos válidos ({z.size})")

    C_s = np.zeros_like(C, dtype=float)
    Cg_s = np.zeros_like(cov_grid, dtype=float)
    for j in range(C.shape[1]):
        C_s[:, j], mu, sd = standardize(C[:, j])
        Cg_s[:, j] = (cov_grid[:, j] - mu) / sd

    Xtr = np.column_stack([np.ones(C_s.shape[0]), C_s])
    beta, _, resid = ols_fit(z, Xtr)
    model, *_ = fit_variogram_model(x, y, resid, maxdist_frac=maxdist_frac, bin_no=bin_no)

    ok = gs.krige.Ordinary(model, (x, y), resid)
    ny = len(gridy)
    nx = len(gridx)
    Xg, Yg = np.meshgrid(gridx, gridy)
    r_pred, _ = ok((Xg.ravel(), Yg.ravel()))

    Xg_tr = np.column_stack([np.ones(Cg_s.shape[0]), Cg_s])
    trend = (Xg_tr @ beta).reshape((ny, nx))
    field = trend + r_pred.reshape((ny, nx))

    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 1,
        "dtype": "float32",
        "crs": CRS.from_epsg(crs_epsg) if crs_epsg else None,
        "transform": from_origin(xmin, ymax, cellsize, cellsize),
        "nodata": -9999.0,
        "compress": "DEFLATE",
        "tiled": True,
    }
    out = np.where(np.isfinite(field), field.astype("float32"), profile["nodata"])
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(out, 1)
