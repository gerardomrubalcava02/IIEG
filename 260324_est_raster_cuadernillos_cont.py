#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import warnings
import math

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.validation import make_valid

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# CONFIGURACIÓN PARAMETRIZABLE
# =========================================================

# Tema de trabajo: "temperatura" o "precipitacion"
TEMA = "precipitacion"

# Capa de municipios
MUNICIPIOS_PATH = "/home/serviciosocial/iieg_2026/b_sig/limite_municipal.gpkg"
MUNICIPIOS_LAYER = None
MUNICIPIO_FIELD = "nombre"

# CRS proyectado para cálculo de áreas municipales
WORK_CRS = 6368

# Directorio de salida
OUT_DIR = f"/home/serviciosocial/iieg_2026/06_cuadernillos/Resultados/{TEMA}"

# Prefijo de salida
OUTPUT_PREFIX = TEMA

# Estadísticos a calcular
ESTADISTICOS = ["min", "max", "mean"]

# Diccionario de rasters por periodo
# Ajusta estas rutas a tus archivos reales.
RASTERS = {
    "ENE": "/home/serviciosocial/iieg_2026/02_clima/resultados/interpolaciones_ked_hist/lluvia_total_ENE_HIST_mean_rk.tif",
    "FEB": "/home/serviciosocial/iieg_2026/02_clima/resultados/interpolaciones_ked_hist/lluvia_total_FEB_HIST_mean_rk.tif",
    "MAR": "/home/serviciosocial/iieg_2026/02_clima/resultados/interpolaciones_ked_hist/lluvia_total_MAR_HIST_mean_rk.tif",
    "ABR": "/home/serviciosocial/iieg_2026/02_clima/resultados/interpolaciones_ked_hist/lluvia_total_ABR_HIST_mean_rk.tif",
    "MAY": "/home/serviciosocial/iieg_2026/02_clima/resultados/interpolaciones_ked_hist/lluvia_total_MAY_HIST_mean_rk.tif",
    "JUN": "/home/serviciosocial/iieg_2026/02_clima/resultados/interpolaciones_ked_hist/lluvia_total_JUN_HIST_mean_rk.tif",
    "JUL": "/home/serviciosocial/iieg_2026/02_clima/resultados/interpolaciones_ked_hist/lluvia_total_JUL_HIST_mean_rk.tif",
    "AGO": "/home/serviciosocial/iieg_2026/02_clima/resultados/interpolaciones_ked_hist/lluvia_total_AGO_HIST_mean_rk.tif",
    "SEP": "/home/serviciosocial/iieg_2026/02_clima/resultados/interpolaciones_ked_hist/lluvia_total_SEP_HIST_mean_rk.tif",
    "OCT": "/home/serviciosocial/iieg_2026/02_clima/resultados/interpolaciones_ked_hist/lluvia_total_OCT_HIST_mean_rk.tif",
    "NOV": "/home/serviciosocial/iieg_2026/02_clima/resultados/interpolaciones_ked_hist/lluvia_total_NOV_HIST_mean_rk.tif",
    "DIC": "/home/serviciosocial/iieg_2026/02_clima/resultados/interpolaciones_ked_hist/lluvia_total_DIC_HIST_mean_rk.tif",
    "ANUAL": "/home/serviciosocial/iieg_2026/02_clima/resultados/interpolaciones_ked_hist/lluvia_total_ACUM_HIST_mean_rk.tif",
}

# Si quieres ignorar periodos cuyo archivo no exista, dejar en True.
# Si quieres que falle al no encontrar uno, poner False.
SKIP_MISSING_RASTERS = True

# Redondeo final
ROUND_DECIMALS = 2

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def read_vector(path, layer=None):
    if layer:
        return gpd.read_file(path, layer=layer)
    return gpd.read_file(path)

def validate_fields(gdf, fields, gdf_name="capa"):
    missing = [f for f in fields if f not in gdf.columns]
    if missing:
        raise ValueError(f"En {gdf_name} faltan los campos: {missing}")

def fix_geometries(gdf):
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: make_valid(geom) if geom is not None else None
    )
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.is_valid].copy()
    return gdf.reset_index(drop=True)

def multipart_to_singlepart_safe(gdf):
    gdf = gdf.copy()
    try:
        gdf = gdf.explode(index_parts=False)
    except TypeError:
        gdf = gdf.explode()
    return gdf.reset_index(drop=True)

def build_municipios_area(gdf_munis, municipio_field):
    mun = gdf_munis[[municipio_field, "geometry"]].copy()
    mun["superficie_municipio_ha"] = mun.geometry.area / 10000.0
    mun["superficie_municipio_ha"] = mun["superficie_municipio_ha"].round(ROUND_DECIMALS)
    return mun[[municipio_field, "superficie_municipio_ha"]]

def raster_exists(path):
    return path is not None and os.path.exists(path)

def get_valid_data_from_masked_array(data, nodata):
    """
    data: ndarray con forma (1, rows, cols) o (rows, cols)
    retorna vector 1D de valores válidos
    """
    arr = data[0] if data.ndim == 3 else data
    arr = arr.astype("float64")

    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        arr[arr == nodata] = np.nan

    valid = arr[np.isfinite(arr)]
    return valid

def compute_stats(values, estadisticos):
    out = {}
    if values.size == 0:
        for est in estadisticos:
            out[est] = np.nan
        out["n_pixeles_validos"] = 0
        return out

    if "min" in estadisticos:
        out["min"] = float(np.min(values))
    if "max" in estadisticos:
        out["max"] = float(np.max(values))
    if "mean" in estadisticos:
        out["mean"] = float(np.mean(values))

    out["n_pixeles_validos"] = int(values.size)
    return out

def extract_stats_by_municipio(raster_path, gdf_munis, municipio_field, estadisticos):
    resultados = []

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        raster_crs = src.crs

        if raster_crs is None:
            raise ValueError(f"El raster no tiene CRS definido: {raster_path}")

        munis_raster = gdf_munis.to_crs(raster_crs)

        for idx, row in munis_raster.iterrows():
            municipio = row[municipio_field]
            geom = row.geometry

            try:
                out_image, out_transform = mask(
                    src,
                    [geom],
                    crop=True,
                    all_touched=False,
                    filled=True
                )

                vals = get_valid_data_from_masked_array(out_image, nodata)
                stats = compute_stats(vals, estadisticos)

                res = {
                    "municipio": municipio,
                    "n_pixeles_validos": stats.get("n_pixeles_validos", 0)
                }

                for est in estadisticos:
                    res[est] = stats.get(est, np.nan)

                resultados.append(res)

            except ValueError:
                # típicamente cuando no hay intersección
                res = {
                    "municipio": municipio,
                    "n_pixeles_validos": 0
                }
                for est in estadisticos:
                    res[est] = np.nan
                resultados.append(res)

    return pd.DataFrame(resultados)

def long_format_results(df_stats, tema, periodo, municipio_area_df, estadisticos):
    df = df_stats.copy()
    df["tema"] = tema
    df["periodo"] = periodo

    df = df.merge(
        municipio_area_df,
        left_on="municipio",
        right_on=MUNICIPIO_FIELD,
        how="left"
    ).drop(columns=[MUNICIPIO_FIELD])

    cols = ["municipio", "tema", "periodo", "superficie_municipio_ha", "n_pixeles_validos"] + estadisticos
    return df[cols].copy()

def pivot_results(df_long, estadisticos):
    """
    Convierte de formato largo a ancho:
    municipio, tema, superficie_municipio_ha, n_pixeles_validos_ENE, min_ENE, max_ENE, mean_ENE, ...
    """
    base_cols = ["municipio", "tema", "superficie_municipio_ha"]

    pivot_frames = []

    # n_pixeles_validos
    p_count = df_long.pivot_table(
        index=base_cols,
        columns="periodo",
        values="n_pixeles_validos",
        aggfunc="first"
    )
    p_count.columns = [f"n_pixeles_validos_{c}" for c in p_count.columns]
    pivot_frames.append(p_count)

    for est in estadisticos:
        p = df_long.pivot_table(
            index=base_cols,
            columns="periodo",
            values=est,
            aggfunc="first"
        )
        p.columns = [f"{est}_{c}" for c in p.columns]
        pivot_frames.append(p)

    out = pd.concat(pivot_frames, axis=1).reset_index()
    return out

# =========================================================
# PROCESO PRINCIPAL
# =========================================================

def main():
    ensure_dir(OUT_DIR)

    print("1. Cargando municipios...")
    gdf_munis = read_vector(MUNICIPIOS_PATH, MUNICIPIOS_LAYER)
    print(f"   Municipios cargados: {len(gdf_munis)}")

    validate_fields(gdf_munis, [MUNICIPIO_FIELD], "capa municipios")

    print("2. Corrigiendo geometrías...")
    gdf_munis = fix_geometries(gdf_munis)
    gdf_munis = multipart_to_singlepart_safe(gdf_munis)
    print(f"   Municipios válidos: {len(gdf_munis)}")

    print("3. Calculando área municipal...")
    if gdf_munis.crs is None:
        raise ValueError("La capa de municipios no tiene CRS definido.")

    gdf_munis_area = gdf_munis.to_crs(WORK_CRS)
    municipio_area_df = build_municipios_area(gdf_munis_area, MUNICIPIO_FIELD)

    all_results = []

    print("4. Procesando rasters por periodo...")
    for periodo, raster_path in RASTERS.items():
        print(f"   - {periodo}: {raster_path}")

        if not raster_exists(raster_path):
            msg = f"Raster no encontrado para {periodo}: {raster_path}"
            if SKIP_MISSING_RASTERS:
                print(f"     Aviso: {msg}. Se omite.")
                continue
            raise FileNotFoundError(msg)

        df_stats = extract_stats_by_municipio(
            raster_path=raster_path,
            gdf_munis=gdf_munis,
            municipio_field=MUNICIPIO_FIELD,
            estadisticos=ESTADISTICOS
        )

        df_long = long_format_results(
            df_stats=df_stats,
            tema=TEMA,
            periodo=periodo,
            municipio_area_df=municipio_area_df,
            estadisticos=ESTADISTICOS
        )

        all_results.append(df_long)

    if not all_results:
        raise ValueError("No se generaron resultados. Revisa rutas de rasters.")

    print("5. Consolidando resultados...")
    results_long = pd.concat(all_results, ignore_index=True)

    # redondeo
    for col in ["superficie_municipio_ha"] + ESTADISTICOS:
        if col in results_long.columns:
            results_long[col] = results_long[col].round(ROUND_DECIMALS)

    results_wide = pivot_results(results_long, ESTADISTICOS)

    for col in results_wide.columns:
        if col not in ["municipio", "tema"]:
            if pd.api.types.is_numeric_dtype(results_wide[col]):
                results_wide[col] = results_wide[col].round(ROUND_DECIMALS)

    # ordenar por municipio y periodo
    period_order = list(RASTERS.keys())
    results_long["periodo"] = pd.Categorical(results_long["periodo"], categories=period_order, ordered=True)
    results_long = results_long.sort_values(["municipio", "periodo"]).reset_index(drop=True)

    print("6. Exportando CSV...")
    out_long_csv = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_estadisticas_municipales_long.csv")
    out_wide_csv = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_estadisticas_municipales_wide.csv")

    results_long.to_csv(out_long_csv, index=False, encoding="utf-8-sig")
    results_wide.to_csv(out_wide_csv, index=False, encoding="utf-8-sig")

    print("\nProceso finalizado correctamente.")
    print(f"CSV largo: {out_long_csv}")
    print(f"CSV ancho: {out_wide_csv}")

if __name__ == "__main__":
    main()