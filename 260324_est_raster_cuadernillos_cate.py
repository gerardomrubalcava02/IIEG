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

# Nombre del tema
TEMA = "clima"

# Raster de entrada
RASTER_PATH = "/home/serviciosocial/iieg_2026/02_clima/resultados/koppen_garcia_operativo/HIST/koppen_garcia_operativo_HIST_con_m.tif"

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

# ---------------------------------------------------------
# MODO DE TRABAJO
# ---------------------------------------------------------
# "categorico"    -> el raster ya viene con clases enteras/categóricas
# "reclasificar"  -> el raster es continuo y se reclasifica con cortes
RASTER_MODE = "categorico"

# ---------------------------------------------------------
# OPCIÓN A: raster ya categórico
# ---------------------------------------------------------
# Etiquetas opcionales para clases existentes del raster.
# Si no se define una clase aquí, se usará su valor numérico como texto.
# CATEGORY_LABELS = {
#     # ejemplo:
#     1: "Sin sequía",
#     2: "Anormalmente seco",
#     3: "Sequía moderada",
#     4: "Sequía severa",
#     5: "Sequía extrema",
#     6: "Sequía excepcional"
# }

# CATEGORY_LABELS = {    # Pendientes
#     1: "0 a 2%",
#     2: "Mayor a 2% y hasta 5%",
#     3: "Mayor a 5% y hasta 10%",
#     4: "Mayor a 10% y hasta 15%",
#     5: "Mayor a 15% y hasta 30%",
#     6: "Mayor a 30%",
# }

CATEGORY_LABELS = {    # Climas (clasificación tipo Köppen modificada)
    1: "BS1kw",
    2: "BS1hw",
    3: "BS1(h')w",
    4: "BS1(h')hw",
    5: "Aw0",
    6: "Aw1",
    7: "Aw2",
    8: "Am",
    9: "A(C)w0",
    10: "A(C)w1",
    11: "A(C)w2",
    12: "A(C)m",
    13: "(A)Ca(w0)",
    14: "(A)Ca(w1)",
    15: "(A)Ca(w2)",
    16: "(A)Ca(m)",
    17: "(A)Cb(w0)",
    18: "(A)Cb(w1)",
    19: "(A)Cb(w2)",
    20: "(A)Cb(m)",
    21: "Ca(w0)",
    22: "Ca(w1)",
    23: "Cb(w0)",
    24: "Cb(w1)",
    25: "Cb(w2)",
    26: "Cb(m)",
    27: "C(b')(w2)",
    28: "Cc(w2)",
    29: "E(T)C",
    30: "NC",
}

# ---------------------------------------------------------
# OPCIÓN B: raster continuo a reclasificar
# ---------------------------------------------------------
# Regla: [min_incl, max_excl, valor_clase, etiqueta]
# Usa np.inf o -np.inf si hace falta.
# RECLASS_RULES = [
#     [-np.inf, 0,    1, "No susceptible"],
#     [0,       5,    2, "Erosión muy baja"],
#     [5,       10,   3, "Erosión baja"],
#     [10,      25,   4, "Erosión leve"],
#     [25,      50,   5, "Erosión moderada"],
#     [50,      100,  6, "Erosión grave"],
#     [100,     200,  7, "Erosión muy grave"],
#     [200,     np.inf, 8, "Erosión extrema"],
# ]

# RECLASS_RULES = [
#     [-np.inf, 0.22, 1, "Sin vegetación / superficies no vegetadas"],
#     [0.22,    0.35, 2, "Vegetación muy escasa"],
#     [0.35,    0.50, 3, "Vegetación escasa"],
#     [0.50,    0.65, 4, "Vegetación moderada"],
#     [0.65,    np.inf, 5, "Vegetación densa"],
# ]

# RECLASS_RULES = [
#     [-np.inf, -0.10, 1, "Vegetación muy seca / estrés hídrico alto"],
#     [-0.10,   -0.05, 2, "Vegetación seca"],
#     [-0.05,    0.05, 3, "Condición hídrica baja"],
#     [0.05,     0.15, 4, "Condición hídrica moderada"],
#     [0.15,     np.inf, 5, "Alta humedad en vegetación"],
# ]

# Exportar raster reclasificado completo
EXPORT_RECLASSIFIED_RASTER = True

# Nombre del raster reclasificado de salida
RECLASSIFIED_RASTER_NAME = f"{OUTPUT_PREFIX}_reclasificado.tif"

# ---------------------------------------------------------
# PARÁMETROS DE EXTRACCIÓN
# ---------------------------------------------------------
# False = más conservador
# True  = incluye cualquier pixel tocado por el polígono
ALL_TOUCHED = False

# Redondeo final
ROUND_DECIMALS = 2

# Si el raster no existe, fallar
FAIL_IF_RASTER_MISSING = True

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
    mun = mun.dissolve(by=municipio_field, as_index=False)
    mun["superficie_municipio_ha"] = mun.geometry.area / 10000.0
    mun["superficie_municipio_ha"] = mun["superficie_municipio_ha"].round(ROUND_DECIMALS)
    return mun[[municipio_field, "superficie_municipio_ha"]]

def raster_exists(path):
    return path is not None and os.path.exists(path)

def get_pixel_area_ha(src):
    transform = src.transform
    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)
    pixel_area_m2 = pixel_width * pixel_height
    return pixel_area_m2 / 10000.0

def get_valid_array(data, nodata):
    arr = data[0] if data.ndim == 3 else data
    arr = arr.astype("float64")

    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        arr[arr == nodata] = np.nan

    return arr

def validate_reclass_rules(rules):
    if not rules:
        raise ValueError("RECLASS_RULES está vacío.")
    for i, rule in enumerate(rules):
        if len(rule) != 4:
            raise ValueError(f"La regla {i} debe tener 4 elementos: [min, max, clase, etiqueta].")

def build_label_dict_from_rules(rules):
    return {int(rule[2]): str(rule[3]) for rule in rules}

def reclassify_array(arr, rules):
    """
    Reclasifica un array continuo según reglas [min_incl, max_excl, clase, etiqueta].
    Los nodata/NaN quedan como NaN.
    """
    out = np.full(arr.shape, np.nan, dtype="float64")

    for min_val, max_val, class_value, label in rules:
        mask_rule = np.isfinite(arr) & (arr >= min_val) & (arr < max_val)
        out[mask_rule] = class_value

    return out

def map_category_label(class_value, label_dict):
    if pd.isna(class_value):
        return "SIN_DATO"
    class_int = int(class_value)
    return label_dict.get(class_int, str(class_int))

def tabulate_categories_for_geom(src, geom, raster_mode, label_dict, reclass_rules=None, all_touched=False):
    """
    Devuelve DataFrame con:
    class_value, categoria, n_pixeles, superficie_ha
    """

    out_image, _ = mask(
        src,
        [geom],
        crop=True,
        all_touched=all_touched,
        filled=False
    )

    # out_image es MaskedArray
    arr = out_image[0].astype("float64")

    # convertir a NaN todo lo enmascarado (fuera del polígono)
    arr = np.where(np.ma.getmaskarray(arr), np.nan, arr)

    # si el raster trae nodata explícito, también convertirlo a nan
    if src.nodata is not None and not (isinstance(src.nodata, float) and math.isnan(src.nodata)):
        arr[arr == src.nodata] = np.nan

    if raster_mode == "reclasificar":
        arr = reclassify_array(arr, reclass_rules)

    valid = arr[np.isfinite(arr)]

    if valid.size == 0:
        return pd.DataFrame(columns=["class_value", "categoria", "n_pixeles", "superficie_ha"])

    unique_vals, counts = np.unique(valid.astype(int), return_counts=True)
    pixel_area_ha = get_pixel_area_ha(src)

    rows = []
    for val, cnt in zip(unique_vals, counts):
        rows.append({
            "class_value": int(val),
            "categoria": map_category_label(int(val), label_dict),
            "n_pixeles": int(cnt),
            "superficie_ha": round(cnt * pixel_area_ha, ROUND_DECIMALS)
        })

    return pd.DataFrame(rows)

def build_detail_stats(tabulated_df, municipios_area_df, municipio_field):
    if tabulated_df.empty:
        return pd.DataFrame(columns=[
            municipio_field, "class_value", "categoria", "n_pixeles",
            "superficie_ha_obs", "superficie_municipio_ha",
            "factor_ajuste_area", "superficie_ha_aj",
            "porcentaje", "orden_pct", "es_dominante"
        ])

    # --- base ---
    detail = tabulated_df.copy()
    detail = detail.rename(columns={"superficie_ha": "superficie_ha_obs"})

    # unir área municipal
    detail = detail.merge(
        municipios_area_df,
        on=municipio_field,
        how="left"
    )

    # validar que sí haya superficie municipal
    if detail["superficie_municipio_ha"].isna().any():
        faltantes = detail.loc[detail["superficie_municipio_ha"].isna(), municipio_field].drop_duplicates().tolist()
        raise ValueError(
            f"No se encontró superficie municipal para estos municipios: {faltantes}"
        )

    # --- total observado por municipio ---
    total_obs = (
        detail.groupby(municipio_field, as_index=False)["superficie_ha_obs"]
        .sum()
        .rename(columns={"superficie_ha_obs": "superficie_total_obs"})
    )

    detail = detail.merge(total_obs, on=municipio_field, how="left")

    # --- factor de ajuste ---
    detail["factor_ajuste_area"] = np.where(
        (detail["superficie_total_obs"] > 0) & (detail["superficie_municipio_ha"] > 0),
        detail["superficie_municipio_ha"] / detail["superficie_total_obs"],
        np.nan
    )

    # --- superficie ajustada ---
    detail["superficie_ha_aj"] = detail["superficie_ha_obs"] * detail["factor_ajuste_area"]

    # --- porcentaje final ---
    detail["porcentaje"] = np.where(
        detail["superficie_municipio_ha"] > 0,
        (detail["superficie_ha_aj"] / detail["superficie_municipio_ha"]) * 100.0,
        np.nan
    )

    # limpiar infinitos por seguridad
    cols_clean = ["factor_ajuste_area", "superficie_ha_aj", "porcentaje"]
    for c in cols_clean:
        detail[c] = detail[c].replace([np.inf, -np.inf], np.nan)

    # --- orden ---
    detail = detail.sort_values(
        by=[municipio_field, "porcentaje", "superficie_ha_aj"],
        ascending=[True, False, False]
    ).copy()

    rank_series = detail.groupby(municipio_field)["porcentaje"].rank(
        method="first",
        ascending=False
    )

    # usar entero anulable de pandas para tolerar NA
    detail["orden_pct"] = rank_series.astype("Int64")
    detail["es_dominante"] = (detail["orden_pct"] == 1).astype("Int64")

    # --- redondeo ---
    cols_round = [
        "superficie_ha_obs",
        "superficie_municipio_ha",
        "superficie_total_obs",
        "factor_ajuste_area",
        "superficie_ha_aj",
        "porcentaje"
    ]

    for c in cols_round:
        detail[c] = detail[c].round(2)

    return detail

def build_municipal_coverage(detail_df, municipios_area_df, municipio_field):
    if detail_df.empty:
        cov = municipios_area_df.copy()
        cov["superficie_total_obs"] = 0.0
        cov["pct_cobertura_tematica_obs"] = 0.0
        return cov

    # tomar un solo registro por municipio, porque superficie_total_obs
    # ya es el mismo valor repetido en todas las categorías del municipio
    cov = (
        detail_df[[municipio_field, "superficie_municipio_ha", "superficie_total_obs"]]
        .drop_duplicates(subset=[municipio_field])
        .copy()
    )

    cov["pct_cobertura_tematica_obs"] = (
        cov["superficie_total_obs"] / cov["superficie_municipio_ha"]
    ) * 100.0

    cov["superficie_total_obs"] = cov["superficie_total_obs"].round(2)
    cov["pct_cobertura_tematica_obs"] = cov["pct_cobertura_tematica_obs"].round(2)

    return cov

def build_summary_stats(detail_df, coverage_df, municipio_field):
    if detail_df.empty:
        return pd.DataFrame(columns=[
            municipio_field,
            "max_categoria",
            "max_categoria_ha",
            "max_categoria_pct",
            "superficie_municipio_ha",
            "superficie_total_obs",
            "pct_cobertura_tematica_obs"
        ])

    dominant = detail_df[detail_df["orden_pct"] == 1].copy()

    summary = dominant[[
        municipio_field,
        "categoria",
        "superficie_ha_aj",
        "porcentaje",
        "superficie_municipio_ha"
    ]].rename(columns={
        "categoria": "max_categoria",
        "superficie_ha_aj": "max_categoria_ha",
        "porcentaje": "max_categoria_pct"
    })

    summary = summary.merge(
        coverage_df[[municipio_field, "superficie_total_obs", "pct_cobertura_tematica_obs"]],
        on=municipio_field,
        how="left"
    )

    return summary

def export_reclassified_raster(src_path, out_path, rules):
    with rasterio.open(src_path) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata

        if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
            arr[arr == nodata] = np.nan

        recl = reclassify_array(arr, rules)

        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.int16,
            count=1,
            nodata=-9999,
            compress="lzw"
        )

        out_arr = np.where(np.isfinite(recl), recl, -9999).astype("int16")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_arr, 1)

# =========================================================
# PROCESO PRINCIPAL
# =========================================================

def main():
    ensure_dir(OUT_DIR)

    with rasterio.open(RASTER_PATH) as src:
        print("CRS raster:", src.crs)
        print("Nodata raster:", src.nodata)
        print("Resolución:", src.res)
    
    print("1. Validando raster...")
    if not raster_exists(RASTER_PATH):
        msg = f"Raster no encontrado: {RASTER_PATH}"
        if FAIL_IF_RASTER_MISSING:
            raise FileNotFoundError(msg)
        print(f"Aviso: {msg}")
        return

    if RASTER_MODE not in ["categorico", "reclasificar"]:
        raise ValueError("RASTER_MODE debe ser 'categorico' o 'reclasificar'.")

    if RASTER_MODE == "reclasificar":
        validate_reclass_rules(RECLASS_RULES)
        label_dict = build_label_dict_from_rules(RECLASS_RULES)
    else:
        label_dict = {int(k): str(v) for k, v in CATEGORY_LABELS.items()}

    print("2. Cargando municipios...")
    gdf_munis = read_vector(MUNICIPIOS_PATH, MUNICIPIOS_LAYER)
    print(f"   Municipios cargados: {len(gdf_munis)}")

    validate_fields(gdf_munis, [MUNICIPIO_FIELD], "capa municipios")

    print("3. Corrigiendo geometrías...")
    gdf_munis = fix_geometries(gdf_munis)
    gdf_munis = multipart_to_singlepart_safe(gdf_munis)
    print(f"   Municipios válidos: {len(gdf_munis)}")

    print("4. Calculando área municipal...")
    if gdf_munis.crs is None:
        raise ValueError("La capa de municipios no tiene CRS definido.")

    gdf_munis_area = gdf_munis.to_crs(WORK_CRS)
    municipios_area_df = build_municipios_area(gdf_munis_area, MUNICIPIO_FIELD)

    if EXPORT_RECLASSIFIED_RASTER and RASTER_MODE == "reclasificar":
        out_reclass = os.path.join(OUT_DIR, RECLASSIFIED_RASTER_NAME)
        print("5. Exportando raster reclasificado...")
        export_reclassified_raster(RASTER_PATH, out_reclass, RECLASS_RULES)
        print(f"   Raster reclasificado: {out_reclass}")

    print("6. Extrayendo estadísticas categóricas por municipio...")
    all_rows = []

    with rasterio.open(RASTER_PATH) as src:
        if src.crs is None:
            raise ValueError("El raster no tiene CRS definido.")

        gdf_munis_raster = gdf_munis.to_crs(src.crs)

        for idx, row in gdf_munis_raster.iterrows():
            municipio = row[MUNICIPIO_FIELD]
            geom = row.geometry

            try:
                df_tab = tabulate_categories_for_geom(
                    src=src,
                    geom=geom,
                    raster_mode=RASTER_MODE,
                    label_dict=label_dict,
                    reclass_rules=RECLASS_RULES if RASTER_MODE == "reclasificar" else None,
                    all_touched=ALL_TOUCHED
                )

                if df_tab.empty:
                    continue

                df_tab[MUNICIPIO_FIELD] = municipio
                all_rows.append(df_tab)

            except ValueError:
                # sin intersección
                continue

    if all_rows:
        tabulated = pd.concat(all_rows, ignore_index=True)
    else:
        tabulated = pd.DataFrame(columns=[MUNICIPIO_FIELD, "class_value", "categoria", "n_pixeles", "superficie_ha"])

    print("7. Generando tabla detalle...")
    detail = build_detail_stats(tabulated, municipios_area_df, MUNICIPIO_FIELD)

    print("8. Generando cobertura municipal...")
    coverage = build_municipal_coverage(detail, municipios_area_df, MUNICIPIO_FIELD)

    print("9. Generando resumen municipal...")
    summary = build_summary_stats(detail, coverage, MUNICIPIO_FIELD)

    # Orden final de columnas
    if not detail.empty:
        detail = detail[[
            MUNICIPIO_FIELD,
            "class_value",
            "categoria",
            "n_pixeles",
            "superficie_ha_obs",
            "superficie_municipio_ha",
            "superficie_total_obs",
            "factor_ajuste_area",
            "superficie_ha_aj",
            "porcentaje",
            "orden_pct",
            "es_dominante"
        ]].copy()

    if not summary.empty:
        summary = summary[[
            MUNICIPIO_FIELD,
            "max_categoria",
            "max_categoria_ha",
            "max_categoria_pct",
            "superficie_municipio_ha",
            "superficie_total_obs",
            "pct_cobertura_tematica_obs"
        ]].copy()

    coverage = coverage[[
        MUNICIPIO_FIELD,
        "superficie_municipio_ha",
        "superficie_total_obs",
        "pct_cobertura_tematica_obs"
    ]].copy()

    print("10. Exportando CSV...")
    out_detail_csv = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_estadistica_detalle.csv")
    out_summary_csv = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_estadistica_resumen.csv")
    out_coverage_csv = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_cobertura_municipal.csv")

    detail.to_csv(out_detail_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(out_summary_csv, index=False, encoding="utf-8-sig")
    coverage.to_csv(out_coverage_csv, index=False, encoding="utf-8-sig")

    print("\nProceso finalizado correctamente.")
    print(f"Detalle CSV:   {out_detail_csv}")
    print(f"Resumen CSV:   {out_summary_csv}")
    print(f"Cobertura CSV: {out_coverage_csv}")

if __name__ == "__main__":
    main()