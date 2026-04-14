#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid
from shapely.geometry import MultiPolygon

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# CONFIGURACIÓN PARAMETRIZABLE
# =========================================================

# Capa temática: cuencas
THEMATIC_PATH = "/home/serviciosocial/iieg_2026/06_cuadernillos/Insumos/capas/cuencas/cuencasv1.gpkg"
THEMATIC_LAYER = None

# Campos de categoría a analizar
THEMATIC_CATEGORY_FIELDS = ["clasificac", "categ"]

# Campo para texto concatenado de nombres de cuenca
CUENCA_NAME_FIELD = "cuenca"

# Capa de municipios
MUNICIPIOS_PATH = "/home/serviciosocial/iieg_2026/b_sig/limite_municipal.gpkg"
MUNICIPIOS_LAYER = None
MUNICIPIO_FIELD = "nombre"

# CRS de trabajo para cálculo de áreas
WORK_CRS = 6368

# Carpeta de salida
OUT_DIR = "/home/serviciosocial/iieg_2026/06_cuadernillos/Resultados/cuencas"

# Prefijo de archivos de salida
OUTPUT_PREFIX = "cuencas"

# Si True, conserva todos los campos de ambas capas en la salida cartográfica
KEEP_ALL_FIELDS = True

# Umbral para eliminar geometrías residuales muy pequeñas (en m²)
MIN_AREA_M2 = 0.0

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

def extract_polygonal_part(geom):
    """
    Extrae solo la parte poligonal de una geometría.
    """
    if geom is None or geom.is_empty:
        return None

    gtype = geom.geom_type

    if gtype in ["Polygon", "MultiPolygon"]:
        return geom

    if gtype == "GeometryCollection":
        polys = []
        for part in geom.geoms:
            if part.geom_type == "Polygon":
                polys.append(part)
            elif part.geom_type == "MultiPolygon":
                polys.extend(list(part.geoms))

        if not polys:
            return None
        if len(polys) == 1:
            return polys[0]
        return MultiPolygon(polys)

    return None

def fix_geometries(gdf):
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: make_valid(geom) if geom is not None else None
    )

    gdf["geometry"] = gdf["geometry"].apply(extract_polygonal_part)

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf[gdf.is_valid].copy()

    return gdf.reset_index(drop=True)

def keep_only_polygonal(gdf):
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    return gdf.reset_index(drop=True)

def multipart_to_singlepart_safe(gdf):
    gdf = gdf.copy()
    try:
        gdf = gdf.explode(index_parts=False)
    except TypeError:
        gdf = gdf.explode()
    return gdf.reset_index(drop=True)

def normalize_text_field(gdf, field):
    gdf = gdf.copy()
    gdf[field] = gdf[field].fillna("SIN_DATO").astype(str).str.strip()
    gdf.loc[gdf[field] == "", field] = "SIN_DATO"
    return gdf

def normalize_multiple_fields(gdf, fields):
    gdf = gdf.copy()
    for field in fields:
        gdf = normalize_text_field(gdf, field)
    return gdf

def overlay_intersection(munis, thematic):
    inter = gpd.overlay(munis, thematic, how="intersection", keep_geom_type=False)
    inter = inter[inter.geometry.notnull()].copy()
    inter = inter[~inter.geometry.is_empty].copy()
    return inter.reset_index(drop=True)

def add_area_fields(gdf):
    gdf = gdf.copy()
    gdf["area_m2"] = gdf.geometry.area
    gdf["area_ha"] = gdf["area_m2"] / 10000.0
    return gdf

def build_municipios_area(gdf_munis, municipio_field):
    mun = gdf_munis[[municipio_field, "geometry"]].copy()
    mun["superficie_municipio_ha"] = mun.geometry.area / 10000.0
    mun["superficie_municipio_ha"] = mun["superficie_municipio_ha"].round(2)
    return mun[[municipio_field, "superficie_municipio_ha"]]

def concat_unique_values(series):
    vals = (
        series.fillna("SIN_DATO")
        .astype(str)
        .str.strip()
    )
    vals = vals[vals != ""]
    vals = vals.drop_duplicates().tolist()
    return ", ".join(vals) if vals else "SIN_DATO"

def build_cuencas_text(intersection_gdf, municipio_field, cuenca_name_field):
    txt = (
        intersection_gdf.groupby(municipio_field)[cuenca_name_field]
        .apply(concat_unique_values)
        .reset_index(name="cuencas_txt")
    )
    return txt

def build_detail_stats(intersection_gdf, municipios_area_df, municipio_field, category_field):
    df = intersection_gdf[[municipio_field, category_field, "area_ha"]].copy()

    detail = (
        df.groupby([municipio_field, category_field], dropna=False, as_index=False)["area_ha"]
        .sum()
        .rename(columns={
            category_field: "categoria",
            "area_ha": "superficie_ha"
        })
    )

    detail = detail.merge(municipios_area_df, on=municipio_field, how="left")

    detail["porcentaje"] = (
        detail["superficie_ha"] / detail["superficie_municipio_ha"]
    ) * 100.0

    detail = detail.sort_values(
        by=[municipio_field, "porcentaje", "superficie_ha", "categoria"],
        ascending=[True, False, False, True]
    ).copy()

    detail["orden_pct"] = (
        detail.groupby(municipio_field)["porcentaje"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    detail["es_dominante"] = (detail["orden_pct"] == 1).astype(int)
    detail["campo_origen"] = category_field

    detail["superficie_ha"] = detail["superficie_ha"].round(2)
    detail["porcentaje"] = detail["porcentaje"].round(2)

    return detail

def build_municipal_coverage(intersection_gdf, municipios_area_df, municipio_field, campo_origen):
    cov = (
        intersection_gdf.groupby(municipio_field, as_index=False)["area_ha"]
        .sum()
        .rename(columns={"area_ha": "superficie_tematica_total_ha"})
    )

    cov = municipios_area_df.merge(cov, on=municipio_field, how="left")
    cov["superficie_tematica_total_ha"] = cov["superficie_tematica_total_ha"].fillna(0)

    cov["pct_cobertura_tematica"] = (
        cov["superficie_tematica_total_ha"] / cov["superficie_municipio_ha"]
    ) * 100.0

    cov["superficie_tematica_total_ha"] = cov["superficie_tematica_total_ha"].round(2)
    cov["pct_cobertura_tematica"] = cov["pct_cobertura_tematica"].round(2)
    cov["campo_origen"] = campo_origen

    return cov

def build_summary_stats(detail_df, coverage_df, municipio_field):
    dominant = detail_df[detail_df["orden_pct"] == 1].copy()

    summary = dominant[[
        municipio_field,
        "campo_origen",
        "categoria",
        "superficie_ha",
        "porcentaje",
        "superficie_municipio_ha"
    ]].rename(columns={
        "categoria": "max_categoria",
        "superficie_ha": "max_categoria_ha",
        "porcentaje": "max_categoria_pct"
    })

    summary = summary.merge(
        coverage_df[[municipio_field, "campo_origen", "superficie_tematica_total_ha", "pct_cobertura_tematica"]],
        on=[municipio_field, "campo_origen"],
        how="left"
    )

    return summary

# =========================================================
# PROCESO PRINCIPAL
# =========================================================

def main():
    ensure_dir(OUT_DIR)

    print("1. Cargando capas...")
    gdf_thematic = read_vector(THEMATIC_PATH, THEMATIC_LAYER)
    gdf_munis = read_vector(MUNICIPIOS_PATH, MUNICIPIOS_LAYER)

    print(f"   Capa temática: {len(gdf_thematic)} registros")
    print(f"   Capa municipios: {len(gdf_munis)} registros")

    required_thematic_fields = THEMATIC_CATEGORY_FIELDS + [CUENCA_NAME_FIELD]
    validate_fields(gdf_thematic, required_thematic_fields, "capa temática")
    validate_fields(gdf_munis, [MUNICIPIO_FIELD], "capa municipios")

    print("2. Revisando y corrigiendo geometrías...")
    gdf_thematic = fix_geometries(gdf_thematic)
    gdf_munis = fix_geometries(gdf_munis)

    gdf_thematic = multipart_to_singlepart_safe(gdf_thematic)
    gdf_munis = multipart_to_singlepart_safe(gdf_munis)

    gdf_thematic = keep_only_polygonal(gdf_thematic)
    gdf_munis = keep_only_polygonal(gdf_munis)

    print(f"   Temática válida: {len(gdf_thematic)} registros")
    print(f"   Municipios válidos: {len(gdf_munis)} registros")

    print("3. Homologando CRS...")
    if gdf_thematic.crs is None:
        raise ValueError("La capa temática no tiene CRS definido.")
    if gdf_munis.crs is None:
        raise ValueError("La capa de municipios no tiene CRS definido.")

    gdf_thematic = gdf_thematic.to_crs(WORK_CRS)
    gdf_munis = gdf_munis.to_crs(WORK_CRS)

    print(f"   CRS de trabajo: EPSG:{WORK_CRS}")

    print("4. Normalizando campos...")
    gdf_thematic = normalize_multiple_fields(gdf_thematic, required_thematic_fields)

    print("5. Realizando intersección espacial...")
    inter = overlay_intersection(gdf_munis, gdf_thematic)

    if inter.empty:
        raise ValueError("La intersección quedó vacía. Revisa CRS, extensión o geometrías.")

    print(f"   Registros intersectados: {len(inter)}")

    print("6. Calculando áreas...")
    inter = add_area_fields(inter)
    inter = inter[inter["area_m2"] > MIN_AREA_M2].copy()

    print("7. Calculando área total de municipios...")
    municipios_area = build_municipios_area(gdf_munis, MUNICIPIO_FIELD)

    print("8. Generando texto de cuencas por municipio...")
    cuencas_txt_df = build_cuencas_text(inter, MUNICIPIO_FIELD, CUENCA_NAME_FIELD)

    print("9. Generando estadísticas por cada campo categórico...")
    details = []
    summaries = []
    coverages = []

    for category_field in THEMATIC_CATEGORY_FIELDS:
        print(f"   -> Procesando campo: {category_field}")

        detail = build_detail_stats(
            inter,
            municipios_area,
            MUNICIPIO_FIELD,
            category_field
        )

        coverage = build_municipal_coverage(
            inter,
            municipios_area,
            MUNICIPIO_FIELD,
            category_field
        )

        summary = build_summary_stats(
            detail,
            coverage,
            MUNICIPIO_FIELD
        )

        detail = detail.merge(cuencas_txt_df, on=MUNICIPIO_FIELD, how="left")
        coverage = coverage.merge(cuencas_txt_df, on=MUNICIPIO_FIELD, how="left")
        summary = summary.merge(cuencas_txt_df, on=MUNICIPIO_FIELD, how="left")

        details.append(detail)
        coverages.append(coverage)
        summaries.append(summary)

    detail_all = pd.concat(details, ignore_index=True)
    coverage_all = pd.concat(coverages, ignore_index=True)
    summary_all = pd.concat(summaries, ignore_index=True)

    # Orden final de columnas
    detail_all = detail_all[[
        MUNICIPIO_FIELD,
        "campo_origen",
        "categoria",
        "superficie_ha",
        "superficie_municipio_ha",
        "porcentaje",
        "orden_pct",
        "es_dominante",
        "cuencas_txt"
    ]].copy()

    summary_all = summary_all[[
        MUNICIPIO_FIELD,
        "campo_origen",
        "max_categoria",
        "max_categoria_ha",
        "max_categoria_pct",
        "superficie_municipio_ha",
        "superficie_tematica_total_ha",
        "pct_cobertura_tematica",
        "cuencas_txt"
    ]].copy()

    coverage_all = coverage_all[[
        MUNICIPIO_FIELD,
        "campo_origen",
        "superficie_municipio_ha",
        "superficie_tematica_total_ha",
        "pct_cobertura_tematica",
        "cuencas_txt"
    ]].copy()

    if not KEEP_ALL_FIELDS:
        inter = inter[[
            MUNICIPIO_FIELD,
            *THEMATIC_CATEGORY_FIELDS,
            CUENCA_NAME_FIELD,
            "area_m2",
            "area_ha",
            "geometry"
        ]].copy()

    # =====================================================
    # EXPORTACIÓN
    # =====================================================

    out_gpkg = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_cartografia_union.gpkg")
    out_detail_csv = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_estadistica_detalle.csv")
    out_summary_csv = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_estadistica_resumen.csv")
    out_coverage_csv = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_cobertura_municipal.csv")

    print("10. Exportando base cartográfica...")
    inter.to_file(out_gpkg, layer="cartografia_union", driver="GPKG")

    print("11. Exportando tablas estadísticas...")
    detail_all.to_csv(out_detail_csv, index=False, encoding="utf-8-sig")
    summary_all.to_csv(out_summary_csv, index=False, encoding="utf-8-sig")
    coverage_all.to_csv(out_coverage_csv, index=False, encoding="utf-8-sig")

    print("\nProceso finalizado correctamente.")
    print(f"Base cartográfica:        {out_gpkg}")
    print(f"Detalle estadístico CSV:  {out_detail_csv}")
    print(f"Resumen estadístico CSV:  {out_summary_csv}")
    print(f"Cobertura municipal CSV:  {out_coverage_csv}")

if __name__ == "__main__":
    main()