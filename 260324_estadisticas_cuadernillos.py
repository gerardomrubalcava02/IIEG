#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# CONFIGURACIÓN PARAMETRIZABLE
# =========================================================

# Capa temática
THEMATIC_PATH = "/home/serviciosocial/iieg_2026/06_cuadernillos/Insumos/capas/espacios_pub/espacios_publicos_y_lugares_recreativos.gpkg"
THEMATIC_LAYER = None                 # None = primera capa
THEMATIC_CATEGORY_FIELD = "geografico" # campo de categoría a analizar

# Capa de municipios
MUNICIPIOS_PATH = "/home/serviciosocial/iieg_2026/b_sig/limite_municipal.gpkg"
MUNICIPIOS_LAYER = None
MUNICIPIO_FIELD = "nombre"

# CRS de trabajo para cálculo de áreas
WORK_CRS = 6368

# Carpeta de salida
OUT_DIR = "/home/serviciosocial/iieg_2026/06_cuadernillos/Resultados/espacios_publicos"

# Prefijo de archivos de salida
OUTPUT_PREFIX = "espacios_publicos"

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

from shapely.geometry import GeometryCollection, MultiPolygon
from shapely.ops import unary_union

def extract_polygonal_part(geom):
    """
    Extrae solo la parte poligonal de una geometría.
    Si la geometría es Polygon o MultiPolygon, la devuelve igual.
    Si es GeometryCollection, extrae polígonos/multipolígonos.
    Si no tiene parte poligonal, devuelve None.
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

def normalize_category_field(gdf, category_field):
    gdf = gdf.copy()
    gdf[category_field] = gdf[category_field].fillna("SIN_DATO").astype(str).str.strip()
    gdf.loc[gdf[category_field] == "", category_field] = "SIN_DATO"
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

def build_detail_stats(intersection_gdf, municipios_area_df, municipio_field, category_field):
    """
    Tabla detallada por municipio + categoría:
    - municipio
    - categoría
    - superficie_ha
    - superficie_municipio_ha
    - porcentaje (respecto al área total del municipio)
    - orden_pct
    - es_dominante
    """
    df = intersection_gdf[[municipio_field, category_field, "area_ha"]].copy()

    detail = (
        df.groupby([municipio_field, category_field], dropna=False, as_index=False)["area_ha"]
        .sum()
        .rename(columns={"area_ha": "superficie_ha"})
    )

    detail = detail.merge(municipios_area_df, on=municipio_field, how="left")

    detail["porcentaje"] = (
        detail["superficie_ha"] / detail["superficie_municipio_ha"]
    ) * 100.0

    detail = detail.sort_values(
        by=[municipio_field, "porcentaje", "superficie_ha", category_field],
        ascending=[True, False, False, True]
    ).copy()

    detail["orden_pct"] = (
        detail.groupby(municipio_field)["porcentaje"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    detail["es_dominante"] = (detail["orden_pct"] == 1).astype(int)

    detail["superficie_ha"] = detail["superficie_ha"].round(2)
    detail["porcentaje"] = detail["porcentaje"].round(2)

    return detail

def build_municipal_coverage(intersection_gdf, municipios_area_df, municipio_field):
    """
    Cobertura temática total por municipio:
    - superficie_tematica_total_ha
    - pct_cobertura_tematica
    """
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

    return cov

def build_summary_stats(detail_df, coverage_df, municipio_field, category_field):
    """
    Tabla resumen por municipio:
    - municipio
    - max_categoria
    - max_categoria_ha
    - max_categoria_pct
    - superficie_municipio_ha
    - superficie_tematica_total_ha
    - pct_cobertura_tematica
    """
    dominant = detail_df[detail_df["orden_pct"] == 1].copy()

    summary = dominant[[
        municipio_field,
        category_field,
        "superficie_ha",
        "porcentaje",
        "superficie_municipio_ha"
    ]].rename(columns={
        category_field: f"max_{category_field}",
        "superficie_ha": f"max_{category_field}_ha",
        "porcentaje": f"max_{category_field}_pct"
    })

    summary = summary.merge(
        coverage_df[[municipio_field, "superficie_tematica_total_ha", "pct_cobertura_tematica"]],
        on=municipio_field,
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

    validate_fields(gdf_thematic, [THEMATIC_CATEGORY_FIELD], "capa temática")
    validate_fields(gdf_munis, [MUNICIPIO_FIELD], "capa municipios")

    print("2. Revisando y corrigiendo geometrías...")
    gdf_thematic = fix_geometries(gdf_thematic)
    gdf_munis = fix_geometries(gdf_munis)

    gdf_thematic = multipart_to_singlepart_safe(gdf_thematic)
    gdf_munis = multipart_to_singlepart_safe(gdf_munis)

    # filtro de seguridad posterior al explode
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

    print("4. Normalizando campo de categoría...")
    gdf_thematic = normalize_category_field(gdf_thematic, THEMATIC_CATEGORY_FIELD)

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

    print("8. Generando estadísticas detalladas...")
    detail = build_detail_stats(
        inter,
        municipios_area,
        MUNICIPIO_FIELD,
        THEMATIC_CATEGORY_FIELD
    )

    print("9. Generando cobertura temática municipal...")
    coverage = build_municipal_coverage(
        inter,
        municipios_area,
        MUNICIPIO_FIELD
    )

    print("10. Generando resumen municipal...")
    summary = build_summary_stats(
        detail,
        coverage,
        MUNICIPIO_FIELD,
        THEMATIC_CATEGORY_FIELD
    )

    # Orden final de columnas
    detail = detail[[
        MUNICIPIO_FIELD,
        THEMATIC_CATEGORY_FIELD,
        "superficie_ha",
        "superficie_municipio_ha",
        "porcentaje",
        "orden_pct",
        "es_dominante"
    ]].copy()

    summary = summary[[
        MUNICIPIO_FIELD,
        f"max_{THEMATIC_CATEGORY_FIELD}",
        f"max_{THEMATIC_CATEGORY_FIELD}_ha",
        f"max_{THEMATIC_CATEGORY_FIELD}_pct",
        "superficie_municipio_ha",
        "superficie_tematica_total_ha",
        "pct_cobertura_tematica"
    ]].copy()

    coverage = coverage[[
        MUNICIPIO_FIELD,
        "superficie_municipio_ha",
        "superficie_tematica_total_ha",
        "pct_cobertura_tematica"
    ]].copy()

    if not KEEP_ALL_FIELDS:
        inter = inter[[MUNICIPIO_FIELD, THEMATIC_CATEGORY_FIELD, "area_m2", "area_ha", "geometry"]].copy()

    # =====================================================
    # EXPORTACIÓN
    # =====================================================

    out_gpkg = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_cartografia_union.gpkg")
    out_detail_csv = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_estadistica_detalle.csv")
    out_summary_csv = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_estadistica_resumen.csv")
    out_coverage_csv = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_cobertura_municipal.csv")

    print("11. Exportando base cartográfica...")
    inter.to_file(out_gpkg, layer="cartografia_union", driver="GPKG")

    print("12. Exportando tablas estadísticas...")
    detail.to_csv(out_detail_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(out_summary_csv, index=False, encoding="utf-8-sig")
    coverage.to_csv(out_coverage_csv, index=False, encoding="utf-8-sig")

    print("\nProceso finalizado correctamente.")
    print(f"Base cartográfica:        {out_gpkg}")
    print(f"Detalle estadístico CSV:  {out_detail_csv}")
    print(f"Resumen estadístico CSV:  {out_summary_csv}")
    print(f"Cobertura municipal CSV:  {out_coverage_csv}")

if __name__ == "__main__":
    main()