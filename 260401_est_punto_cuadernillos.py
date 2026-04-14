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

# -------------------------
# CAPA TEMÁTICA DE PUNTOS
# -------------------------
THEMATIC_PATH = "/home/serviciosocial/iieg_2026/06_cuadernillos/Insumos/capas/espacios_pub/espacios_publicos_y_lugares_recreativos.gpkg"
THEMATIC_LAYER = None                  # None = primera capa
THEMATIC_CATEGORY_FIELD = "geografico"

# -------------------------
# CAPA DE MUNICIPIOS
# -------------------------
MUNICIPIOS_PATH = "/home/serviciosocial/iieg_2026/b_sig/limite_municipal.gpkg"
MUNICIPIOS_LAYER = None
MUNICIPIO_FIELD = "nombre"

# -------------------------
# CRS DE TRABAJO
# -------------------------
WORK_CRS = 6368

# -------------------------
# SALIDAS
# -------------------------
OUT_DIR = "/home/serviciosocial/iieg_2026/06_cuadernillos/Resultados/espacios_publicos"
OUTPUT_PREFIX = "espacios_publicos"

# Si True, conserva todos los campos originales del punto + municipio
KEEP_ALL_FIELDS = True

# Predicado espacial:
# "within" = el punto debe estar dentro del municipio
# "intersects" = incluye puntos en borde
SPATIAL_PREDICATE = "intersects"

# Si True, elimina puntos que no caen en ningún municipio
DROP_POINTS_WITHOUT_MUNICIPIO = True

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

def fix_point_geometries(gdf):
    """
    Conserva únicamente geometrías puntuales válidas:
    Point y MultiPoint.
    """
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: make_valid(geom) if geom is not None else None
    )

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
    gdf = gdf[gdf.is_valid].copy()

    return gdf.reset_index(drop=True)

def fix_polygon_geometries(gdf):
    """
    Conserva únicamente geometrías poligonales válidas:
    Polygon y MultiPolygon.
    """
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: make_valid(geom) if geom is not None else None
    )

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf[gdf.is_valid].copy()

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

def normalize_text_field(gdf, field_name):
    gdf = gdf.copy()
    gdf[field_name] = gdf[field_name].fillna("SIN_DATO").astype(str).str.strip()
    gdf.loc[gdf[field_name] == "", field_name] = "SIN_DATO"
    return gdf

def spatial_join_points_to_municipios(points_gdf, munis_gdf, municipio_field, predicate="within"):
    """
    Asigna cada punto al municipio correspondiente.
    """
    munis_base = munis_gdf[[municipio_field, "geometry"]].copy()

    joined = gpd.sjoin(
        points_gdf,
        munis_base,
        how="left",
        predicate=predicate
    )

    # limpia campo auxiliar si existe
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    return joined.reset_index(drop=True)

def build_detail_stats(joined_gdf, municipio_field, category_field):
    """
    Tabla detallada por municipio + categoría:
    - municipio
    - categoría
    - total_puntos
    - total_puntos_municipio
    - porcentaje
    - orden_pct
    - es_dominante
    """
    df = joined_gdf[[municipio_field, category_field]].copy()

    detail = (
        df.groupby([municipio_field, category_field], dropna=False, as_index=False)
          .size()
          .rename(columns={"size": "total_puntos"})
    )

    total_municipal = (
        df.groupby(municipio_field, dropna=False, as_index=False)
          .size()
          .rename(columns={"size": "total_puntos_municipio"})
    )

    detail = detail.merge(total_municipal, on=municipio_field, how="left")

    detail["porcentaje"] = (
        detail["total_puntos"] / detail["total_puntos_municipio"]
    ) * 100.0

    detail = detail.sort_values(
        by=[municipio_field, "porcentaje", "total_puntos", category_field],
        ascending=[True, False, False, True]
    ).copy()

    detail["orden_pct"] = (
        detail.groupby(municipio_field)["porcentaje"]
              .rank(method="first", ascending=False)
              .astype(int)
    )

    detail["es_dominante"] = (detail["orden_pct"] == 1).astype(int)
    detail["porcentaje"] = detail["porcentaje"].round(2)

    return detail

def build_municipal_coverage(joined_gdf, gdf_munis, municipio_field):
    """
    Cobertura municipal para puntos:
    - total_puntos_municipio
    - municipios sin puntos quedan en 0
    """
    municipios = gdf_munis[[municipio_field]].drop_duplicates().copy()

    cov = (
        joined_gdf.groupby(municipio_field, as_index=False)
                  .size()
                  .rename(columns={"size": "total_puntos_municipio"})
    )

    cov = municipios.merge(cov, on=municipio_field, how="left")
    cov["total_puntos_municipio"] = cov["total_puntos_municipio"].fillna(0).astype(int)

    return cov

def build_summary_stats(detail_df, coverage_df, municipio_field, category_field):
    """
    Tabla resumen por municipio:
    - municipio
    - max_categoria
    - max_categoria_conteo
    - max_categoria_pct
    - total_puntos_municipio
    """
    dominant = detail_df[detail_df["orden_pct"] == 1].copy()

    summary = dominant[[
        municipio_field,
        category_field,
        "total_puntos",
        "porcentaje"
    ]].rename(columns={
        category_field: f"max_{category_field}",
        "total_puntos": f"max_{category_field}_conteo",
        "porcentaje": f"max_{category_field}_pct"
    })

    summary = coverage_df.merge(summary, on=municipio_field, how="left")

    summary[f"max_{category_field}"] = summary[f"max_{category_field}"].fillna("SIN_DATO")
    summary[f"max_{category_field}_conteo"] = summary[f"max_{category_field}_conteo"].fillna(0).astype(int)
    summary[f"max_{category_field}_pct"] = summary[f"max_{category_field}_pct"].fillna(0).round(2)

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
    gdf_thematic = fix_point_geometries(gdf_thematic)
    gdf_munis = fix_polygon_geometries(gdf_munis)

    gdf_thematic = multipart_to_singlepart_safe(gdf_thematic)
    gdf_munis = multipart_to_singlepart_safe(gdf_munis)

    # filtros finales de seguridad
    gdf_thematic = gdf_thematic[gdf_thematic.geometry.geom_type.isin(["Point"])].copy()
    gdf_munis = gdf_munis[gdf_munis.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

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
    gdf_thematic = normalize_category_field(gdf_thematic, THEMATIC_CATEGORY_FIELD)
    gdf_munis = normalize_text_field(gdf_munis, MUNICIPIO_FIELD)

    print("5. Asignando puntos a municipios...")
    joined = spatial_join_points_to_municipios(
        gdf_thematic,
        gdf_munis,
        MUNICIPIO_FIELD,
        predicate=SPATIAL_PREDICATE
    )

    print(f"   Registros con unión espacial: {len(joined)}")

    puntos_sin_municipio = joined[MUNICIPIO_FIELD].isna().sum()
    print(f"   Puntos sin municipio asignado: {puntos_sin_municipio}")

    if DROP_POINTS_WITHOUT_MUNICIPIO:
        joined = joined[joined[MUNICIPIO_FIELD].notna()].copy()
        print(f"   Registros conservados con municipio: {len(joined)}")

    if joined.empty:
        raise ValueError("La unión espacial quedó vacía. Revisa CRS, extensión, geometrías o predicado espacial.")

    print("6. Generando estadísticas detalladas...")
    detail = build_detail_stats(
        joined,
        MUNICIPIO_FIELD,
        THEMATIC_CATEGORY_FIELD
    )

    print("7. Generando cobertura municipal...")
    coverage = build_municipal_coverage(
        joined,
        gdf_munis,
        MUNICIPIO_FIELD
    )

    print("8. Generando resumen municipal...")
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
        "total_puntos",
        "total_puntos_municipio",
        "porcentaje",
        "orden_pct",
        "es_dominante"
    ]].copy()

    summary = summary[[
        MUNICIPIO_FIELD,
        f"max_{THEMATIC_CATEGORY_FIELD}",
        f"max_{THEMATIC_CATEGORY_FIELD}_conteo",
        f"max_{THEMATIC_CATEGORY_FIELD}_pct",
        "total_puntos_municipio"
    ]].copy()

    coverage = coverage[[
        MUNICIPIO_FIELD,
        "total_puntos_municipio"
    ]].copy()

    if not KEEP_ALL_FIELDS:
        joined = joined[[MUNICIPIO_FIELD, THEMATIC_CATEGORY_FIELD, "geometry"]].copy()

    # =====================================================
    # EXPORTACIÓN
    # =====================================================

    out_gpkg = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_cartografia_union.gpkg")
    out_detail_csv = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_estadistica_detalle.csv")
    out_summary_csv = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_estadistica_resumen.csv")
    out_coverage_csv = os.path.join(OUT_DIR, f"{OUTPUT_PREFIX}_cobertura_municipal.csv")

    print("9. Exportando base cartográfica...")
    joined.to_file(out_gpkg, layer="cartografia_union", driver="GPKG")

    print("10. Exportando tablas estadísticas...")
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