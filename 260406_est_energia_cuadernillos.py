#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid
from shapely.geometry import (
    Point, MultiPoint,
    LineString, MultiLineString,
    Polygon, MultiPolygon, GeometryCollection
)

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

ENERGIA_GPKG = "/home/serviciosocial/iieg_2026/06_cuadernillos/Insumos/capas/energia/energíav1.gpkg"

MUNICIPIOS_PATH = "/home/serviciosocial/iieg_2026/b_sig/limite_municipal.gpkg"
MUNICIPIOS_LAYER = None
MUNICIPIO_FIELD = "nombre"

WORK_CRS = 6368

BASE_OUT_DIR = "/home/serviciosocial/iieg_2026/06_cuadernillos/Resultados/energia"

KEEP_ALL_FIELDS = True

# Para puntos en borde municipal:
POINT_PREDICATE = "within"   # cambiar a "intersects" si detectas omisiones en límites

# =========================================================
# CONFIGURACIÓN DE TEMAS
# =========================================================
# metric_type:
#   - "count"  -> conteo de entidades
#   - "length" -> suma de longitudes
#   - "area"   -> suma de superficies
#
# geom_type:
#   - "point"
#   - "line"
#   - "polygon"
#
# category_field:
#   - None si solo quieres agregado municipal total
#
# filters:
#   - lista de tuplas: (campo, operador, valor)
#   - operadores soportados: ==, !=, in, not in

TOPICS = [
    {
        "name": "subestacion",
        "layer": "subestacion",
        "geom_type": "point",
        "metric_type": "count",
        "category_field": None,
        "filters": [],
        "output_prefix": "subestacion",
        "out_dir": os.path.join(BASE_OUT_DIR, "subestacion"),
    },
    {
        "name": "denue_energia",
        "layer": "denue_energia",
        "geom_type": "point",
        "metric_type": "count",
        "category_field": "categoria",
        "filters": [],
        "output_prefix": "denue_energia",
        "out_dir": os.path.join(BASE_OUT_DIR, "denue_energia"),
    },
    {
        "name": "centrales_electricas",
        "layer": "centrales_electricas",
        "geom_type": "point",
        "metric_type": "count",
        "category_field": "tecno_simp",
        "filters": [
            ("fase", "==", "En operación"),
        ],
        "output_prefix": "centrales_electricas",
        "out_dir": os.path.join(BASE_OUT_DIR, "centrales_electricas"),
    },
    {
        "name": "linea_transm_l",
        "layer": "linea_transm_l",
        "geom_type": "line",
        "metric_type": "length",
        "category_field": None,
        "filters": [],
        "output_prefix": "linea_transm_l",
        "out_dir": os.path.join(BASE_OUT_DIR, "linea_transm_l"),
    },
    {
        "name": "conducto_l",
        "layer": "conducto_l",
        "geom_type": "line",
        "metric_type": "length",
        "category_field": None,
        "filters": [],
        "output_prefix": "conducto_l",
        "out_dir": os.path.join(BASE_OUT_DIR, "conducto_l"),
    },
    {
        "name": "ductos_sistrangas",
        "layer": "ductos_sistrangas",
        "geom_type": "line",
        "metric_type": "length",
        "category_field": None,
        "filters": [],
        "output_prefix": "ductos_sistrangas",
        "out_dir": os.path.join(BASE_OUT_DIR, "ductos_sistrangas"),
    },
    {
        "name": "ductos_no_sistrangas",
        "layer": "ductos_no_sistrangas",
        "geom_type": "line",
        "metric_type": "length",
        "category_field": None,
        "filters": [],
        "output_prefix": "ductos_no_sistrangas",
        "out_dir": os.path.join(BASE_OUT_DIR, "ductos_no_sistrangas"),
    },
    {
        "name": "predios_centrales_electricas",
        "layer": "predios_centrales_electricas",
        "geom_type": "polygon",
        "metric_type": "area",
        "category_field": "tecno_simp",
        "filters": [
            ("fase", "==", "En operación"),
        ],
        "output_prefix": "predios_centrales_electricas",
        "out_dir": os.path.join(BASE_OUT_DIR, "predios_centrales_electricas"),
    },
]

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
    missing = [f for f in fields if f is not None and f not in gdf.columns]
    if missing:
        raise ValueError(f"En {gdf_name} faltan los campos: {missing}")

def normalize_text_field(gdf, field_name):
    gdf = gdf.copy()
    gdf[field_name] = gdf[field_name].fillna("SIN_DATO").astype(str).str.strip()
    gdf.loc[gdf[field_name] == "", field_name] = "SIN_DATO"
    return gdf

def normalize_category_field(gdf, category_field):
    if category_field is None:
        return gdf.copy()
    gdf = gdf.copy()
    gdf[category_field] = gdf[category_field].fillna("SIN_DATO").astype(str).str.strip()
    gdf.loc[gdf[category_field] == "", category_field] = "SIN_DATO"
    return gdf

def multipart_to_singlepart_safe(gdf):
    gdf = gdf.copy()
    try:
        gdf = gdf.explode(index_parts=False)
    except TypeError:
        gdf = gdf.explode()
    return gdf.reset_index(drop=True)

def extract_geometry_by_type(geom, geom_type):
    if geom is None or geom.is_empty:
        return None

    if geom_type == "point":
        valid_types = ["Point", "MultiPoint"]
    elif geom_type == "line":
        valid_types = ["LineString", "MultiLineString"]
    elif geom_type == "polygon":
        valid_types = ["Polygon", "MultiPolygon"]
    else:
        raise ValueError(f"Tipo geométrico no soportado: {geom_type}")

    if geom.geom_type in valid_types:
        return geom

    if geom.geom_type == "GeometryCollection":
        parts = []
        for part in geom.geoms:
            if part.geom_type in valid_types:
                if part.geom_type.startswith("Multi"):
                    parts.extend(list(part.geoms))
                else:
                    parts.append(part)

        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]

        if geom_type == "point":
            return MultiPoint(parts)
        elif geom_type == "line":
            return MultiLineString(parts)
        elif geom_type == "polygon":
            return MultiPolygon(parts)

    return None

def fix_geometries_by_type(gdf, geom_type):
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: make_valid(geom) if geom is not None else None
    )

    gdf["geometry"] = gdf["geometry"].apply(lambda geom: extract_geometry_by_type(geom, geom_type))

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.is_valid].copy()

    return gdf.reset_index(drop=True)

def apply_filters(gdf, filters):
    gdf = gdf.copy()
    for field, op, value in filters:
        if field not in gdf.columns:
            raise ValueError(f"El campo de filtro '{field}' no existe en la capa.")
        if op == "==":
            gdf = gdf[gdf[field] == value].copy()
        elif op == "!=":
            gdf = gdf[gdf[field] != value].copy()
        elif op == "in":
            gdf = gdf[gdf[field].isin(value)].copy()
        elif op == "not in":
            gdf = gdf[~gdf[field].isin(value)].copy()
        else:
            raise ValueError(f"Operador no soportado: {op}")
    return gdf.reset_index(drop=True)

def build_municipios_area(gdf_munis, municipio_field):
    mun = gdf_munis[[municipio_field, "geometry"]].copy()
    mun["superficie_municipio_ha"] = mun.geometry.area / 10000.0
    mun["superficie_municipio_ha"] = mun["superficie_municipio_ha"].round(2)
    return mun[[municipio_field, "superficie_municipio_ha"]]

# =========================================================
# PUNTOS
# =========================================================

def spatial_join_points_to_municipios(points_gdf, munis_gdf, municipio_field, predicate="within"):
    munis_base = munis_gdf[[municipio_field, "geometry"]].copy()
    joined = gpd.sjoin(points_gdf, munis_base, how="left", predicate=predicate)

    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    return joined.reset_index(drop=True)

def build_count_detail(joined_gdf, municipio_field, category_field=None):
    if category_field:
        detail = (
            joined_gdf.groupby([municipio_field, category_field], dropna=False, as_index=False)
            .size()
            .rename(columns={"size": "conteo"})
        )

        tot = (
            joined_gdf.groupby(municipio_field, as_index=False)
            .size()
            .rename(columns={"size": "conteo_total_municipio"})
        )

        detail = detail.merge(tot, on=municipio_field, how="left")
        detail["porcentaje"] = (detail["conteo"] / detail["conteo_total_municipio"]) * 100.0
        detail["porcentaje"] = detail["porcentaje"].round(2)

        detail = detail.sort_values(
            by=[municipio_field, "porcentaje", "conteo", category_field],
            ascending=[True, False, False, True]
        ).copy()

        detail["orden_pct"] = (
            detail.groupby(municipio_field)["porcentaje"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        detail["es_dominante"] = (detail["orden_pct"] == 1).astype(int)
        return detail

    detail = (
        joined_gdf.groupby(municipio_field, as_index=False)
        .size()
        .rename(columns={"size": "conteo_total_municipio"})
    )
    return detail

def build_count_coverage(joined_gdf, gdf_munis, municipio_field):
    municipios = gdf_munis[[municipio_field]].drop_duplicates().copy()
    cov = (
        joined_gdf.groupby(municipio_field, as_index=False)
        .size()
        .rename(columns={"size": "conteo_total_municipio"})
    )
    cov = municipios.merge(cov, on=municipio_field, how="left")
    cov["conteo_total_municipio"] = cov["conteo_total_municipio"].fillna(0).astype(int)
    return cov

def build_count_summary(detail_df, coverage_df, municipio_field, category_field=None):
    if category_field is None:
        return coverage_df.copy()

    dominant = detail_df[detail_df["orden_pct"] == 1].copy()

    summary = dominant[[
        municipio_field,
        category_field,
        "conteo",
        "porcentaje"
    ]].rename(columns={
        category_field: f"max_{category_field}",
        "conteo": f"max_{category_field}_conteo",
        "porcentaje": f"max_{category_field}_pct"
    })

    summary = coverage_df.merge(summary, on=municipio_field, how="left")
    summary[f"max_{category_field}"] = summary[f"max_{category_field}"].fillna("SIN_DATO")
    summary[f"max_{category_field}_conteo"] = summary[f"max_{category_field}_conteo"].fillna(0).astype(int)
    summary[f"max_{category_field}_pct"] = summary[f"max_{category_field}_pct"].fillna(0).round(2)

    return summary

# =========================================================
# LÍNEAS / POLÍGONOS
# =========================================================

def overlay_intersection(munis, thematic):
    inter = gpd.overlay(munis, thematic, how="intersection", keep_geom_type=False)
    inter = inter[inter.geometry.notnull()].copy()
    inter = inter[~inter.geometry.is_empty].copy()
    return inter.reset_index(drop=True)

def add_metric_fields(gdf, metric_type):
    gdf = gdf.copy()

    if metric_type == "length":
        gdf["metric_m"] = gdf.geometry.length
        gdf["metric_km"] = gdf["metric_m"] / 1000.0
    elif metric_type == "area":
        gdf["metric_m2"] = gdf.geometry.area
        gdf["metric_ha"] = gdf["metric_m2"] / 10000.0
    else:
        raise ValueError(f"Tipo de métrica no soportado en add_metric_fields: {metric_type}")

    return gdf

def build_metric_detail(intersection_gdf, municipios_area_df, municipio_field, category_field, metric_type):
    if metric_type == "length":
        raw_metric = "metric_km"
        out_metric = "longitud_km"
    elif metric_type == "area":
        raw_metric = "metric_ha"
        out_metric = "superficie_ha"
    else:
        raise ValueError(f"Tipo de métrica no soportado: {metric_type}")

    if category_field:
        detail = (
            intersection_gdf.groupby([municipio_field, category_field], as_index=False)[raw_metric]
            .sum()
            .rename(columns={raw_metric: out_metric})
        )

        total = (
            intersection_gdf.groupby(municipio_field, as_index=False)[raw_metric]
            .sum()
            .rename(columns={raw_metric: f"{out_metric}_total_municipio"})
        )

        detail = detail.merge(total, on=municipio_field, how="left")
        detail = detail.merge(municipios_area_df, on=municipio_field, how="left")

        detail["porcentaje"] = (detail[out_metric] / detail[f"{out_metric}_total_municipio"]) * 100.0
        detail["porcentaje"] = detail["porcentaje"].round(2)
        detail[out_metric] = detail[out_metric].round(2)
        detail[f"{out_metric}_total_municipio"] = detail[f"{out_metric}_total_municipio"].round(2)

        detail = detail.sort_values(
            by=[municipio_field, "porcentaje", out_metric, category_field],
            ascending=[True, False, False, True]
        ).copy()

        detail["orden_pct"] = (
            detail.groupby(municipio_field)["porcentaje"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        detail["es_dominante"] = (detail["orden_pct"] == 1).astype(int)
        return detail

    detail = (
        intersection_gdf.groupby(municipio_field, as_index=False)[raw_metric]
        .sum()
        .rename(columns={raw_metric: f"{out_metric}_total_municipio"})
    )
    detail = detail.merge(municipios_area_df, on=municipio_field, how="left")
    detail[f"{out_metric}_total_municipio"] = detail[f"{out_metric}_total_municipio"].round(2)
    return detail

def build_metric_coverage(intersection_gdf, municipios_area_df, municipio_field, metric_type):
    if metric_type == "length":
        raw_metric = "metric_km"
        out_metric = "longitud_km_total_municipio"
    elif metric_type == "area":
        raw_metric = "metric_ha"
        out_metric = "superficie_ha_total_municipio"
    else:
        raise ValueError(f"Tipo de métrica no soportado: {metric_type}")

    cov = (
        intersection_gdf.groupby(municipio_field, as_index=False)[raw_metric]
        .sum()
        .rename(columns={raw_metric: out_metric})
    )

    cov = municipios_area_df.merge(cov, on=municipio_field, how="left")
    cov[out_metric] = cov[out_metric].fillna(0).round(2)

    if metric_type == "area":
        cov["pct_cobertura_tematica"] = (
            cov["superficie_ha_total_municipio"] / cov["superficie_municipio_ha"]
        ) * 100.0
        cov["pct_cobertura_tematica"] = cov["pct_cobertura_tematica"].round(2)

    return cov

def build_metric_summary(detail_df, coverage_df, municipio_field, category_field, metric_type):
    if metric_type == "length":
        metric_col = "longitud_km"
        total_col = "longitud_km_total_municipio"
    elif metric_type == "area":
        metric_col = "superficie_ha"
        total_col = "superficie_ha_total_municipio"
    else:
        raise ValueError(f"Tipo de métrica no soportado: {metric_type}")

    if category_field is None:
        return coverage_df.copy()

    dominant = detail_df[detail_df["orden_pct"] == 1].copy()

    summary = dominant[[
        municipio_field,
        category_field,
        metric_col,
        "porcentaje"
    ]].rename(columns={
        category_field: f"max_{category_field}",
        metric_col: f"max_{category_field}_{metric_col}",
        "porcentaje": f"max_{category_field}_pct"
    })

    cols_cov = [municipio_field, "superficie_municipio_ha", total_col]
    if "pct_cobertura_tematica" in coverage_df.columns:
        cols_cov.append("pct_cobertura_tematica")

    summary = summary.merge(coverage_df[cols_cov], on=municipio_field, how="left")
    return summary

# =========================================================
# PROCESAMIENTO POR TEMA
# =========================================================

def process_point_topic(gdf_thematic, gdf_munis, topic):
    category_field = topic["category_field"]
    municipio_field = MUNICIPIO_FIELD

    joined = spatial_join_points_to_municipios(
        gdf_thematic,
        gdf_munis,
        municipio_field,
        predicate=POINT_PREDICATE
    )

    joined = joined[joined[municipio_field].notna()].copy()

    if joined.empty:
        raise ValueError("La unión espacial de puntos quedó vacía.")

    detail = build_count_detail(joined, municipio_field, category_field)
    coverage = build_count_coverage(joined, gdf_munis, municipio_field)
    summary = build_count_summary(detail, coverage, municipio_field, category_field)

    if not KEEP_ALL_FIELDS:
        cols = [municipio_field, "geometry"]
        if category_field:
            cols.insert(1, category_field)
        joined = joined[cols].copy()

    return joined, detail, summary, coverage

def process_line_or_polygon_topic(gdf_thematic, gdf_munis, topic):
    municipio_field = MUNICIPIO_FIELD
    category_field = topic["category_field"]
    metric_type = topic["metric_type"]

    inter = overlay_intersection(gdf_munis, gdf_thematic)
    if inter.empty:
        raise ValueError("La intersección espacial quedó vacía.")

    inter = add_metric_fields(inter, metric_type)

    municipios_area = build_municipios_area(gdf_munis, municipio_field)
    detail = build_metric_detail(inter, municipios_area, municipio_field, category_field, metric_type)
    coverage = build_metric_coverage(inter, municipios_area, municipio_field, metric_type)
    summary = build_metric_summary(detail, coverage, municipio_field, category_field, metric_type)

    if not KEEP_ALL_FIELDS:
        cols = [municipio_field, "geometry"]
        if category_field:
            cols.insert(1, category_field)

        if metric_type == "length":
            cols = cols[:-1] + ["metric_m", "metric_km", "geometry"]
        elif metric_type == "area":
            cols = cols[:-1] + ["metric_m2", "metric_ha", "geometry"]

        inter = inter[cols].copy()

    return inter, detail, summary, coverage

def export_results(carto_gdf, detail_df, summary_df, coverage_df, topic):
    out_dir = topic["out_dir"]
    output_prefix = topic["output_prefix"]

    ensure_dir(out_dir)

    out_gpkg = os.path.join(out_dir, f"{output_prefix}_cartografia_union.gpkg")
    out_detail_csv = os.path.join(out_dir, f"{output_prefix}_estadistica_detalle.csv")
    out_summary_csv = os.path.join(out_dir, f"{output_prefix}_estadistica_resumen.csv")
    out_coverage_csv = os.path.join(out_dir, f"{output_prefix}_cobertura_municipal.csv")

    carto_gdf.to_file(out_gpkg, layer="cartografia_union", driver="GPKG")
    detail_df.to_csv(out_detail_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_summary_csv, index=False, encoding="utf-8-sig")
    coverage_df.to_csv(out_coverage_csv, index=False, encoding="utf-8-sig")

    print(f"   Base cartográfica: {out_gpkg}")
    print(f"   Detalle CSV:       {out_detail_csv}")
    print(f"   Resumen CSV:       {out_summary_csv}")
    print(f"   Cobertura CSV:     {out_coverage_csv}")

# =========================================================
# MAIN
# =========================================================

def main():
    print("1. Cargando municipios...")
    gdf_munis = read_vector(MUNICIPIOS_PATH, MUNICIPIOS_LAYER)
    validate_fields(gdf_munis, [MUNICIPIO_FIELD], "capa municipios")

    gdf_munis = fix_geometries_by_type(gdf_munis, "polygon")
    gdf_munis = multipart_to_singlepart_safe(gdf_munis)
    gdf_munis = fix_geometries_by_type(gdf_munis, "polygon")

    if gdf_munis.crs is None:
        raise ValueError("La capa de municipios no tiene CRS definido.")

    gdf_munis = gdf_munis.to_crs(WORK_CRS)
    gdf_munis = normalize_text_field(gdf_munis, MUNICIPIO_FIELD)

    print(f"   Municipios válidos: {len(gdf_munis)}")
    print(f"   CRS de trabajo: EPSG:{WORK_CRS}")

    print("\n2. Procesando temas de energía...")

    for topic in TOPICS:
        print("\n" + "=" * 70)
        print(f"Procesando: {topic['name']}")
        print("=" * 70)

        gdf_thematic = read_vector(ENERGIA_GPKG, topic["layer"])
        print(f"   Registros originales: {len(gdf_thematic)}")

        required_fields = []
        if topic["category_field"] is not None:
            required_fields.append(topic["category_field"])
        for field, _, _ in topic["filters"]:
            required_fields.append(field)

        validate_fields(gdf_thematic, required_fields, f"capa temática {topic['name']}")

        gdf_thematic = apply_filters(gdf_thematic, topic["filters"])
        print(f"   Registros tras filtros: {len(gdf_thematic)}")

        if gdf_thematic.empty:
            print("   Sin registros después del filtrado. Se omite.")
            continue

        gdf_thematic = fix_geometries_by_type(gdf_thematic, topic["geom_type"])
        gdf_thematic = multipart_to_singlepart_safe(gdf_thematic)
        gdf_thematic = fix_geometries_by_type(gdf_thematic, topic["geom_type"])

        if gdf_thematic.crs is None:
            raise ValueError(f"La capa '{topic['name']}' no tiene CRS definido.")

        gdf_thematic = gdf_thematic.to_crs(WORK_CRS)
        gdf_thematic = normalize_category_field(gdf_thematic, topic["category_field"])

        print(f"   Registros válidos: {len(gdf_thematic)}")

        if gdf_thematic.empty:
            print("   Sin geometrías válidas. Se omite.")
            continue

        if topic["geom_type"] == "point":
            carto, detail, summary, coverage = process_point_topic(gdf_thematic, gdf_munis, topic)

        elif topic["geom_type"] in ["line", "polygon"]:
            carto, detail, summary, coverage = process_line_or_polygon_topic(gdf_thematic, gdf_munis, topic)

        else:
            raise ValueError(f"Tipo geométrico no soportado: {topic['geom_type']}")

        export_results(carto, detail, summary, coverage, topic)

    print("\nProceso finalizado correctamente.")

if __name__ == "__main__":
    main()