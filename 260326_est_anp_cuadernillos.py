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
# CONFIGURACIÓN
# =========================================================

# -------------------------
# Municipios
# -------------------------
MUNICIPIOS_PATH = "/home/serviciosocial/iieg_2026/b_sig/limite_municipal.gpkg"
MUNICIPIOS_LAYER = None
MUNICIPIO_FIELD = "nombre"

# -------------------------
# ANP
# -------------------------
ANP_PATH = "/home/serviciosocial/iieg_2026/06_cuadernillos/Insumos/capas/anp/anps_2026_v1.gpkg"
ANP_LAYER = None

ANP_JUR_FIELD = "jur"
ANP_TIP_FIELD = "tip"
ANP_NOM_FIELD = "nom"

# -------------------------
# Humedales
# Si CATEGORY_FIELD = None, todo quedará como "Humedales"
# Si TEXT_FIELD = None, la cadena se construye con la categoría
# -------------------------
HUMEDALES_PATH = "/home/serviciosocial/iieg_2026/06_cuadernillos/Insumos/capas/anp/humedalesv1.gpkg"
HUMEDALES_LAYER = None
HUMEDALES_CATEGORY_FIELD = None     # ejemplo: "tipo"
HUMEDALES_TEXT_FIELD = None         # ejemplo: "nom"
HUMEDALES_FIXED_LABEL = "Humedales"

# -------------------------
# Manglares
# Si CATEGORY_FIELD = None, todo quedará como "Manglares"
# Si TEXT_FIELD = None, la cadena se construye con la categoría
# -------------------------
MANGLARES_PATH = "/home/serviciosocial/iieg_2026/06_cuadernillos/Insumos/capas/anp/manglaresv1.gpkg"
MANGLARES_LAYER = None
MANGLARES_CATEGORY_FIELD = None     # ejemplo: "tipo"
MANGLARES_TEXT_FIELD = None         # ejemplo: "nom"
MANGLARES_FIXED_LABEL = "Manglares"

# -------------------------
# CRS de trabajo
# -------------------------
WORK_CRS = 6368

# -------------------------
# Salida
# -------------------------
OUT_DIR = "/home/serviciosocial/iieg_2026/06_cuadernillos/Resultados/anp_humedales_manglares"
OUTPUT_PREFIX = "anp_humedales_manglares"

# Si True, conserva más campos en la cartografía de salida
KEEP_ALL_FIELDS = False

# Umbral para eliminar residuos muy pequeños
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
    missing = [f for f in fields if f and f not in gdf.columns]
    if missing:
        raise ValueError(f"En {gdf_name} faltan los campos: {missing}")

def clean_text(value, fallback="SIN_DATO"):
    if pd.isna(value):
        return fallback
    value = str(value).strip()
    return value if value else fallback

def clean_text_nullable(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value if value else None

def normalize_upper(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()

def extract_polygonal_part(geom):
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

def overlay_intersection(munis, thematic):
    inter = gpd.overlay(munis, thematic, how="intersection", keep_geom_type=False)
    inter = inter[inter.geometry.notnull()].copy()
    inter = inter[~inter.geometry.is_empty].copy()
    inter = keep_only_polygonal(inter)
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

def concat_unique_texts(series):
    vals = []
    seen = set()

    for v in series.dropna():
        v = str(v).strip()
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            vals.append(v)

    vals = sorted(vals, key=lambda x: x.lower())
    return ", ".join(vals)

# =========================================================
# PREPARACIÓN DE CAPAS TEMÁTICAS
# =========================================================

def prepare_layer(path, layer=None):
    gdf = read_vector(path, layer)
    gdf = fix_geometries(gdf)
    gdf = multipart_to_singlepart_safe(gdf)
    gdf = keep_only_polygonal(gdf)
    return gdf

def build_anp_intersection(gdf_munis, gdf_anp, municipio_field):
    validate_fields(
        gdf_anp,
        [ANP_JUR_FIELD, ANP_TIP_FIELD, ANP_NOM_FIELD],
        "capa ANP"
    )

    inter = overlay_intersection(gdf_munis, gdf_anp)
    if inter.empty:
        return inter

    inter = inter.copy()

    inter["_jur"] = inter[ANP_JUR_FIELD].apply(clean_text)
    inter["_tip"] = inter[ANP_TIP_FIELD].apply(clean_text)
    inter["_nom"] = inter[ANP_NOM_FIELD].apply(clean_text)
    inter["_jur_up"] = inter[ANP_JUR_FIELD].apply(normalize_upper)

    def make_tip_nom(row):
        tip = clean_text_nullable(row["_tip"])
        nom = clean_text_nullable(row["_nom"])
        if tip and nom:
            return f"{tip} {nom}"
        if nom:
            return nom
        if tip:
            return tip
        return "SIN_DATO"

    inter["_tip_nom"] = inter.apply(make_tip_nom, axis=1)

    def classify_anp(row):
        jur_up = row["_jur_up"]
        tip = row["_tip"]
        nom = row["_nom"]
        tip_nom = row["_tip_nom"]

        if jur_up in {"MUNICIPAL", "ESTATAL", "FEDERAL"}:
            categoria = f"ANP {row['_jur'].title()}"
            texto_item = tip_nom

        elif jur_up == "RAMSAR":
            categoria = "RAMSAR"
            texto_item = nom

        else:
            categoria = tip
            texto_item = nom

        return pd.Series([categoria, texto_item])

    inter[["categoria_final", "texto_item"]] = inter.apply(classify_anp, axis=1)
    inter["fuente"] = "ANP"

    cols = [municipio_field, "fuente", "categoria_final", "texto_item", "geometry"]

    if KEEP_ALL_FIELDS:
        return inter.copy()
    return inter[cols].copy()

def build_generic_intersection(
    gdf_munis,
    gdf_theme,
    municipio_field,
    source_label,
    category_field=None,
    text_field=None,
    fixed_label=None
):
    if category_field:
        validate_fields(gdf_theme, [category_field], f"capa {source_label}")
    if text_field:
        validate_fields(gdf_theme, [text_field], f"capa {source_label}")

    inter = overlay_intersection(gdf_munis, gdf_theme)
    if inter.empty:
        return inter

    inter = inter.copy()

    if category_field:
        inter["_cat"] = inter[category_field].apply(clean_text)
        inter["categoria_final"] = inter["_cat"].apply(lambda x: f"{source_label} {x}")
    else:
        inter["categoria_final"] = fixed_label if fixed_label else source_label

    if text_field:
        inter["texto_item"] = inter[text_field].apply(clean_text)
    else:
        if category_field:
            inter["texto_item"] = inter["_cat"].apply(clean_text)
        else:
            inter["texto_item"] = inter["categoria_final"]

    inter["fuente"] = source_label

    cols = [municipio_field, "fuente", "categoria_final", "texto_item", "geometry"]

    if KEEP_ALL_FIELDS:
        return inter.copy()
    return inter[cols].copy()

# =========================================================
# ESTADÍSTICAS
# =========================================================

def build_detail_stats(intersection_gdf, municipios_area_df, municipio_field):
    """
    Tabla detallada por municipio + categoría:
    - municipio
    - categoria
    - superficie_ha
    - superficie_municipio_ha
    - porcentaje
    - cadena_texto
    - orden_pct
    - es_dominante
    """
    df = intersection_gdf[[municipio_field, "categoria_final", "texto_item", "area_ha"]].copy()

    area_grp = (
        df.groupby([municipio_field, "categoria_final"], dropna=False, as_index=False)["area_ha"]
        .sum()
        .rename(columns={
            "categoria_final": "categoria",
            "area_ha": "superficie_ha"
        })
    )

    txt_grp = (
        df.groupby([municipio_field, "categoria_final"], dropna=False)["texto_item"]
        .apply(concat_unique_texts)
        .reset_index()
        .rename(columns={
            "categoria_final": "categoria",
            "texto_item": "cadena_texto"
        })
    )

    detail = area_grp.merge(txt_grp, on=[municipio_field, "categoria"], how="left")
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

    detail["superficie_ha"] = detail["superficie_ha"].round(2)
    detail["porcentaje"] = detail["porcentaje"].round(2)

    return detail

def build_municipal_coverage_unique(intersection_gdf, municipios_area_df, municipio_field):
    """
    Cobertura total no duplicada por municipio:
    realiza unión geométrica de todas las capas temáticas dentro de cada municipio.
    """
    cov_geom = (
        intersection_gdf[[municipio_field, "geometry"]]
        .dissolve(by=municipio_field)
        .reset_index()
    )

    cov_geom["superficie_tematica_total_ha"] = cov_geom.geometry.area / 10000.0
    cov_geom["superficie_tematica_total_ha"] = cov_geom["superficie_tematica_total_ha"].round(2)

    cov = municipios_area_df.merge(
        cov_geom[[municipio_field, "superficie_tematica_total_ha"]],
        on=municipio_field,
        how="left"
    )

    cov["superficie_tematica_total_ha"] = cov["superficie_tematica_total_ha"].fillna(0)
    cov["pct_cobertura_tematica"] = (
        cov["superficie_tematica_total_ha"] / cov["superficie_municipio_ha"]
    ) * 100.0
    cov["pct_cobertura_tematica"] = cov["pct_cobertura_tematica"].round(2)

    return cov

def build_summary_stats(detail_df, coverage_df, municipio_field):
    """
    Tabla resumen por municipio:
    - municipio
    - categoria_dominante
    - categoria_dominante_ha
    - categoria_dominante_pct
    - cadena_texto_dominante
    - superficie_municipio_ha
    - superficie_tematica_total_ha
    - pct_cobertura_tematica
    """
    dominant = detail_df[detail_df["orden_pct"] == 1].copy()

    summary = dominant[[
        municipio_field,
        "categoria",
        "superficie_ha",
        "porcentaje",
        "cadena_texto",
        "superficie_municipio_ha"
    ]].rename(columns={
        "categoria": "categoria_dominante",
        "superficie_ha": "categoria_dominante_ha",
        "porcentaje": "categoria_dominante_pct",
        "cadena_texto": "cadena_texto_dominante"
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

    print("1. Cargando municipios...")
    gdf_munis = prepare_layer(MUNICIPIOS_PATH, MUNICIPIOS_LAYER)
    validate_fields(gdf_munis, [MUNICIPIO_FIELD], "capa municipios")

    print(f"   Municipios válidos: {len(gdf_munis)}")

    print("2. Revisando CRS...")
    if gdf_munis.crs is None:
        raise ValueError("La capa de municipios no tiene CRS definido.")

    gdf_munis = gdf_munis.to_crs(WORK_CRS)

    print(f"   CRS de trabajo: EPSG:{WORK_CRS}")

    # -----------------------------------
    # ANP
    # -----------------------------------
    print("3. Procesando ANP...")
    gdf_anp = prepare_layer(ANP_PATH, ANP_LAYER)

    if gdf_anp.crs is None:
        raise ValueError("La capa ANP no tiene CRS definido.")
    gdf_anp = gdf_anp.to_crs(WORK_CRS)

    inter_anp = build_anp_intersection(gdf_munis, gdf_anp, MUNICIPIO_FIELD)
    print(f"   Intersecciones ANP: {len(inter_anp)}")

    # -----------------------------------
    # Humedales
    # -----------------------------------
    print("4. Procesando humedales...")
    gdf_humedales = prepare_layer(HUMEDALES_PATH, HUMEDALES_LAYER)

    if gdf_humedales.crs is None:
        raise ValueError("La capa de humedales no tiene CRS definido.")
    gdf_humedales = gdf_humedales.to_crs(WORK_CRS)

    inter_humedales = build_generic_intersection(
        gdf_munis=gdf_munis,
        gdf_theme=gdf_humedales,
        municipio_field=MUNICIPIO_FIELD,
        source_label="Humedales",
        category_field=HUMEDALES_CATEGORY_FIELD,
        text_field=HUMEDALES_TEXT_FIELD,
        fixed_label=HUMEDALES_FIXED_LABEL
    )
    print(f"   Intersecciones humedales: {len(inter_humedales)}")

    # -----------------------------------
    # Manglares
    # -----------------------------------
    print("5. Procesando manglares...")
    gdf_manglares = prepare_layer(MANGLARES_PATH, MANGLARES_LAYER)

    if gdf_manglares.crs is None:
        raise ValueError("La capa de manglares no tiene CRS definido.")
    gdf_manglares = gdf_manglares.to_crs(WORK_CRS)

    inter_manglares = build_generic_intersection(
        gdf_munis=gdf_munis,
        gdf_theme=gdf_manglares,
        municipio_field=MUNICIPIO_FIELD,
        source_label="Manglares",
        category_field=MANGLARES_CATEGORY_FIELD,
        text_field=MANGLARES_TEXT_FIELD,
        fixed_label=MANGLARES_FIXED_LABEL
    )
    print(f"   Intersecciones manglares: {len(inter_manglares)}")

    # -----------------------------------
    # Unificación
    # -----------------------------------
    print("6. Integrando capas temáticas...")
    inter_list = [x for x in [inter_anp, inter_humedales, inter_manglares] if not x.empty]

    if not inter_list:
        raise ValueError("No hubo intersecciones válidas con ninguna capa temática.")

    inter = pd.concat(inter_list, ignore_index=True)
    inter = gpd.GeoDataFrame(inter, geometry="geometry", crs=f"EPSG:{WORK_CRS}")

    inter = add_area_fields(inter)
    inter = inter[inter["area_m2"] > MIN_AREA_M2].copy()

    if inter.empty:
        raise ValueError("Después del filtro de área mínima, no quedaron registros.")

    print(f"   Registros integrados: {len(inter)}")

    print("7. Calculando área total municipal...")
    municipios_area = build_municipios_area(gdf_munis, MUNICIPIO_FIELD)

    print("8. Generando estadísticas detalladas...")
    detail = build_detail_stats(
        inter,
        municipios_area,
        MUNICIPIO_FIELD
    )

    print("9. Generando cobertura municipal no duplicada...")
    coverage = build_municipal_coverage_unique(
        inter,
        municipios_area,
        MUNICIPIO_FIELD
    )

    print("10. Generando resumen municipal...")
    summary = build_summary_stats(
        detail,
        coverage,
        MUNICIPIO_FIELD
    )

    # Orden final de columnas
    detail = detail[[
        MUNICIPIO_FIELD,
        "categoria",
        "superficie_ha",
        "superficie_municipio_ha",
        "porcentaje",
        "cadena_texto",
        "orden_pct",
        "es_dominante"
    ]].copy()

    summary = summary[[
        MUNICIPIO_FIELD,
        "categoria_dominante",
        "categoria_dominante_ha",
        "categoria_dominante_pct",
        "cadena_texto_dominante",
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
        inter = inter[[
            MUNICIPIO_FIELD,
            "fuente",
            "categoria_final",
            "texto_item",
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