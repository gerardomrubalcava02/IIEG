# Cokriging Project

Proyecto para generar cartografía climática continua a resolución geoespacial homogénea a partir de observaciones puntuales de estaciones climatológicas, usando **Regression-Kriging** con covariables fisiográficas (por ejemplo, elevación y distancia a costa).

## Objetivo

Producir superficies raster climáticas (GeoTIFF) para variables de temperatura y precipitación, integrando:

- Variación espacial observada en estaciones.
- Tendencia explicada por covariables (DEM y distancia a costa).
- Estructura espacial residual modelada con geoestadística (variograma + kriging ordinario).

## Alcance del código

El pipeline está orientado a:

- Filtrar estaciones por condición operativa, año e índice climático.
- Interpolar campos mensuales y agregados (por ejemplo `PROM` o `ACUM`).
- Generar salidas raster con malla regular y CRS del insumo.
- Aplicar post-procesamiento físico básico:
  - Precipitación: `log1p`/`expm1`, no-negatividad y límite superior robusto.
  - Temperatura: recorte suave al rango observado ± buffer.

## Requisitos

## Software

- Python 3.10+ (recomendado).
- GDAL/PROJ compatibles con tu instalación de `geopandas` y `rasterio`.

## Librerías de Python

- `numpy`
- `pandas`
- `geopandas`
- `rasterio`
- `gstools`
- `matplotlib`
- `pyyaml`

## Datos necesarios

Para ejecutar correctamente, necesitas:

1. **GeoPackage de estaciones** con geometría puntual y campos de atributos climáticos.
2. **GeoPackage de costa** (con clases de costa oeste/este o equivalente).
3. **Raster DEM** en el sistema de referencia de trabajo.
4. Definir una **malla objetivo** (xmin, xmax, ymin, ymax, cellsize).

> Importante: estaciones, costa y DEM deben poder re-proyectarse a un CRS común para que las distancias sean consistentes.

## Estructura del proyecto

```text
cokriging_project/
├── config/
│   └── default.yaml
├── src/
│   └── cokriging/
│       ├── __init__.py
│       ├── config.py
│       ├── interpolation.py
│       ├── logging_utils.py
│       └── pipeline.py
└── run.py
```

## Configuración

El archivo `config/default.yaml` controla rutas y parámetros:

- `paths`: entradas/salida del proceso.
- `domain`: extensión y resolución de la malla raster.
- `filters`: reglas de filtrado (situación, año, índice y columnas fuente).
- `coast`: columna de clase costera y etiquetas para oeste/este.
- `model`: parámetros de interpolación y post-proceso.
- `months`: lista de campos mensuales esperados.

Antes de ejecutar, ajusta este archivo con tus rutas reales y nombres de columnas.

## Ejecución

Desde la carpeta `cokriging_project`:

```bash
python run.py --config config/default.yaml
```

## Salidas

En `outdir` (definido en YAML) se generan:

- GeoTIFF interpolados por variable/campo.
- `_logs/`: bitácora del proceso.
- `_diagnosticos/`: espacio reservado para diagnósticos.
- `_hist/`: histogramas de observaciones usadas.

## Recomendaciones de uso

- Verificar CRS y unidades de distancia antes de correr interpolación.
- Revisar cobertura espacial y número mínimo de estaciones válidas.
- Ajustar `cellsize`, `min_points`, `maxdist_frac` y límites de post-proceso según región/variable.
- Validar resultados con control visual y métricas externas (si se dispone de estaciones independientes).
