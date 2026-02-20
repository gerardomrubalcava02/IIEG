# Cokriging Project

Estructura propuesta para convertir `260126_cokriging.py` en un proyecto mantenible de GitHub.

## Estructura

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

## Buenas prácticas aplicadas

- **Configuración desacoplada**: rutas y parámetros viven en `config/default.yaml`.
- **Código modular**: separación en módulos (configuración, utilidades de interpolación y pipeline).
- **Punto de entrada claro**: `run.py` con argumentos CLI.
- **Logs y artefactos organizados**: `_logs`, `_diagnosticos`, `_hist` dentro de `outdir`.

## Requisitos

Usa el mismo entorno del script original (`numpy`, `pandas`, `geopandas`, `rasterio`, `gstools`, `matplotlib`, `pyyaml`).

## Ejecución

Desde la carpeta del proyecto:

```bash
python run.py --config config/default.yaml
```

## Siguientes mejoras recomendadas

1. Agregar `pyproject.toml` y lockfile.
2. Añadir pruebas unitarias para funciones puras (`standardize`, `fit_variogram_model`, `discover_numeric_fields`).
3. Incluir CI con `ruff` + `pytest`.
