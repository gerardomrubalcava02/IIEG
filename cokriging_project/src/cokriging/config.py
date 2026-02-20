from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    stations_gpkg: str
    stations_layer: str
    coast_gpkg: str
    coast_layer: str | None
    dem_raster: str
    outdir: str


@dataclass(frozen=True)
class DomainConfig:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    cellsize: float


@dataclass(frozen=True)
class FilterConfig:
    base_situation: str
    base_year: int
    temp_index: str
    rain_index: str
    col_situation: str
    col_year: str
    col_index: str


@dataclass(frozen=True)
class CoastConfig:
    class_column: str
    west_tags: set[str]
    east_tags: set[str]


@dataclass(frozen=True)
class ModelConfig:
    min_points: int
    bin_no: int
    cv_splits: int
    cv_seed: int
    maxdist_frac: float
    temp_clip_buffer: float
    rain_max_factor: float


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    domain: DomainConfig
    filters: FilterConfig
    coast: CoastConfig
    model: ModelConfig
    months: tuple[str, ...]


def _as_set(value: list[str] | set[str]) -> set[str]:
    return {str(v).strip().upper() for v in value}


def load_config(path: str | Path) -> AppConfig:
    cfg_path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    return AppConfig(
        paths=PathsConfig(**raw["paths"]),
        domain=DomainConfig(**raw["domain"]),
        filters=FilterConfig(**raw["filters"]),
        coast=CoastConfig(
            class_column=raw["coast"]["class_column"],
            west_tags=_as_set(raw["coast"]["west_tags"]),
            east_tags=_as_set(raw["coast"]["east_tags"]),
        ),
        model=ModelConfig(**raw["model"]),
        months=tuple(raw["months"]),
    )
