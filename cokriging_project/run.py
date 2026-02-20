#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.cokriging.config import load_config
from src.cokriging.logging_utils import configure_logger
from src.cokriging.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de cokriging para climatología.")
    parser.add_argument("--config", default="config/default.yaml", help="Ruta al archivo YAML de configuración.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger, _, plotdir, histdir = configure_logger(cfg.paths.outdir)
    logger.info("Inicio interpolación con configuración: %s", args.config)
    run_pipeline(cfg, logger, plotdir, histdir)


if __name__ == "__main__":
    main()
