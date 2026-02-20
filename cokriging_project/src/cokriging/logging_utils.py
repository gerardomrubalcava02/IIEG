from __future__ import annotations

import logging
from pathlib import Path


def configure_logger(outdir: str, filename: str = "interpolacion_rk.log") -> tuple[logging.Logger, Path, Path, Path]:
    out_path = Path(outdir)
    logdir = out_path / "_logs"
    plotdir = out_path / "_diagnosticos"
    histdir = out_path / "_hist"
    for directory in (out_path, logdir, plotdir, histdir):
        directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rk_kriging")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(logdir / filename, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger, logdir, plotdir, histdir
