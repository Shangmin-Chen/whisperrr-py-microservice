"""Central logging configuration."""

import logging
import sys

from .config import settings


def configure_logging() -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level, logging.INFO))
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, settings.log_level, logging.INFO))
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for h in root_logger.handlers:
            h.setLevel(getattr(logging, settings.log_level, logging.INFO))


def get_uvicorn_log_level() -> str:
    return settings.log_level.lower()
