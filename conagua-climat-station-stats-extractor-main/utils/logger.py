import logging
import logging.config

from typing import Any

# Create the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the base level to DEBUG for all logs

# Formatter to specify the log output format
formatter = logging.Formatter("{asctime}:{levelname}:{message}", style="{")

# Create and configure the console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set level to INFO for console (so it will log INFO and above)
console_handler.setFormatter(formatter)

# Filter to only allow INFO logs to appear in the console
class InfoFilter(logging.Filter):
    def filter(self, record):
        # Only allow INFO logs to be displayed in the console
        return record.levelno == logging.INFO

console_handler.addFilter(InfoFilter())  # Add the filter to the console handler

# Create and configure the file handler
file_handler = logging.FileHandler("climstats.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)  # Set level to WARNING for file (to capture WARNING and above)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


class Logger:

    @staticmethod
    def info(message: Any) -> None:
        logger.info(message)

    @staticmethod
    def debug(message: Any) -> None:
        logger.debug(message)

    @staticmethod
    def warning(message: Any) -> None:
        logger.warning(message)

    @staticmethod
    def error(message: Any) -> None:
        logger.error(message)