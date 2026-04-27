"""Project-wide logging configuration.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("...")
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger for the given module name.

    Handlers are added only once, so calling get_logger multiple times
    for the same name is safe.

    Args:
        name: Logger name — use __name__ in each module.
        level: Logging level (default: INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(level)
    return logger
