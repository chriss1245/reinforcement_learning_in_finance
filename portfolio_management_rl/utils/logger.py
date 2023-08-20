"""
Utilities for logging.
"""

import logging
from pathlib import Path

from portfolio_management_rl.utils.contstants import PROJECT_DIR


def get_logger(name: str, level=logging.INFO):
    """
    Return a logger with the specified name and level.
    """
    path = Path(name)
    logger_name = (
        path.relative_to(PROJECT_DIR).with_suffix("").as_posix().replace("/", ".")
    )

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    formater = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formater)
    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    logger = get_logger(__file__)
    logger.info("This is a test")
