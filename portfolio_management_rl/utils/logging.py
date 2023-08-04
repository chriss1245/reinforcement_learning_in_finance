"""
Utilities for logging.
"""

import logging

def get_logger(name: str, level=logging.INFO):
    """
    Return a logger with the specified name and level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formater)

    logger.addHandler(handler)

    return logger
