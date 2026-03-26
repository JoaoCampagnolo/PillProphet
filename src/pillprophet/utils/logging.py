"""Logging configuration."""

import logging


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the project logger."""
    logger = logging.getLogger("pillprophet")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    return logger
