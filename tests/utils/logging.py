"""Test utils for logging."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import loguru
from loguru import logger


@contextmanager
def capture_logs(
    level: str = "INFO", log_format: str = "{level}:{name}:{message}"
) -> Generator[list[loguru.Message], None, None]:
    """Capture loguru-based logs.

    Custom context manager to test loguru-based logs. For details and usage examples,
    see https://loguru.readthedocs.io/en/latest/resources/migration.html#replacing-assertlogs-method-from-unittest-library

    Parameters
    ----------
    level : str, optional
        Log level, by default "INFO"
    log_format : str, optional
        Log format, by default "{level}:{name}:{message}"

    Yields
    -------
    list[loguru.Message]
        List of log messages

    Returns
    -------
    Generator[list[loguru.Message], None, None]
        List of log messages
    """
    output: list[loguru.Message] = []
    handler_id = logger.add(output.append, level=level, format=log_format)
    yield output
    logger.remove(handler_id)
