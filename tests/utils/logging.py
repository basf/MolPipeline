"""Test utils for logging."""

from contextlib import contextmanager
from typing import Generator

from loguru import logger


@contextmanager
def capture_logs(
    level="INFO", format="{level}:{name}:{message}"
) -> Generator[list[str], None, None]:
    """Capture loguru-based logs.

    Custom context manager to test loguru-based logs. For details and usage examples,
    see https://loguru.readthedocs.io/en/latest/resources/migration.html#replacing-assertlogs-method-from-unittest-library

    Parameters
    ----------
    level : str, optional
        Log level, by default "INFO"
    format : str, optional
        Log format, by default "{level}:{name}:{message}"

    Yields
    -------
    list[str]
        List of log messages
    """
    output: list[str] = []
    handler_id = logger.add(output.append, level=level, format=format)
    yield output
    logger.remove(handler_id)
