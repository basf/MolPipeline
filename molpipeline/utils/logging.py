"""Logging helper functions."""

from __future__ import annotations

import timeit
from contextlib import contextmanager
from typing import Generator

from loguru import logger


def _message_with_time(source: str, message: str, time: float) -> str:
    """Create one line message for logging purposes.

    Adapted from sklearn's function to stay consistent with the logging style:
    https://github.com/scikit-learn/scikit-learn/blob/e16a6ddebd527e886fc22105710ee20ce255f9f0/sklearn/utils/_user_interface.py

    Parameters
    ----------
    source : str
        String indicating the source or the reference of the message.
    message : str
        Short message.
    time : float
        Time in seconds.

    Returns
    -------
    str
        Message with elapsed time.
    """
    start_message = f"[{source}] "

    # adapted from joblib.logger.short_format_time without the Windows -.1s
    # adjustment
    if time > 60:
        time_str = f"{(time / 60):4.1f}min"
    else:
        time_str = f" {time:5.1f}s"

    end_message = f" {message}, total={time_str}"
    dots_len = 70 - len(start_message) - len(end_message)
    return f"{start_message}{dots_len * '.'}{end_message}"


@contextmanager
def print_elapsed_time(
    source: str, message: str | None = None, use_logger: bool = False
) -> Generator[None, None, None]:
    """Log elapsed time to stdout when the context is exited.

    Adapted from sklearn's function to stay consistent with the logging style:
    https://github.com/scikit-learn/scikit-learn/blob/e16a6ddebd527e886fc22105710ee20ce255f9f0/sklearn/utils/_user_interface.py

    Parameters
    ----------
    source : str
        String indicating the source or the reference of the message.
    message : str, default=None
        Short message. If None, nothing will be printed.
    use_logger : bool, default=False
        If True, the message will be logged using the logger.

    Returns
    -------
    context_manager
        Prints elapsed time upon exit if verbose.
    """
    if message is None:
        yield
    else:
        start = timeit.default_timer()
        yield
        message_to_print = _message_with_time(
            source, message, timeit.default_timer() - start
        )

        if use_logger:
            logger.info(message_to_print)
        else:
            print(message_to_print)
