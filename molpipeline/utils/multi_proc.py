"""Utility functions for multiprocessing."""
from __future__ import annotations

import multiprocessing
import warnings


def check_available_cores(n_requested_cores: int) -> int:
    """Compare number of requested cores with available cores and return a (corrected) number.

    Parameters
    ----------
    n_requested_cores: int
        Number of requested cores.

    Returns
    -------
    int
        Number of used cores.
    """
    if not isinstance(n_requested_cores, int):
        raise TypeError(f"Not an integer: {n_requested_cores}")
    try:
        n_available_cores = multiprocessing.cpu_count()
    except ModuleNotFoundError:
        warnings.warn(
            "Cannot import multiprocessing library. Falling back to single core!"
        )
        return 1

    if n_requested_cores > n_available_cores:
        warnings.warn(
            "Requested more cores than available. Using maximum number of cores!"
        )
        return n_available_cores
    if n_requested_cores < 0:
        return n_available_cores

    return n_requested_cores


def calc_chunksize(n_jobs: int, len_iterable: int, factor: int = 4) -> int:
    """Calculate the chunksize for chunking an iterable of len_iterable length for processing with n_jobs.

    This function corresponds to the implementation in `multiprocessing.pool.Pool._map_async` and
    was inspired from: https://stackoverflow.com/a/54032744

    Parameters
    ----------
    n_jobs: int
        Number of jobs.
    len_iterable: int
        Length of iterable.
    factor: int
        Factor used by the heuristic to scale the number of workers.

    Returns
    -------
    int
        Chunksize.
    """
    chunksize, extra = divmod(len_iterable, n_jobs * factor)
    if extra:
        chunksize += 1
    return chunksize
