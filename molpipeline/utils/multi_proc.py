"""Utility functions for multiprocessing."""

import multiprocessing
import warnings


def check_available_cores(n_requested_cores: int) -> int:
    """ Compare number of requested cores with available cores and return a (corrected) number.

    Parameters
    ----------
    n_requested_cores: int
        Number of requested cores.

    Returns
    -------
    int
        Number of used cores.
    """
    try:
        n_available_cores = multiprocessing.cpu_count()
    except ModuleNotFoundError:
        warnings.warn("Cannot import multiprocessing library. Falling back to single core!")
        return 1

    if n_requested_cores > n_available_cores:
        warnings.warn("Requested more cores than available. Using maximum number of cores!")
        return n_available_cores
    elif n_requested_cores < 0:
        return n_available_cores
    else:
        return n_requested_cores
