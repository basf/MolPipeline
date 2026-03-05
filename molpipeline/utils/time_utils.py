"""Wibbly wobbly, timey wimey ... stuff."""

import pandas as pd


def split_intervals(
    start: pd.Timestamp,
    end: pd.Timestamp,
    n_intervals: int,
) -> list[pd.Timestamp]:
    """Calculate interval boundaries for splitting a time range into equal parts.

    Parameters
    ----------
    start : pd.Timestamp
        Start of the interval.
    end : pd.Timestamp
        End of the interval.
    n_intervals : int
        Number of sub-intervals to create.

    Returns
    -------
    list[pd.Timestamp]
        List of interval boundaries (excluding start and end).
        Thus, the returned list will have length `n_intervals - 1`.

    Raises
    ------
    ValueError
        If n_intervals is not a positive integer.
    ValueError
        If start is not before end.

    """
    if n_intervals <= 0:
        raise ValueError("n_intervals must be a positive integer.")
    if start >= end:
        raise ValueError("start must be before end.")

    time_delta = (end - start) / n_intervals
    return [start + i * time_delta for i in range(1, n_intervals)]
