"""Wibbly wobbly, timey wimey ... stuff."""

from typing import Literal

import pandas as pd


def floor_timestamp(
    time_stamp: pd.Timestamp,
    round_to: Literal["day", "month", "hour"] | None,
) -> pd.Timestamp:
    """Round a timestamp down to the requested precision.

    Parameters
    ----------
    time_stamp : pd.Timestamp
        Timestamp to round.
    round_to : {"day", "month", "hour"} or None
        Target precision. If ``None``, the timestamp is returned unchanged.

    Returns
    -------
    pd.Timestamp
        Rounded timestamp.

    Raises
    ------
    ValueError
        If round_to is not one of the valid options.

    """
    if round_to is not None and round_to not in {"day", "month", "hour"}:
        raise ValueError(
            f"round_to must be 'day', 'month', 'hour', or None, got '{round_to}'",
        )

    if round_to == "day":
        return time_stamp.normalize()
    if round_to == "hour":
        return time_stamp.floor("h")
    if round_to == "month":
        return pd.Timestamp(
            year=time_stamp.year,
            month=time_stamp.month,
            day=1,
        )
    return time_stamp


def split_intervals(
    start: pd.Timestamp,
    end: pd.Timestamp,
    n_intervals: int,
    round_to: Literal["day", "month", "hour"] | None = None,
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
    round_to : {"day", "month", "hour"} or None, optional
        If set, rounds the interval boundaries to the specified precision.

    Returns
    -------
    list[pd.Timestamp]
        List of interval boundaries (excluding start and end).

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

    total_seconds = end - start
    time_delta = total_seconds / n_intervals
    return [
        floor_timestamp(start + i * time_delta, round_to) for i in range(1, n_intervals)
    ]
