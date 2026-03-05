"""Wibbly wobbly, timey wimey ... stuff."""

import contextlib
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

NamedTimeStamps = Literal["today", "Q1", "Q2", "Q3", "Q4"]


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


def thresholds_for_n_years(
    final_threshold: pd.Timestamp | NamedTimeStamps,
    n_years: int,
    splits_per_year: int,
    round_to: str | None,
) -> list[pd.Timestamp]:
    """Generate a threshold list from a final threshold and a nof. years to go back.

    Parameters
    ----------
    final_threshold : pandas.Timestamp or {"today", "Q1", "Q2", "Q3", "Q4"}
        The upper bound for the generated thresholds.
    n_years : int
        Number of years to create the splits for.
    splits_per_year : int
        Number of splits per year. Must be at least 1.
    round_to : str | None
        Rounding precision for threshold timestamps.

    Returns
    -------
    list[pd.Timestamp]
        A list of constructed threshold timestamps.

    Raises
    ------
    ValueError
        If required parameters are missing or invalid.

    """
    if splits_per_year < 1:
        raise ValueError("splits_per_year must be at least 1.")

    resolved_final = resolve_named_time_stamps(final_threshold)
    threshold_list: list[pd.Timestamp] = []

    # We go backwards in time from the resolved final threshold over n_years
    for year_offset in range(n_years):
        end = resolved_final - pd.DateOffset(years=year_offset)
        start = end - pd.DateOffset(years=1)
        threshold_list.extend(split_intervals(start, end, splits_per_year))
        # Split intervals does not include start or end
        threshold_list.append(end)

    threshold_list = [pd.Timestamp(ts) for ts in threshold_list]

    if round_to == "normalize":
        threshold_list = [ts.normalize() for ts in threshold_list]
    elif round_to is not None:
        threshold_list = [ts.round(round_to) for ts in threshold_list]

    return sorted(threshold_list, reverse=True)


def timestamp_to_group(
    groups: npt.ArrayLike,
    threshold_list: list[pd.Timestamp] | pd.Series | pd.DatetimeIndex,
) -> npt.NDArray[np.int64]:
    """Convert time data to group indices.

    Parameters
    ----------
    groups : npt.ArrayLike
        Time data to convert to group indices. Should be datetime-like values.
    threshold_list : list[pd.Timestamp]
        List of timestamp thresholds that define the group boundaries.

    Returns
    -------
    npt.NDArray[np.int64]
        Group indices for each data point.

    """
    groups = np.asarray(groups, dtype=np.datetime64)

    split_index = np.zeros(len(groups), dtype=np.int64)
    for threshold in threshold_list:
        split_index[groups >= threshold] += 1
    return split_index


def resolve_named_time_stamps(
    time_stamp: NamedTimeStamps | pd.Timestamp,
) -> pd.Timestamp:
    """Convert special time strings to pd.Timestamp.

    Parameters
    ----------
    time_stamp : FinalThresholdStr | pd.Timestamp
        Can be a pd.Timestamp, "today", or one of "Q1", "Q2", "Q3", "Q4".

    Returns
    -------
    pd.Timestamp
        Resolved timestamp.

    Raises
    ------
    ValueError
        If the input string is not recognized.

    """
    if isinstance(time_stamp, str):
        with contextlib.suppress(ValueError):
            time_stamp = pd.Timestamp(time_stamp)
    if isinstance(time_stamp, pd.Timestamp):
        return time_stamp

    now = pd.Timestamp.now()
    quarter_start_map = {
        "Q1": pd.Timestamp(year=now.year, month=1, day=1),
        "Q2": pd.Timestamp(year=now.year, month=4, day=1),
        "Q3": pd.Timestamp(year=now.year, month=7, day=1),
        "Q4": pd.Timestamp(year=now.year, month=10, day=1),
    }
    mapped_threshold = quarter_start_map.get(time_stamp)
    if mapped_threshold is not None:
        return mapped_threshold

    raise ValueError(
        f"Unsupported final_threshold value: {time_stamp} "
        "Use a Timestamp, or one of 'Q1', 'Q2', 'Q3', 'Q4'.",
    )
