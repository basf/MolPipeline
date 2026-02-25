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
