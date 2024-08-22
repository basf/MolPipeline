"""Module for utilities converting values."""

from typing import Optional, Sequence, Union


def count_value_to_tuple(
    count: Union[int, tuple[Optional[int], Optional[int]]]
) -> tuple[Optional[int], Optional[int]]:
    """Convert a count value to a tuple.

    Parameters
    ----------
    count: Union[int, float, tuple[Optional[int], Optional[int]]]
        Count value. Can be a single float or int or a tuple of two values.

    Returns
    -------
    tuple[Optional[int], Optional[int]]
        Tuple of count values.
    """
    if isinstance(count, int):
        return count, count
    if isinstance(count, Sequence):
        count_tuple = tuple(count)
        if len(count_tuple) != 2:
            raise ValueError(f"Expected a sequence of length 2, got: {count_tuple}")
        return count_tuple
    raise TypeError(f"Got unexpected type: {type(count)}")
