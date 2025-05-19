"""Module for utilities converting values."""

from collections.abc import Sequence

from molpipeline.utils.molpipeline_types import IntCountRange, IntOrIntCountRange


def count_value_to_tuple(count: IntOrIntCountRange) -> IntCountRange:
    """Convert a count value to a tuple.

    Parameters
    ----------
    count: IntOrIntCountRange
        Count value. Can be a single int or a tuple of two values.

    Raises
    ------
    ValueError
        If the count is a sequence of length other than 2.
    TypeError
        If the count is not an int or a sequence.

    Returns
    -------
    IntCountRange
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
