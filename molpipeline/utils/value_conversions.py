"""Module for utilities converting values."""

from typing import Optional, Sequence, TypeAlias, Union

FloatCountRange: TypeAlias = tuple[Optional[float], Optional[float]]

IntCountRange: TypeAlias = tuple[Optional[int], Optional[int]]

# IntOrIntCountRange for Typing of count ranges
# - a single int for an exact value match
# - a range given as a tuple with a lower and upper bound
#   - both limits are optional
IntOrIntCountRange: TypeAlias = Union[int, IntCountRange]


def count_value_to_tuple(count: IntOrIntCountRange) -> IntCountRange:
    """Convert a count value to a tuple.

    Parameters
    ----------
    count: Union[int, IntCountRange]
        Count value. Can be a single int or a tuple of two values.

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
