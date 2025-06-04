"""Module for utilities converting values."""

from collections.abc import Sequence
from typing import TypeVar

VarNumber = TypeVar("VarNumber", float | None, int | None)


def assure_range(value: VarNumber | Sequence[VarNumber]) -> tuple[VarNumber, VarNumber]:
    """Assure that the value is defining a range.

    Integers or floats are converted to a range with the same value for both

    Parameters
    ----------
    value: VarNumber | Sequence[VarNumber]
        Count value. Can be a single int | float or a Sequence of two values.

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
    if isinstance(value, (float, int)):
        return value, value
    if isinstance(value, Sequence):
        range_tuple = tuple(value)
        if len(range_tuple) != 2:  # noqa: PLR2004
            raise ValueError(f"Expected a sequence of length 2, got: {range_tuple}")
        return range_tuple
    raise TypeError(f"Got unexpected type: {type(value)}")
