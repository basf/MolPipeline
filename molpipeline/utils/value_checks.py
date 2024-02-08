"""Module for checking values."""

from __future__ import annotations

from typing import Any

__all__ = ["is_empty", "get_length"]


def is_empty(value: Any) -> bool:
    """Check if value is empty.

    Parameters
    ----------
    value: Any
        Value to be checked.

    Returns
    -------
    bool
        True if value is empty, False otherwise.

    Raises
    ------
    TypeError
        If value does not have a shape or len attribute.
    """
    if get_length(value) == 0:
        return True
    return False


def get_length(values: Any) -> int:
    """Get the length of the values as given by the shape or len attribute.

    Parameters
    ----------
    values: Any
        Values to be checked.

    Returns
    -------
    int
        Length of the values.
    """
    if hasattr(values, "shape"):
        return values.shape[0]
    if hasattr(values, "__len__"):
        return len(values)
    raise TypeError("Values must have a shape or len attribute.")
