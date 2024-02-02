"""Module for checking values."""

from __future__ import annotations

from typing import Any

__all__ = ["is_empty"]


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
    """
    if hasattr(value, "shape"):
        if value.shape[0] == 0:
            return True
        return False
    if hasattr(value, "__len__"):
        if len(value) == 0:
            return True
        return False
    return False
