"""Classes and functions for detecting and handling None values."""
from __future__ import annotations

from typing import Any, Union, overload
import numpy as np
import numpy.typing as npt


# pylint: disable=R0903
class NoneCollector:
    """Collects None values and can fill Dummy values to matrices where None values were removed."""

    fill_value: Any
    none_indices: list[int]

    def __init__(self, fill_value: Any = None) -> None:
        """Initialize NoneCollector."""
        self.fill_value = fill_value
        self.none_indices = []

    def _fill_list(self, list_to_fill: list[Any]) -> list[Any]:
        filled_list = []
        next_value_pos = 0
        for index in range(len(list_to_fill) + len(self.none_indices)):
            if index in self.none_indices:
                filled_list.append(self.fill_value)
            else:
                filled_list.append(list_to_fill[next_value_pos])
                next_value_pos += 1
        if len(list_to_fill) != next_value_pos:
            raise AssertionError()
        return filled_list

    def _fill_numpy_arr(self, value_array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        fill_value = self.fill_value
        if fill_value is None:
            fill_value = np.nan
        output_shape = list(value_array.shape)
        output_shape[0] += len(self.none_indices)
        has_value_indices = np.ones(output_shape[0], dtype=bool)
        has_value_indices[self.none_indices] = False

        output_matrix: npt.NDArray[Any]
        output_matrix = np.ones(output_shape, dtype=value_array.dtype) * fill_value
        output_matrix[has_value_indices, ...] = value_array
        return output_matrix

    def fill_with_dummy(
        self,
        value_container: Union[list[Any], npt.NDArray[Any]],
    ) -> Union[list[Any], npt.NDArray[Any]]:
        """Insert dummy values at the positions in the value container."""
        if isinstance(value_container, list):
            return self._fill_list(value_container)
        if isinstance(value_container, np.ndarray):
            return self._fill_numpy_arr(value_container)
        raise TypeError(f"Unexpected Type: {type(value_container)}")
