"""Classes and functions for detecting and handling None values."""
from __future__ import annotations

from typing import Any, Iterable, Optional, TypeVar

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self
import numpy as np
import numpy.typing as npt

from molpipeline.abstract_pipeline_elements.core import (
    ABCPipelineElement,
    InvalidInstance,
    TransformingPipelineElement,
    RemovedInstance,
)
from molpipeline.utils.molpipeline_types import (
    AnyIterable,
    AnyType,
    AnyNumpyElement,
    Number,
)


_T = TypeVar("_T")


# pylint: disable=R0903
class NoneFilter(ABCPipelineElement):
    """Collects None values and can fill Dummy values to matrices where None values were removed."""

    element_ids: set[str]
    none_indices: list[int]
    n_total: int

    def __init__(
        self,
        element_ids: set[str],
        name: str = "NoneFilter",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize NoneCollector.

        Parameters
        ----------
        element_ids: list[str]
            List of Pipeline Elements for which None can be removed.

        Returns
        -------
        None
        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        self.none_indices = []
        if not isinstance(element_ids, set):
            element_ids = set(element_ids)
        self.element_ids = element_ids
        self.n_total = 0
        self._requires_fitting = True

    @classmethod
    def from_element_list(
        cls,
        element_list: Iterable[TransformingPipelineElement],
        name: str = "NoneFilter",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> Self:
        element_ids = set([element.uuid for element in element_list])
        return cls(element_ids, name=name, n_jobs=n_jobs, uuid=uuid)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this element."""
        params = super().get_params(deep=deep)
        if deep:
            params["element_ids"] = set(
                [str(element_id) for element_id in self.element_ids]
            )
        else:
            params["element_ids"] = self.element_ids
        return params

    def set_params(self, parameters: dict[str, Any]) -> Self:
        """Set parameters for this element."""
        param_copy = dict(parameters)
        element_ids = param_copy.pop("element_ids", None)
        if element_ids is not None:
            if not isinstance(element_ids, set):
                raise TypeError(f"Unexpected Type: {type(element_ids)}")
            self.element_ids = element_ids
        super().set_params(**param_copy)
        return self

    def check_removal(self, value: AnyType) -> bool:
        """Check if value should be removed.

        Parameters
        ----------
        value: AnyType
            Value to be checked.

        Returns
        -------
        bool
            True if value should be removed.
        """
        if isinstance(value, InvalidInstance) and value.element_id in self.element_ids:
            return True
        return False

    def fit(self, values: AnyIterable) -> Self:
        """Transform values and return a list without the None values."""
        return self

    def fit_transform(self, values: AnyIterable) -> AnyIterable:
        """Transform values and return a list without the None values."""
        self.fit(values)
        return self.transform(values)

    def co_transform(self, values: AnyIterable) -> AnyIterable:
        if self.n_total != len(values):
            raise ValueError("Length of values does not match length of values in fit")
        if isinstance(values, list):
            out_list = []
            for idx, value in enumerate(values):
                if idx not in self.none_indices:
                    out_list.append(value)
            return out_list
        if isinstance(values, np.ndarray):
            return np.delete(values, self.none_indices, axis=0)
        raise TypeError(f"Unexpected Type: {type(values)}")

    def transform(self, values: AnyIterable) -> AnyIterable:
        """Transform values and return a list without the None values."""
        self.n_total = len(values)
        self.none_indices = []
        for i, value in enumerate(values):
            if self.check_removal(value):
                self.none_indices.append(i)
        return self.co_transform(values)

    def transform_single(self, value: Any) -> Any:
        """Transform a single value.

        Parameters
        ----------
        value: Any
            Value to be transformed.

        Returns
        -------
        Any
            Transformed value.
        """
        if self.check_removal(value):
            return RemovedInstance(
                filter_element=self,  # type: ignore
            )
        return value


class _MultipleNoneFilter:
    def __init__(self, none_filter_list: list[NoneFilter]) -> None:
        """Initialize NoneCollector.

        Parameters
        ----------
        none_filter_list: list[NoneFilter]
            List of NoneFilter objects.

        Returns
        -------
        None
        """
        self.none_filter_list = none_filter_list
        prior_remover_dict = {}
        for i, remover_element in enumerate(none_filter_list):
            prior_remover_dict[remover_element] = none_filter_list[:i]
        self.prior_remover_dict = prior_remover_dict

    def transform(self, values: AnyIterable) -> AnyIterable:
        """Transform values and return a list without the None values."""
        for none_filter in self.none_filter_list:
            values = none_filter.transform(values)
        return values

    def co_transform(self, values: AnyIterable) -> AnyIterable:
        """Transform values and return a list without the None values."""
        for none_filter in self.none_filter_list:
            values = none_filter.co_transform(values)
        return values

    def fit_transform(self, values: AnyIterable) -> AnyIterable:
        """Transform values and return a list without the None values."""
        for none_filter in self.none_filter_list:
            values = none_filter.fit_transform(values)
        return values

    def register_removed(self, index: int, value: RemovedInstance) -> None:
        """Register an invalid instance.

        Parameters
        ----------
        index: int
            Index of the invalid instance.
        value: Any
            Value of the invalid instance.
        """
        if not isinstance(value, RemovedInstance):
            raise TypeError(f"Unexpected Type: {type(value)}")

        for none_filter in self.none_filter_list:
            if value.filter_element is none_filter:
                new_index = index
                for prior_remover in self.prior_remover_dict[none_filter]:
                    new_index -= len(prior_remover.none_indices)
                none_filter.none_indices.append(index)
                break
        else:
            raise ValueError(
                f"Invalid instance not captured by any NoneFilter: {value.filter_element}"
            )


class NoneFiller(ABCPipelineElement):
    """Fill None values with a Dummy value."""

    fill_value: Any
    none_filter_id: str
    _none_filter: Optional[NoneFilter]
    n_total: int

    def __init__(
        self,
        none_filter_id: str,
        fill_value: Any,
        name: str = "NoneFiller",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize NoneFiller."""
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        self.none_filter_id = none_filter_id
        self._none_filter = None
        self.fill_value = fill_value

    @classmethod
    def from_none_filter(
        cls,
        none_filter: NoneFilter,
        fill_value: Any,
        name: str = "NoneFiller",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> Self:
        """Initialize NoneFiller."""
        filler = cls(
            none_filter_id=none_filter.uuid,
            fill_value=fill_value,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )
        filler.none_filter = none_filter
        return filler

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this element."""
        params = super().get_params(deep=deep)
        if deep:
            params["none_filter_id"] = str(self.none_filter_id)
            params["fill_value"] = type(self.fill_value)(self.fill_value)
        else:
            params["none_filter_id"] = self.none_filter_id
            params["fill_value"] = self.fill_value
        return params

    @property
    def none_filter(self) -> NoneFilter:
        if self._none_filter is None:
            raise ValueError("NoneFilter not set")
        return self._none_filter

    @none_filter.setter
    def none_filter(self, none_filter: NoneFilter) -> None:
        self._none_filter = none_filter

    def select_none_filter(self, none_filter_list: list[NoneFilter]) -> None:
        """Select the NoneFilter from a list of NoneFilters."""
        for none_filter in none_filter_list:
            if none_filter.uuid == self.none_filter_id:
                self.none_filter = none_filter
                break
        else:
            raise ValueError(f"NoneFilter with id {self.none_filter_id} not found")

    def fit(self, values: AnyIterable) -> Self:
        """Transform values and return a list without the None values."""
        return self

    def fit_transform(self, values: AnyIterable) -> AnyIterable:
        self.fit(values)
        return self.transform(values)

    def transform_single(self, value: Any) -> Any:
        """Transform a single value.

        Parameters
        ----------
        value: Any
            Value to be transformed.

        Returns
        -------
        Any
            Transformed value.
        """
        if isinstance(value, RemovedInstance) and value.filter_element == self:
            return self.fill_value
        return value

    def transform(self, values: AnyIterable) -> AnyIterable:
        """Transform values and return a list without the None values."""
        if len(values) != self.none_filter.n_total - len(self.none_filter.none_indices):
            print(self.none_filter.none_indices)
            raise ValueError(
                f"Length of values does not match length of values in fit. "
                f"Expected: {self.none_filter.n_total -len(self.none_filter.none_indices)}  - Received :{len(values)}"
            )
        return self.fill_with_dummy(values)

    def _fill_list(self, list_to_fill: list[Number]) -> list[Number]:
        filled_list = []
        next_value_pos = 0
        for index in range(len(list_to_fill) + len(self.none_filter.none_indices)):
            if index in self.none_filter.none_indices:
                filled_list.append(self.fill_value)
            else:
                filled_list.append(list_to_fill[next_value_pos])
                next_value_pos += 1
        if len(list_to_fill) != next_value_pos:
            raise AssertionError()
        return filled_list

    def _fill_numpy_arr(
        self, value_array: npt.NDArray[AnyNumpyElement]
    ) -> npt.NDArray[AnyNumpyElement]:
        fill_value = self.fill_value
        if fill_value is None:
            fill_value = np.nan
        output_shape = list(value_array.shape)
        output_shape[0] += len(self.none_filter.none_indices)
        has_value_indices = np.ones(output_shape[0], dtype=bool)
        has_value_indices[self.none_filter.none_indices] = False

        output_matrix: npt.NDArray[Any]
        output_matrix = np.ones(output_shape, dtype=value_array.dtype) * fill_value
        output_matrix[has_value_indices, ...] = value_array
        return output_matrix

    def fill_with_dummy(
        self,
        value_container: AnyIterable,
    ) -> AnyIterable:
        """Insert dummy values at the positions in the value container.

        Parameters
        ----------
        value_container: AnyIterable
            Iterable to fill with dummy values.

        Returns
        -------
        AnyIterable
            Iterable where dummy values were inserted to replace molecules which could not be processed.
        """
        if isinstance(value_container, list):
            return self._fill_list(value_container)
        if isinstance(value_container, np.ndarray):
            return self._fill_numpy_arr(value_container)
        raise TypeError(f"Unexpected Type: {type(value_container)}")
