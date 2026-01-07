"""Classes and functions for detecting and handling None values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from molpipeline.abstract_pipeline_elements.core import (
    ABCPipelineElement,
    InvalidInstance,
    RemovedInstance,
    TransformingPipelineElement,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from molpipeline.utils.molpipeline_types import AnyVarSeq, TypeFixedVarSeq

__all__ = ["ErrorFilter", "FilterReinserter", "_MultipleErrorFilter"]

_T = TypeVar("_T")
_S = TypeVar("_S")


class ErrorFilter(ABCPipelineElement):
    """Collects tracks and removes error values."""

    element_ids: set[str]
    error_indices: list[int]
    filter_everything: bool
    n_total: int

    def __init__(
        self,
        element_ids: set[str] | None = None,
        filter_everything: bool = True,
        name: str = "ErrorFilter",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> None:
        """Initialize NoneCollector.

        Parameters
        ----------
        element_ids: list[str] | None, optional
            List of Pipeline Elements for which InvalidInstances can be removed.
        filter_everything: bool, default=True
            If True, element_ids are ignored and all InvalidInstances are removed.
        name: str, default="ErrorFilter"
            Name of the pipeline element.
        n_jobs: int, default=1
            Number of parallel jobs to use.
        uuid: str | None, optional
            UUID of the pipeline element.

        Raises
        ------
        ValueError
            If element_ids is None and filter_everything is False.

        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        self.error_indices = []
        if element_ids is None:
            if not filter_everything:
                raise ValueError(
                    "If element_ids is None, filter_everything must be True",
                )
            element_ids = set()
        if not isinstance(element_ids, set):
            element_ids = set(element_ids)
        self.element_ids = element_ids
        self.filter_everything = filter_everything
        self.n_total = 0
        self._requires_fitting = True

    @classmethod
    def from_element_list(
        cls,
        element_list: Iterable[TransformingPipelineElement],
        name: str = "ErrorFilter",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> Self:
        """Initialize NoneCollector from a list of Pipeline Elements.

        Parameters
        ----------
        element_list: Iterable[TransformingPipelineElement]
            List of Pipeline Elements for which None can be removed.
        name: str, default="ErrorFilter"
            Name of the pipeline element.
        n_jobs: int, default=1
            Number of parallel jobs to use.
        uuid: str | None, optional
            UUID of the pipeline element.

        Returns
        -------
        Self
            Constructed ErrorFilter object.

        """
        element_ids = {element.uuid for element in element_list}
        return cls(
            element_ids,
            filter_everything=False,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this element.

        Parameters
        ----------
        deep: bool
            If True, will return a deep copy of parameters for this estimator.

        Returns
        -------
        dict[str, Any]
            Parameter names mapped to their values.

        """
        params = super().get_params(deep=deep)
        params["filter_everything"] = self.filter_everything
        if deep:
            params["element_ids"] = {str(element_id) for element_id in self.element_ids}
        else:
            params["element_ids"] = self.element_ids
        return params

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters for this element.

        Parameters
        ----------
        parameters: Any
            Dict of arameters to set.

        Raises
        ------
        TypeError
            If element_ids is not a set.

        Returns
        -------
        Self
            Self with updated parameters.

        """
        param_copy = dict(parameters)
        if "element_ids" in param_copy:
            element_ids = param_copy.pop("element_ids")
            if not isinstance(element_ids, set):
                raise TypeError(f"Unexpected Type: {type(element_ids)}")
            self.element_ids = element_ids
        if "filter_everything" in param_copy:
            self.filter_everything = bool(param_copy.pop("filter_everything"))
        super().set_params(**param_copy)
        return self

    def check_removal(self, value: Any) -> bool:
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
        if not isinstance(value, InvalidInstance):
            return False
        return self.filter_everything or value.element_id in self.element_ids

    def fit(self, values: AnyVarSeq, labels: Any = None) -> Self:  # noqa: ARG002
        """Fit to input values.

        Only for compatibility with sklearn Pipelines.

        Parameters
        ----------
        values: AnyVarSeq
            Values used for fitting. For compatibility with sklearn, not used.
        labels: Any
            Label used for fitting. For compatibility with sklearn, not used.

        Returns
        -------
        Self
            Fitted ErrorFilter.

        """
        return self

    def fit_transform(
        self,
        values: TypeFixedVarSeq,
        labels: Any = None,
    ) -> TypeFixedVarSeq:
        """Transform values and return a list without the None values.

        So far fit does nothing and hence is only called for consitency.

        Parameters
        ----------
        values: TypeFixedVarSeq
            Iterable to which element is fitted and which is subsequently transformed.
        labels: Any
            Label used for fitting. For compatibility with sklearn, not used.

        Returns
        -------
        TypeFixedVarSeq
            Iterable where invalid instances were removed.

        """
        self.fit(values, labels)
        return self.transform(values)

    def co_transform(self, values: TypeFixedVarSeq) -> TypeFixedVarSeq:
        """Remove rows at positions with discarded values.

        This ensures that rows of this instance maintain a one to one correspondence
        with the rows of data seen during transformation.

        Parameters
        ----------
        values: TypeFixedVarSeq
            Values to be transformed.

        Raises
        ------
        ValueError
            If the length of the values does not match the length of the values in fit.
        TypeError
            If the type of values is not a list or numpy array.

        Returns
        -------
        TypeFixedVarSeq
            Input where rows are removed.

        """
        if self.n_total != len(values):
            raise ValueError("Length of values does not match length of values in fit")
        if isinstance(values, list):
            out_list = []
            for idx, value in enumerate(values):
                if idx not in self.error_indices:
                    out_list.append(value)
            return out_list  # type: ignore  # mypy does not support higher-kinded types
        if isinstance(values, np.ndarray):
            return np.delete(values, self.error_indices, axis=0)  # type: ignore
        if isinstance(values, pd.Series):
            return values.drop(values.index[self.error_indices])
        raise TypeError(f"Unexpected Type: {type(values)}")

    def transform(self, values: TypeFixedVarSeq) -> TypeFixedVarSeq:
        """Transform values and return a list without the None values.

        IMPORTANT: Changes number of elements in the iterable.

        Parameters
        ----------
        values: TypeFixedVarSeq
            Iterable of which according invalid instances are removed.

        Returns
        -------
        TypeFixedVarSeq
            Iterable where invalid instances were removed.

        """
        self.n_total = len(values)
        self.error_indices = []
        for i, value in enumerate(values):
            if self.check_removal(value):
                self.error_indices.append(i)
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
        return self.pretransform_single(value)

    def pretransform_single(self, value: Any) -> Any:
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
                filter_element_id=self.uuid,
                message=value.message,  # type: ignore
            )
        return value


class _MultipleErrorFilter:
    """Combines multiple ErrorFilters into one object."""

    error_filter_list: list[ErrorFilter]

    def __init__(self, error_filter_list: list[ErrorFilter]) -> None:
        """Initialize NoneCollector.

        Parameters
        ----------
        error_filter_list: list[ErrorFilter]
            List of ErrorFilter objects.

        """
        self.error_filter_list = error_filter_list
        prior_remover_dict = {}
        for i, remover_element in enumerate(error_filter_list):
            prior_remover_dict[remover_element] = error_filter_list[:i]
        self.prior_remover_dict = prior_remover_dict

    def transform(self, values: TypeFixedVarSeq) -> TypeFixedVarSeq:
        """Transform values and return a list without the invalid values.

        Parameters
        ----------
        values: TypeFixedVarSeq
            Iterable to which element is fitted and which is subsequently transformed.

        Returns
        -------
        TypeFixedVarSeq
            Iterable where invalid instances were removed.

        """
        for error_filter in self.error_filter_list:
            values = error_filter.transform(values)
        return values

    def co_transform(self, values: TypeFixedVarSeq) -> TypeFixedVarSeq:
        """Remove rows at positions with discarded values.

        Parameters
        ----------
        values: TypeFixedVarSeq
            Iterable to which element is fitted and which is subsequently transformed.

        Returns
        -------
        TypeFixedVarSeq
            Iterable without rows removed during the transformation.

        """
        for error_filter in self.error_filter_list:
            values = error_filter.co_transform(values)
        return values

    def fit_transform(self, values: TypeFixedVarSeq) -> TypeFixedVarSeq:
        """Transform values and return a list without the None values.

        Parameters
        ----------
        values: TypeFixedVarSeq
            Iterable to which element is fitted and which is subsequently transformed.

        Returns
        -------
        TypeFixedVarSeq
            Iterable where invalid instances were removed.

        """
        for error_filter in self.error_filter_list:
            values = error_filter.fit_transform(values)
        return values

    def register_removed(self, index: int, value: RemovedInstance) -> None:
        """Register an invalid instance.

        Parameters
        ----------
        index: int
            Index of the invalid instance.
        value: Any
            Value of the invalid instance.

        Raises
        ------
        TypeError
            If value is not a RemovedInstance.
        ValueError
            If value is not captured by any ErrorFilter.

        """
        if not isinstance(value, RemovedInstance):
            raise TypeError(f"Unexpected Type: {type(value)}")

        for error_filter in self.error_filter_list:
            if value.filter_element_id == error_filter.uuid:
                new_index = index
                for prior_remover in self.prior_remover_dict[error_filter]:
                    new_index -= len(prior_remover.error_indices)
                error_filter.error_indices.append(index)
                break
        else:
            raise ValueError(
                "Invalid instance not captured by any ErrorFilter:"
                f" {value.filter_element_id}",
            )

    def set_total(self, total: int) -> None:
        """Set the total number of instances.

        Parameters
        ----------
        total: int
            Total number of instances seen during transformation.

        """
        for error_filter in self.error_filter_list:
            error_filter.n_total = total
            total -= len(error_filter.error_indices)


class FilterReinserter(ABCPipelineElement, Generic[_T]):
    """Fill None values with a Dummy value."""

    fill_value: _T
    error_filter_id: str
    _error_filter: ErrorFilter | None
    n_total: int

    def __init__(
        self,
        error_filter_id: str,
        fill_value: _T,
        name: str = "FilterReinserter",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> None:
        """Initialize FilterReinserter.

        Parameters
        ----------
        error_filter_id: str
            Id of the ErrorFilter to use for filling removed values.
        fill_value: Any
            Value which is used to fill removed values.
        name: str, default="FilterReinserter"
            Name of the pipeline element.
        n_jobs: int, default=1
            Number of parallel jobs to use.
        uuid: str | None, optional
            UUID of the pipeline element.

        """
        super().__init__(name=name, n_jobs=n_jobs, uuid=uuid)
        self.error_filter_id = error_filter_id
        self._error_filter = None
        self.fill_value = fill_value

    @classmethod
    def from_error_filter(
        cls,
        error_filter: ErrorFilter,
        fill_value: _T,
        name: str = "FilterReinserter",
        n_jobs: int = 1,
        uuid: str | None = None,
    ) -> Self:
        """Initialize FilterReinserter from a ErrorFilter object.

        Parameters
        ----------
        error_filter: ErrorFilter
            ErrorFilter to use for filling removed values.
        fill_value: Any
            Value which is used to fill removed values.
        name: str, default="FilterReinserter"
            Name of the pipeline element.
        n_jobs: int, default=1
            Number of parallel jobs to use.
        uuid: str | None, optional
            UUID of the pipeline element.

        Returns
        -------
        Self
            Constructed FilterReinserter object.

        """
        filler = cls(
            error_filter_id=error_filter.uuid,
            fill_value=fill_value,
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )
        filler.error_filter = error_filter
        return filler

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this element.

        Parameters
        ----------
        deep: bool
            If True, will return a deep copy of parameters for this estimator.

        Returns
        -------
        dict[str, Any]
            Parameter names mapped to their values.

        """
        params = super().get_params(deep=deep)
        if deep:
            params["error_filter_id"] = str(self.error_filter_id)
            if self.fill_value is not None:
                params["fill_value"] = type(self.fill_value)(self.fill_value)  # type: ignore
            else:
                params["fill_value"] = None
        else:
            params["error_filter_id"] = self.error_filter_id
            params["fill_value"] = self.fill_value
        return params

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters for this element.

        Parameters
        ----------
        parameters: Any
            Parameter dict.

        Returns
        -------
        Self
            The instance itself.

        """
        parameter_copy = dict(parameters)
        if "error_filter_id" in parameter_copy:
            self.error_filter_id = str(parameter_copy.pop("error_filter_id"))
        if "fill_value" in parameter_copy:
            self.fill_value = parameter_copy.pop("fill_value")
        super().set_params(**parameter_copy)
        return self

    @property
    def error_filter(self) -> ErrorFilter:
        """Get the ErrorFilter connected to this FilterReinserter.

        Raises
        ------
        ValueError
            If the ErrorFilter is not set.

        """
        if self._error_filter is None:
            raise ValueError("ErrorFilter not set")
        return self._error_filter

    @error_filter.setter
    def error_filter(self, error_filter: ErrorFilter) -> None:
        """Set the ErrorFilter.

        Parameters
        ----------
        error_filter: ErrorFilter
            ErrorFilter to set.

        """
        self._error_filter = error_filter

    def select_error_filter(self, error_filter_list: list[ErrorFilter]) -> Self:
        """Select the ErrorFilter from a list of ErrorFilters.

        Parameters
        ----------
        error_filter_list: list[ErrorFilter]
            List of ErrorFilters to select from.

        Raises
        ------
        ValueError
            If the ErrorFilter with the given id is not found in the list.

        Returns
        -------
        Self
            FilterReinserter with updated ErrorFilter.

        """
        for error_filter in error_filter_list:
            if error_filter.uuid == self.error_filter_id:
                self.error_filter = error_filter
                break
        else:
            raise ValueError(f"ErrorFilter with id {self.error_filter_id} not found")
        return self

    @staticmethod
    def _validate_input_values(values: TypeFixedVarSeq) -> None:
        """Validate the input values.

        Parameters
        ----------
        values: TypeFixedVarSeq
            Values to be checked.

        Raises
        ------
        TypeError
            If values is not a list, numpy array or pandas Series.

        """
        if not isinstance(values, (list, np.ndarray, pd.Series)):
            raise TypeError(
                "Values must be a list, numpy array or pandas Series. "
                f"Received: {type(values)}",
            )

    # pylint: disable=unused-argument
    def fit(
        self,
        values: TypeFixedVarSeq,
        labels: Any = None,  # noqa: ARG002
        **params: Any,  # noqa: ARG002
    ) -> Self:
        """Fit to input values.

        Only for compatibility with sklearn Pipelines.

        Parameters
        ----------
        values: TypeFixedVarSeq
            Values used for fitting.
        labels: Any
            Label used for fitting. For compatibility with sklearn, not used.
        **params: Any
            Additional keyword arguments. For compatibility with sklearn, not used.

        Returns
        -------
        Self
            Fitted FilterReinserter.

        """
        self._validate_input_values(values)
        return self

    # pylint: disable=unused-argument
    def fit_transform(
        self,
        values: TypeFixedVarSeq,
        labels: Any = None,  # noqa: ARG002
        **params: Any,  # noqa: ARG002
    ) -> TypeFixedVarSeq:
        """Transform values and return a list without the Invalid values.

        So far fit does nothing and hence is only called for consitency.

        Parameters
        ----------
        values: TypeFixedVarSeq
            Iterable to which element is fitted and which is subsequently transformed.
        labels: Any
            Label used for fitting. For compatibility with sklearn, not used.
        **params: Any
            Additional keyword arguments. For compatibility with sklearn, not used.

        Returns
        -------
        TypeFixedVarSeq
            Iterable where invalid instances were removed.

        """
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
        return self.pretransform_single(value)

    def pretransform_single(self, value: Any) -> Any:
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
        if (
            isinstance(value, RemovedInstance)
            and value.filter_element_id == self.error_filter.uuid
        ):
            return self.fill_value
        return value

    def transform(
        self,
        values: TypeFixedVarSeq,
        **_params: Any,
    ) -> TypeFixedVarSeq:
        """Transform iterable of values by removing invalid instances.

        IMPORTANT: Changes number of elements in the iterable.

        Parameters
        ----------
        values: TypeFixedVarSeq
            Iterable of which according invalid instances are removed.
        **_params: Any
            Additional keyword arguments.

        Raises
        ------
        ValueError
            If the length of the values does not match the length of the values in fit.

        Returns
        -------
        TypeFixedVarSeq
            Iterable where invalid instances were removed.

        """
        self._validate_input_values(values)
        if len(values) != self.error_filter.n_total - len(
            self.error_filter.error_indices,
        ):
            expected_length = self.error_filter.n_total - len(
                self.error_filter.error_indices,
            )
            raise ValueError(
                "Length of values does not match length of values in fit. "
                f"Expected: {expected_length} - Received :{len(values)}",
            )
        return self.fill_with_dummy(values)

    def _fill_list(self, list_to_fill: Sequence[_S]) -> Sequence[_S | _T]:
        """Fill a list with dummy values.

        Parameters
        ----------
        list_to_fill: list[Number]
            List to fill with dummy values.

        Raises
        ------
        AssertionError
            If the length of the list does not match the expected length.

        Returns
        -------
        list[Number]
            List where dummy values were inserted to replace instances which could not
            be processed.

        """
        filled_list: list[_S | _T] = []
        next_value_pos = 0
        for index in range(len(list_to_fill) + len(self.error_filter.error_indices)):
            if index in self.error_filter.error_indices:
                filled_list.append(self.fill_value)
            else:
                filled_list.append(list_to_fill[next_value_pos])
                next_value_pos += 1
        if len(list_to_fill) != next_value_pos:
            raise AssertionError(
                "Length of list does not match length of values in fit",
            )
        return filled_list

    def _fill_numpy_arr(self, value_array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Fill a numpy array with dummy values.

        Parameters
        ----------
        value_array: npt.NDArray[Any]
            Numpy array to fill with dummy values.

        Returns
        -------
        npt.NDArray[Any]
            Numpy array where unprocessable values are replaced by dummy values.

        """
        fill_value = self.fill_value
        output_shape = list(value_array.shape)
        output_shape[0] += len(self.error_filter.error_indices)
        has_value_indices = np.ones(output_shape[0], dtype=bool)
        has_value_indices[self.error_filter.error_indices] = False

        try:
            dtype = np.common_type(value_array, np.array([self.fill_value]))
        except TypeError:
            dtype = np.object_

        output_matrix: npt.NDArray[Any]
        output_matrix = np.full(output_shape, fill_value, dtype=dtype)
        output_matrix[has_value_indices, ...] = value_array
        return output_matrix

    def fill_with_dummy(
        self,
        value_container: AnyVarSeq,
    ) -> AnyVarSeq:
        """Insert dummy values at the positions in the value container.

        Parameters
        ----------
        value_container: TypeFixedVarSeq
            Iterable to fill with dummy values.

        Raises
        ------
        TypeError
            If value_container is not a list or numpy array.

        Returns
        -------
        AnyVarSeq
            Iterable where unprocessable values are replaced by dummy values.

        """
        if isinstance(value_container, list):
            return self._fill_list(value_container)  # type: ignore
        if isinstance(value_container, np.ndarray):
            return self._fill_numpy_arr(value_container)  # type: ignore
        raise TypeError(f"Unexpected Type: {type(value_container)}")
