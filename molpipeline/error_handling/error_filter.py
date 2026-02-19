"""Classes and functions for detecting and handling None values."""

from collections.abc import Iterable
from typing import Any, Self, TypeVar

import numpy as np
import pandas as pd

from molpipeline.abstract_pipeline_elements.core import (
    ABCPipelineElement,
    InvalidInstance,
    RemovedInstance,
    TransformingPipelineElement,
)
from molpipeline.utils.molpipeline_types import AnyVarSeq, TypeFixedVarSeq

__all__ = ["ErrorFilter", "_MultipleErrorFilter"]

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
