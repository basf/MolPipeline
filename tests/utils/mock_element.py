"""Mock PipelineElement for testing."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Optional

import numpy as np

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from molpipeline.abstract_pipeline_elements.core import (
    InvalidInstance,
    TransformingPipelineElement,
)


class MockTransformingPipelineElement(TransformingPipelineElement):
    """Mock element for testing."""

    def __init__(
        self,
        *,
        invalid_values: set[Any] | None = None,
        return_as_numpy_array: bool = False,
        name: str = "dummy",
        uuid: Optional[str] = None,
        n_jobs: int = 1,
    ) -> None:
        """Initialize MockTransformingPipelineElement.

        Parameters
        ----------
        invalid_values: set[Any] | None, optional (default=None)
            Set of values to consider invalid.
        return_as_numpy_array: bool, optional (default=False)
            If True return output as numpy array, otherwise as list.
        name: str, optional (default="dummy")
            Name of PipelineElement
        uuid: str, optional (default=None)
            Unique identifier of PipelineElement.
        n_jobs: int, optional (default=1)
            Number of jobs to run in parallel.
        """
        super().__init__(name=name, uuid=uuid, n_jobs=n_jobs)
        if invalid_values is None:
            invalid_values = set()
        self.invalid_values = invalid_values
        self.return_as_numpy_array: bool = return_as_numpy_array

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return all parameters defining the object.

        Parameters
        ----------
        deep: bool
            If True get a deep copy of the parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all parameters defining the object.
        """
        params = super().get_params(deep)
        if deep:
            params["invalid_values"] = copy.deepcopy(self.invalid_values)
            params["return_as_numpy_array"] = copy.deepcopy(self.return_as_numpy_array)
        else:
            params["invalid_values"] = self.invalid_values
            params["return_as_numpy_array"] = self.return_as_numpy_array
        return params

    def set_params(self, **parameters: Any) -> Self:
        """Set parameters of the object.

        Parameters
        ----------
        parameters: Any
            Dictionary containing all parameters defining the object.

        Returns
        -------
        Self
            MockTransformingPipelineElement with updated parameters.
        """
        super().set_params(**parameters)
        if "invalid_values" in parameters:
            self.invalid_values = set(parameters["invalid_values"])
        if "return_as_numpy_array" in parameters:
            self.return_as_numpy_array = bool(parameters["return_as_numpy_array"])
        return self

    def pretransform_single(self, value: Any) -> Any:
        """Transform input value to other value.

        Parameters
        ----------
        value: Any
            Input value.

        Returns
        -------
        Any
            Other value.
        """
        if value in self.invalid_values:
            return InvalidInstance(
                self.uuid,
                f"Invalid input value by mock: {value}",
                self.name,
            )
        return value

    def assemble_output(self, value_list: Iterable[Any]) -> Any:
        """Aggregate rows, which in most cases is just return the list.

        Some representations might be better representd as a single object. For example a list of vectors can
        be transformed to a matrix.

        Parameters
        ----------
        value_list: Iterable[Any]
            Iterable of transformed rows.

        Returns
        -------
        Any
            Aggregated output. This can also be the original input.
        """
        if self.return_as_numpy_array:
            return np.array(list(value_list))
        return list(value_list)
