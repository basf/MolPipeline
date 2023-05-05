"""Defines the pipeline which handles pipeline elements."""
from __future__ import annotations

import copy
import multiprocessing
import warnings
from typing import Any, Literal, Optional, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import numpy as np
from molpipeline.abstract_pipeline_elements.core import ABCPipelineElement
from molpipeline.utils.json_operations import pipeline_element_from_json
from molpipeline.utils.multi_proc import check_available_cores
from molpipeline.utils.none_handling import NoneCollector
from molpipeline.utils.molpipe_types import NoneHandlingOptions


class MolPipeline:
    """Contains the PipeElements which describe the functionality of the pipeline."""

    _n_jobs: int
    _element_list: list[ABCPipelineElement]

    # pylint: disable=R0913
    def __init__(
        self,
        element_list: list[ABCPipelineElement],
        none_handling: NoneHandlingOptions = "raise",
        fill_value: Any = np.nan,
        n_jobs: int = 1,
        name: str = "MolPipeline",
        handle_nones: Optional[NoneHandlingOptions] = None,
    ):
        """Initialize MolPipeline.

        Parameters
        ----------
        element_list: list[ABCPipelineElement]
            List of Pipeline Elements which form the pipeline.
        none_handling: NoneHandlingOptions
            Defines how Nones are handled.
        fill_value: Any
            Value used to fill Nones. Only important if none_handling is "fill_dummy".
        n_jobs:
            Number of cores used.
        name: str
            Name of pipeline.
        handle_nones: Optional[NoneHandlingOptions]
            For backwards compatibility. If not None, this value is used for none_handling.
        """
        self._element_list = element_list

        self.n_jobs = n_jobs
        self.name = name
        self.none_handling = none_handling
        if handle_nones is not None:
            warnings.warn("handle_nones is deprecated. Use none_handling instead.")
            self.none_handling = handle_nones
        self.none_collector = NoneCollector(fill_value=fill_value)

    @classmethod
    def from_json(cls, json_dict: dict[str, Any]) -> Self:
        """Create object from json dict.

        Parameters
        ----------
        json_dict: dict[str, Any]
            Json dict containing the parameters to create the object.
        Returns
        -------
        MolPipeline:
            MolPipeline created from the json dict.
        """
        # Transform pipeline elements from json to objects.
        json_dict_copy = dict(json_dict)  # copy, because the dict is modified
        element_json_list = json_dict_copy.pop("element_list")
        element_list = []
        for element_json in element_json_list:
            element_list.append(pipeline_element_from_json(element_json))
        # Replace json list with list of constructed pipeline elements.
        json_dict_copy["element_list"] = element_list
        return cls(**json_dict_copy)

    @property
    def none_handling(self) -> Literal["raise", "record_remove", "fill_dummy"]:
        """Get string defining the handling of Nones.

        Returns
        -------
        Literal["raise", "record_remove", "fill_dummy"]
        """
        return self._none_handling

    @none_handling.setter
    def none_handling(
        self, none_handling: Literal["raise", "record_remove", "fill_dummy"]
    ) -> None:
        """Set string defining the handling of Nones.

        Parameters
        ----------
        none_handling: Literal["raise", "record_remove", "fill_dummy"]
            Specifies how molecules which map to None are handled:
            - raise: Raises an error if a None is encountered.
            - record_remove: Removes the molecule from the list and records the position.
            - fill_dummy: Fills the output with a dummy value on the position of the None.
        Returns
        -------
        None
        """
        self._none_handling = none_handling
        if none_handling == "raise":
            for element in self._element_list:
                element.none_handling = "raise"
        elif none_handling in ["record_remove", "fill_dummy"]:
            for element in self._element_list:
                element.none_handling = "record_remove"
        else:
            raise ValueError(
                f"none_handling must be one of ['raise', 'record_remove', 'fill_dummy'], but is {none_handling}."
            )

    @property
    def n_jobs(self) -> int:
        """Return the number of cores to use in transformation step."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, requested_jobs: int) -> None:
        """Set the number of cores to use in transformation step.

        Parameters
        ----------
        requested_jobs: int
            Number of cores requested for transformation steps.
            If fewer cores than requested are available, the number of cores is set to maximum available.
        Returns
        -------
        None
        """
        self._n_jobs = check_available_cores(requested_jobs)

    @property
    def none_indices(self) -> list[int]:
        """Get list of indices of None values."""
        return self.none_collector.none_indices

    @property
    def parameters(self) -> dict[str, Any]:
        """Get all parameters defining the object."""
        return self.get_params()

    @parameters.setter
    def parameters(self, parameter_dict: dict[str, Any]) -> None:
        """Set parameters of the pipeline and pipeline elements.

        Parameters
        ----------
        parameter_dict: dict[str, Any]
            Dictionary containing the parameter names and corresponding values to be set.

        Returns
        -------
        None
        """
        self.set_params(parameter_dict)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get all parameters defining the object.

        Parameters
        ----------
        deep: bool
            If True get a deep copy of the parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the parameter names and corresponding values.
        """
        if deep:
            return {
                "element_list": self.element_list,
                "n_jobs": self.n_jobs,
                "name": self.name,
                "none_handling": copy.copy(self.none_handling),
                "fill_value": copy.copy(self.none_collector.fill_value),
            }
        return {
            "element_list": self._element_list,
            "n_jobs": self.n_jobs,
            "name": self.name,
            "none_handling": self.none_handling,
            "fill_value": self.none_collector.fill_value,
        }

    def set_params(self, parameter_dict: dict[str, Any]) -> Self:
        """Set parameters of the pipeline and pipeline elements.

        Parameters
        ----------
        parameter_dict: dict[str, Any]
            Dictionary containing the parameter names and corresponding values to be set.

        Returns
        -------
        Self
            MolPipeline object with updated parameters.
        """
        if "element_list" in parameter_dict:
            self._element_list = parameter_dict["element_list"]
        if "n_jobs" in parameter_dict:
            self.n_jobs = parameter_dict["n_jobs"]
        if "none_handling" in parameter_dict:
            self.none_handling = parameter_dict["none_handling"]
        if "fill_value" in parameter_dict:
            self.none_collector.fill_value = parameter_dict["fill_value"]
        if "name" in parameter_dict:
            self.name = parameter_dict["name"]
        return self

    @property
    def element_list(self) -> list[ABCPipelineElement]:
        """Get a shallow copy from the list of pipeline elements."""
        return self._element_list[:]  # [:] to create shallow copy.

    def fit(
        self,
        x_input: Any,
        y_input: Any = None,
        **fit_params: dict[Any, Any],
    ) -> None:
        """Fit the MolPipeline according to x_input.

        Parameters
        ----------
        x_input: Any
            Molecular representations which are subsequently processed.
        y_input: Any
            Optional label of input. Only for SKlearn compatibility.
        fit_params: Any
            Parameters. Only for SKlearn compatibility.

        Returns
        -------
        None
        """
        _ = y_input  # Making pylint happy
        _ = fit_params  # Making pylint happy
        self.fit_transform(x_input)

    def fit_transform(
        self,
        x_input: Any,
        y_input: Any = None,
        **fit_params: dict[str, Any],
    ) -> Any:
        """Fit the MolPipeline according to x_input and return the transformed molecules.

        Parameters
        ----------
        x_input: Any
            Molecular representations which are subsequently processed.
        y_input: Any
            Optional label of input. Only for SKlearn compatibility.
        fit_params: Any
            Parameters. Only for SKlearn compatibility.

        Returns
        -------
        Any
            Transformed molecules.
        """
        self.none_collector.none_indices = []
        iter_input = x_input
        _ = y_input  # Making pylint happy, does no(t a)thing
        _ = fit_params  # Making pylint happy
        surviving_indices = np.arange(len(iter_input))
        all_indices = np.arange(len(iter_input))
        for p_element in self._element_list:
            iter_input = p_element.fit_transform(iter_input)
            none_values = p_element.none_collector.none_indices
            surviving_indices = np.delete(surviving_indices, none_values)

        nan_indices = np.delete(all_indices, surviving_indices)
        self.none_collector.none_indices = list(nan_indices)
        if self.none_handling == "fill_dummy":
            return self.none_collector.fill_with_dummy(iter_input)

        return iter_input

    def to_json(self) -> dict[str, Any]:
        """Convert the pipeline to a json string.

        Returns
        -------
        dict[str, Any]
            Json representation of the pipeline.
        """
        json_dict = self.parameters
        json_dict["element_list"] = [
            p_element.to_json() for p_element in self.element_list
        ]
        return json_dict

    def _transform_single(self, input_value: Any) -> Any:
        iter_value = input_value
        for p_element in self._element_list:  # type: ABCPipelineElement
            iter_value = p_element.transform_single(iter_value)
            if iter_value is None:
                return iter_value
        return iter_value

    def transform(self, x_input: Any) -> Any:
        """Transform the input according to the sequence of provided PipelineElements.

        Parameters
        ----------
        x_input: Any
            Molecular representations which are subsequently transformed.

        Returns
        -------
        Any
            Transformed molecular representations.
        """
        self.none_collector.none_indices = []
        last_element = self._element_list[-1]
        if hasattr(last_element, "assemble_output"):
            output = last_element.assemble_output(
                (
                    single
                    for single in self._transform_iterator(x_input)
                    if single is not None
                )
            )
        else:
            output = list(self._transform_iterator(x_input))

        if self.none_handling == "fill_dummy":
            return self.none_collector.fill_with_dummy(output)
        return output

    def _finish(self) -> None:
        """Inform each pipeline element that the iterations have finished."""
        for p_element in self._element_list:  # type: ABCPipelineElement
            p_element.finish()

    def _transform_iterator(self, x_input: Any) -> Any:
        if self.n_jobs > 1:
            with multiprocessing.Pool(self.n_jobs) as pool:
                for i, transformed_value in enumerate(
                    pool.imap(self._transform_single, x_input)
                ):
                    if transformed_value is None:
                        if self.none_handling == "raise":
                            raise ValueError(f"Encountered None in position: {i}")

                        if self.none_handling in ["record_remove", "fill_dummy"]:
                            self.none_collector.none_indices.append(i)
                            continue
                        raise AssertionError(
                            "This part of the code should be unreachable!"
                        )
                    yield transformed_value
        else:
            for i, value in enumerate(x_input):
                transformed_value = self._transform_single(value)
                if transformed_value is None:
                    if self.none_handling == "raise":
                        raise ValueError(f"Encountered None in position: {i}")
                    if self.none_handling in ["record_remove", "fill_dummy"]:
                        self.none_collector.none_indices.append(i)
                        continue
                    raise AssertionError("This part of the code should be unreachable!")
                yield transformed_value
        self._finish()

    def copy(self) -> MolPipeline:
        """Return a copy of the MolPipeline.

        PipelineElements are copied as well and thus are not linked to the original.

        Returns
        -------
        MolPipeline
            A copy of the MolPipeline.
        """
        return self[:]

    def __getitem__(self, index: slice) -> MolPipeline:
        """Get new MolPipeline with a slice of elements.

        Parameters
        ----------
        index: slice
            Slice which specifies the elements to be included in the new MolPipeline.
        Returns
        -------
        MolPipeline
            New MolPipeline with the specified elements.
        """
        parameter = self.parameters
        element_list = parameter.pop("element_list")
        element_slice = element_list[index]

        if isinstance(element_slice, list):
            element_slice_copy = [element.copy() for element in element_slice]
            return MolPipeline(element_slice_copy, **parameter)

        if isinstance(element_slice, ABCPipelineElement):
            return MolPipeline([element_slice.copy()], **parameter)

        raise AssertionError(f"Unexpected Element type: {type(element_slice)}")

    def __add__(self, other: Union[ABCPipelineElement, MolPipeline]) -> MolPipeline:
        """Concatenate two Pipelines or add a PipelineElement."""
        element_list = self.element_list[:]
        parameter = {
            key: value
            for key, value in self.parameters.items()
            if key != "element_list"
        }
        if isinstance(other, ABCPipelineElement):
            element_list.append(other)
        elif isinstance(other, MolPipeline):
            element_list.extend(other.element_list)
        else:
            raise TypeError(f"{type(other)} is not supported for addition!")
        return MolPipeline(element_list, **parameter)
