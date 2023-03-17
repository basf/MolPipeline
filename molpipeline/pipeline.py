"""Defines the pipeline which handles pipeline elements."""
from __future__ import annotations

import multiprocessing
from typing import Any, Literal, Union

import numpy as np
from molpipeline.abstract_pipeline_elements.core import ABCPipelineElement
from molpipeline.utils.multi_proc import check_available_cores
from molpipeline.utils.none_handling import NoneCollector


class MolPipeline:
    """Contains the PipeElements which describe the functionality of the pipeline."""

    _n_jobs: int
    _pipeline_element_list: list[ABCPipelineElement]

    handle_nones: Literal["raise", "record_remove", "fill_dummy"]

    def __init__(
        self,
        pipeline_element_list: list[ABCPipelineElement],
        handle_nones: Literal["raise", "record_remove", "fill_dummy"] = "raise",
        fill_value: Any = np.nan,
        n_jobs: int = 1,
        name: str = "MolPipeline",
    ):
        """Initialize MolPipeline."""
        self._pipeline_element_list = pipeline_element_list
        if handle_nones == "raise":
            for element in self.pipeline_elements:
                element.none_handling = "raise"
        if handle_nones in ["record_remove", "fill_dummy"]:
            for element in self.pipeline_elements:
                element.none_handling = "record_remove"

        self.n_jobs = n_jobs
        self.name = name
        self.handle_nones = handle_nones
        self.none_collector = NoneCollector(fill_value=fill_value)

    @property
    def n_jobs(self) -> int:
        """Return the number of cores to use in transformation step."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, requested_jobs: int) -> None:
        """Set the number of cores to use in transformation step."""
        self._n_jobs = check_available_cores(requested_jobs)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "pipeline_element_list": self._pipeline_element_list,
            "n_jobs": self.n_jobs,
            "name": self.name,
            "handle_nones": self.handle_nones,
            "fill_value": self.none_collector.fill_value,
        }

    @property
    def pipeline_elements(self) -> list[ABCPipelineElement]:
        """Get a copy of the list of pipeline elements."""
        return self._pipeline_element_list[:]  # [:] to create shallow copy.

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
        for p_element in self._pipeline_element_list:
            iter_input = p_element.fit_transform(iter_input)
            none_values = p_element.none_collector.none_indices
            surviving_indices = np.delete(surviving_indices, none_values)

        if self.handle_nones == "fill_dummy":
            nan_indices = np.delete(all_indices, surviving_indices)
            self.none_collector.none_indices = list(nan_indices)
            return self.none_collector.fill_with_dummy(iter_input)

        return iter_input

    def _transform_single(self, input_value: Any) -> Any:
        iter_value = input_value
        for p_element in self._pipeline_element_list:  # type: ABCPipelineElement
            iter_value = p_element.transform_single(iter_value)
            if iter_value is None:
                return iter_value
        return iter_value

    def transform(self, x_input: Any) -> Any:
        """Transform the input according to the sequence of provided PipelineElements."""
        self.none_collector.none_indices = []
        last_element = self._pipeline_element_list[-1]
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

        if self.handle_nones == "fill_dummy":
            return self.none_collector.fill_with_dummy(output)
        return output

    def _finish(self) -> None:
        """Inform each pipeline element that the iterations have finished."""
        for p_element in self._pipeline_element_list:  # type: ABCPipelineElement
            p_element.finish()

    def _transform_iterator(self, x_input: Any) -> Any:
        if self.n_jobs > 1:
            with multiprocessing.Pool(self.n_jobs) as pool:
                for i, transformed_value in enumerate(
                    pool.imap(self._transform_single, x_input)
                ):
                    if transformed_value is None:
                        if self.handle_nones == "raise":
                            raise ValueError(f"Encountered None in position: {i}")
                        elif self.handle_nones in ["record_remove", "fill_dummy"]:
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
                    if self.handle_nones == "raise":
                        raise ValueError(f"Encountered None in position: {i}")
                    elif self.handle_nones in ["record_remove", "fill_dummy"]:
                        self.none_collector.none_indices.append(i)
                        continue
                    raise AssertionError("This part of the code should be unreachable!")
                yield transformed_value
        self._finish()

    def copy(self) -> MolPipeline:
        """Return a copy of the MolPipeline."""
        return self[:]

    def __getitem__(self, index: slice) -> MolPipeline:
        """Get new MolPipeline with a slice of elements."""
        element_slice = self.pipeline_elements[index]
        parameter = {
            key: value
            for key, value in self.params.items()
            if key != "pipeline_element_list"
        }
        if isinstance(element_slice, list):
            element_slice_copy = [element.copy() for element in element_slice]
            return MolPipeline(element_slice_copy, **parameter)
        elif isinstance(element_slice, ABCPipelineElement):
            return MolPipeline([element_slice.copy()], **parameter)
        else:
            raise AssertionError(f"Unexpected Element type: {type(element_slice)}")

    def __add__(self, other: Union[ABCPipelineElement, MolPipeline]) -> MolPipeline:
        """Concatenate two Pipelines or add a PipelineElement."""
        pipeline_element_list = self.pipeline_elements[:]
        parameter = {
            key: value
            for key, value in self.params.items()
            if key != "pipeline_element_list"
        }
        if isinstance(other, ABCPipelineElement):
            pipeline_element_list.append(other)
        elif isinstance(other, MolPipeline):
            pipeline_element_list.extend(other.pipeline_elements)
        else:
            raise TypeError(f"{type(other)} is not supported for addition!")
        return MolPipeline(pipeline_element_list, **parameter)
