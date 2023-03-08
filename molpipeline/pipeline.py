"""Defines the pipeline which handles pipeline elements."""
from __future__ import annotations

import multiprocessing
from typing import Any, Union

from molpipeline.abstract_pipeline_elements.core import ABCPipelineElement
from molpipeline.utils.multi_proc import check_available_cores


class MolPipeline:
    """Contains the PipeElements which describe the functionality of the pipeline."""

    _n_jobs: int
    _pipeline_element_list: list[ABCPipelineElement]

    def __init__(
        self,
        pipeline_element_list: list[ABCPipelineElement],
        n_jobs: int = 1,
        name: str = "MolPipeline",
    ):
        """Initialize MolPipeline."""
        self._pipeline_element_list = pipeline_element_list
        self.n_jobs = n_jobs
        self.name = name

    @property
    def n_jobs(self) -> int:
        """Return the number of cores to use in transformation step."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, requested_jobs: int) -> None:
        """Set the number of cores to use in transformation step."""
        self._n_jobs = check_available_cores(requested_jobs)

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
        iter_input = x_input
        _ = y_input  # Making pylint happy
        _ = fit_params  # Making pylint happy
        for p_element in self._pipeline_element_list:
            iter_input = p_element.fit_transform(iter_input)
        return iter_input

    def _transform_single(self, input_value: Any) -> Any:
        iter_value = input_value
        for p_element in self._pipeline_element_list:  # type: ABCPipelineElement
            iter_value = p_element.transform_single(iter_value)
        return iter_value

    def transform(self, x_input: Any) -> Any:
        """Transform the input according to the sequence of provided PipelineElements."""
        last_element = self._pipeline_element_list[-1]
        if hasattr(last_element, "assemble_output"):
            return last_element.assemble_output(
                (single for single in self._transform_iterator(x_input))
            )

        return list(self._transform_iterator(x_input))

    def _finish(self) -> None:
        """Inform each pipeline element that the iterations have finished."""
        for p_element in self._pipeline_element_list:  # type: ABCPipelineElement
            p_element.finish()

    def _transform_iterator(self, x_input: Any) -> Any:
        if self.n_jobs > 1:
            with multiprocessing.Pool(self.n_jobs) as pool:
                for transformed_value in pool.imap(self._transform_single, x_input):
                    yield transformed_value
        else:
            for value in x_input:
                yield self._transform_single(value)
        self._finish()

    def copy(self) -> MolPipeline:
        """Return a copy of the MolPipeline."""
        return self[:]

    def __getitem__(self, index: slice) -> MolPipeline:
        """Get new MolPipeline with a slice of elements."""
        element_slice = self.pipeline_elements[index]
        element_slice_copy = [element.copy() for element in element_slice]
        return MolPipeline(element_slice_copy, self.n_jobs)

    def __add__(self, other: Union[ABCPipelineElement, MolPipeline]) -> MolPipeline:
        """Concatenate two Pipelines or add a PipelineElement."""
        pipeline_element_list = self.pipeline_elements[:]
        if isinstance(other, ABCPipelineElement):
            pipeline_element_list.append(other)
        elif isinstance(other, MolPipeline):
            pipeline_element_list.extend(other.pipeline_elements)
        else:
            raise TypeError(f"{type(other)} is not supported for addition!")
        return MolPipeline(pipeline_element_list, self.n_jobs)
