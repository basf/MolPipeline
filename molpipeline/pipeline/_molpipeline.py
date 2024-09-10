"""Defines the pipeline which handles pipeline elements for molecular operations."""

from __future__ import annotations

from typing import Any, Iterable, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import numpy as np
from joblib import Parallel, delayed
from rdkit.Chem.rdchem import MolSanitizeException
from rdkit.rdBase import BlockLogs

from molpipeline.abstract_pipeline_elements.core import (
    ABCPipelineElement,
    InvalidInstance,
    RemovedInstance,
    TransformingPipelineElement,
)
from molpipeline.error_handling import (
    ErrorFilter,
    FilterReinserter,
    _MultipleErrorFilter,
)
from molpipeline.utils.molpipeline_types import NumberIterable
from molpipeline.utils.multi_proc import check_available_cores


class _MolPipeline:
    """Contains the PipeElements which describe the functionality of the pipeline."""

    _n_jobs: int
    _element_list: list[ABCPipelineElement]
    _requires_fitting: bool

    def __init__(
        self,
        element_list: list[ABCPipelineElement],
        n_jobs: int = 1,
        name: str = "MolPipeline",
        raise_nones: bool = False,
    ) -> None:
        """Initialize MolPipeline.

        Parameters
        ----------
        element_list: list[ABCPipelineElement]
            List of Pipeline Elements which form the pipeline.
        n_jobs:
            Number of cores used.
        name: str
            Name of pipeline.
        raise_nones: bool
            If True, raise an error if a None is encountered in the pipeline.

        Returns
        -------
        None
        """
        self._element_list = element_list
        self.n_jobs = n_jobs
        self.name = name
        self._requires_fitting = any(
            element.requires_fitting for element in self._element_list
        )
        self.raise_nones = raise_nones

    @property
    def _filter_elements(self) -> list[ErrorFilter]:
        """Get the elements which filter the input."""
        return [
            element
            for element in self._element_list
            if isinstance(element, ErrorFilter)
        ]

    @property
    def _filter_elements_agg(self) -> _MultipleErrorFilter:
        """Get the aggregated filter element."""
        return _MultipleErrorFilter(self._filter_elements)

    @property
    def _transforming_elements(
        self,
    ) -> list[Union[TransformingPipelineElement, _MolPipeline]]:
        """Get the elements which transform the input."""
        return [
            element
            for element in self._element_list
            if isinstance(element, (TransformingPipelineElement, _MolPipeline))
        ]

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
        self.set_params(**parameter_dict)

    @property
    def requires_fitting(self) -> bool:
        """Return whether the pipeline requires fitting."""
        return self._requires_fitting

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
                "raise_nones": self.raise_nones,
            }
        return {
            "element_list": self._element_list,
            "n_jobs": self.n_jobs,
            "name": self.name,
            "raise_nones": self.raise_nones,
        }

    def set_params(self, **parameter_dict: Any) -> Self:
        """Set parameters of the pipeline and pipeline elements.

        Parameters
        ----------
        parameter_dict: Any
            Dictionary containing the parameter names and corresponding values to be set.

        Returns
        -------
        Self
            MolPipeline object with updated parameters.
        """
        if "element_list" in parameter_dict:
            self._element_list = parameter_dict["element_list"]
        if "n_jobs" in parameter_dict:
            self.n_jobs = int(parameter_dict["n_jobs"])
        if "name" in parameter_dict:
            self.name = str(parameter_dict["name"])
        if "raise_nones" in parameter_dict:
            self.raise_nones = bool(parameter_dict["raise_nones"])
        return self

    @property
    def element_list(self) -> list[ABCPipelineElement]:
        """Get a shallow copy from the list of pipeline elements."""
        return self._element_list[:]  # [:] to create shallow copy.

    def _get_meta_element_list(
        self,
    ) -> list[Union[ABCPipelineElement, _MolPipeline]]:
        """Merge elements which do not require fitting to a meta element which improves parallelization.

        Returns
        -------
        list[Union[ABCPipelineElement, _MolPipeline]]
            List of pipeline elements and meta elements.
        """
        meta_element_list: list[Union[ABCPipelineElement, _MolPipeline]] = []
        no_fit_element_list: list[ABCPipelineElement] = []
        for element in self._element_list:
            if (
                isinstance(element, TransformingPipelineElement)
                and not element.requires_fitting
            ):
                no_fit_element_list.append(element)
            else:
                if len(no_fit_element_list) == 1:
                    meta_element_list.append(no_fit_element_list[0])
                elif len(no_fit_element_list) > 1:
                    meta_element_list.append(
                        _MolPipeline(no_fit_element_list, n_jobs=self.n_jobs)
                    )
                no_fit_element_list = []
                meta_element_list.append(element)
        if len(no_fit_element_list) == 1:
            meta_element_list.append(no_fit_element_list[0])
        elif len(no_fit_element_list) > 1:
            meta_element_list.append(
                _MolPipeline(no_fit_element_list, n_jobs=self.n_jobs)
            )
        return meta_element_list

    def fit(
        self,
        x_input: Any,
        y: Any = None,  # pylint: disable=invalid-name
        **fit_params: dict[Any, Any],
    ) -> Self:
        """Fit the MolPipeline according to x_input.

        Parameters
        ----------
        x_input: Any
            Molecular representations which are subsequently processed.
        y: Any
            Optional label of input. Only for SKlearn compatibility.
        fit_params: Any
            Parameters. Only for SKlearn compatibility.

        Returns
        -------
        Self
            Fitted MolPipeline.
        """
        _ = y  # Making pylint happy
        _ = fit_params  # Making pylint happy
        if self.requires_fitting:
            self.fit_transform(x_input)
        return self

    def fit_transform(
        self,
        x_input: Any,
        y: Any = None,  # pylint: disable=invalid-name
        **fit_params: dict[str, Any],
    ) -> Any:
        """Fit the MolPipeline according to x_input and return the transformed molecules.

        Parameters
        ----------
        x_input: Any
            Molecular representations which are subsequently processed.
        y: Any
            Optional label of input. Only for SKlearn compatibility.
        fit_params: Any
            Parameters. Only for SKlearn compatibility.

        Returns
        -------
        Any
            Transformed molecules.
        """
        iter_input = x_input
        _ = y  # Making pylint happy, does no(t a)thing
        _ = fit_params  # Making pylint happy

        # The meta elements merge steps which do not require fitting to improve parallelization
        iter_element_list = self._get_meta_element_list()
        removed_rows: dict[ErrorFilter, list[int]] = {}
        for error_filter in self._filter_elements:
            removed_rows[error_filter] = []
        iter_idx_array = np.arange(len(iter_input))
        for i_element in iter_element_list:
            if not isinstance(i_element, (TransformingPipelineElement, _MolPipeline)):
                continue
            i_element.n_jobs = self.n_jobs
            iter_input = i_element.pretransform(iter_input)
            for error_filter in self._filter_elements:
                iter_input = error_filter.transform(iter_input)
                for idx in error_filter.error_indices:
                    idx = iter_idx_array[idx]
                    removed_rows[error_filter].append(idx)
                iter_idx_array = error_filter.co_transform(iter_idx_array)
            if i_element.requires_fitting:
                if isinstance(i_element, _MolPipeline):
                    raise AssertionError("No subpipline should require fitting!")
                i_element.fit_to_result(iter_input)
            if isinstance(i_element, TransformingPipelineElement):
                iter_input = i_element.finalize_list(iter_input)
            iter_input = i_element.assemble_output(iter_input)
            i_element.n_jobs = 1

        # Set removed rows to filter elements to allow for correct co_transform
        iter_idx_array = np.arange(len(x_input))
        for error_filter in self._filter_elements:
            removed_idx_list = removed_rows[error_filter]
            error_filter.error_indices = []
            for new_idx, idx in enumerate(iter_idx_array):
                if idx in removed_idx_list:
                    error_filter.error_indices.append(new_idx)
            error_filter.n_total = len(iter_idx_array)
            iter_idx_array = error_filter.co_transform(iter_idx_array)
        error_replacer_list = [
            ele for ele in self._element_list if isinstance(ele, FilterReinserter)
        ]
        for error_replacer in error_replacer_list:
            error_replacer.select_error_filter(self._filter_elements)
            iter_input = error_replacer.transform(iter_input)
        return iter_input

    def transform_single(self, input_value: Any) -> Any:
        """Transform a single input according to the sequence of provided PipelineElements.

        Parameters
        ----------
        input_value: Any
            Molecular representation which is subsequently transformed.

        Returns
        -------
        Any
            Transformed molecular representation.
        """
        log_block = BlockLogs()
        iter_value = input_value
        for p_element in self._element_list:
            try:
                if not isinstance(iter_value, RemovedInstance):
                    iter_value = p_element.transform_single(iter_value)
                elif isinstance(p_element, FilterReinserter):
                    iter_value = p_element.transform_single(iter_value)
            except MolSanitizeException as err:
                iter_value = InvalidInstance(
                    p_element.uuid,
                    f"RDKit MolSanitizeException: {err.args}",
                    p_element.name,
                )
        del log_block
        return iter_value

    def pretransform(self, x_input: Any) -> Any:
        """Transform the input according to the sequence BUT skip the assemble output step.

        Parameters
        ----------
        x_input: Any
            Molecular representations which are subsequently transformed.

        Returns
        -------
        Any
            Transformed molecular representations.
        """
        return list(self._transform_iterator(x_input))

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
        output_generator = self._transform_iterator(x_input)
        return self.assemble_output(output_generator)

    def assemble_output(self, value_list: Iterable[Any]) -> Any:
        """Assemble the output of the pipeline.

        Parameters
        ----------
        value_list: Iterable[Any]
            Generator which yields the output of the pipeline.

        Returns
        -------
        Any
            Assembled output.
        """
        last_element = self._transforming_elements[-1]
        if hasattr(last_element, "assemble_output"):
            return last_element.assemble_output(value_list)
        return list(value_list)

    def _finish(self) -> None:
        """Inform each pipeline element that the iterations have finished."""
        for p_element in self._element_list:
            p_element.finish()

    def _transform_iterator(self, x_input: Any) -> Any:
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
        agg_filter = self._filter_elements_agg
        for filter_element in self._filter_elements:
            filter_element.error_indices = []
        parallel = Parallel(
            n_jobs=self.n_jobs,
            return_as="generator",
            batch_size="auto",
        )
        output_generator = parallel(
            delayed(self.transform_single)(value) for value in x_input
        )
        for i, transformed_value in enumerate(output_generator):
            if isinstance(transformed_value, RemovedInstance):
                agg_filter.register_removed(i, transformed_value)
            else:
                yield transformed_value
        agg_filter.set_total(len(x_input))
        self._finish()

    def co_transform(self, x_input: NumberIterable) -> NumberIterable:
        """Filter flagged rows from the input.

        Parameters
        ----------
        x_input: Any
            Molecular representations which are subsequently filtered.

        Returns
        -------
        Any
            Filtered molecular representations.
        """
        return self._filter_elements_agg.co_transform(x_input)
