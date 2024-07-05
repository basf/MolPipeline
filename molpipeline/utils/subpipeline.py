"""Helper functions to extract subpipelines from a pipeline."""

from __future__ import annotations

from typing import Any, Callable

from sklearn.base import BaseEstimator

from molpipeline import FilterReinserter, Pipeline, PostPredictionWrapper
from molpipeline.abstract_pipeline_elements.core import (
    AnyToMolPipelineElement,
    MolToAnyPipelineElement,
)


def _get_molecule_reading_position_from_pipeline(pipeline: Pipeline) -> int | None:
    """Heuristic to select the position of the central molecule reading element in a pipeline.

    This function searches the last AnyToMolPipelineElement in the pipeline. We select the last
    AnyToMolPipelineElement in the pipeline because we have some standardization pipelines
    that write the molecule a smiles and read them back in to ensure they are readable.

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline to search for the molecule reading element.

    Returns
    -------
    int | None
        The position of the molecule reading element in the pipeline.
    """
    for i, step in enumerate(reversed(pipeline.steps)):
        if isinstance(step[1], AnyToMolPipelineElement):
            return len(pipeline.steps) - i - 1
    return None


def _get_model_element_position_from_pipeline(pipeline: Pipeline) -> int | None:
    """Heuristic to select the position of the machine learning estimator model in a pipeline.

    The returned element should be the element returning the pipeline's predictions. So, the
    pipeline up to this element can be executed to obtain the predictions.

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline to search for the model element.

    Returns
    -------
    int | None
        The position of the model element in the pipeline or None if no model element is found.
    """
    for i, step in enumerate(reversed(pipeline.steps)):
        if isinstance(step[1], BaseEstimator):
            if isinstance(step[1], PostPredictionWrapper):
                # skip PostPredictionWrappers.
                continue
            return len(pipeline.steps) - i - 1
    return None


def _get_featurization_element_position_from_pipeline(pipeline: Pipeline) -> int | None:
    """Heuristic to select the position of the featurization element in a pipeline.

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline to search for the featurization element.

    Returns
    -------
    int | None
        The position of the featurization element in the pipeline or None if no featurization element is found.

    """
    for i, step in enumerate(reversed(pipeline.steps)):
        if isinstance(step[1], MolToAnyPipelineElement):
            return len(pipeline.steps) - i - 1
    return None


class SubpipelineExtractor:
    """A helper class to extract parts of a pipeline."""

    def __init__(self, pipeline: Pipeline) -> None:
        """Initialize the SubpipelineExtractor.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to extract subpipelines from.
        """
        self.pipeline = pipeline

    def _get_index_of_element_by_id(self, element: Any) -> int | None:
        """Get the index of an element by id (the pointer or memory address).

        Parameters
        ----------
        element : Any
            The element to extract.

        Returns
        -------
        int | None
            The index of the element or None if the element was not found.
        """
        for i, (_, pipeline_element) in enumerate(self.pipeline.steps):
            if pipeline_element is element:
                return i
        return None

    def _get_index_of_element_by_name(self, element_name: str) -> int | None:
        """Get the index of an element by name.

        Parameters
        ----------
        element_name : str
            The name of the element to extract.

        Returns
        -------
        int | None
            The index of the element or None if the element was not found.
        """
        for i, (name, _) in enumerate(self.pipeline.steps):
            if name == element_name:
                return i
        return None

    def _extract_single_element_index(
        self,
        element_name: str | None,
        get_index_function: Callable[[Pipeline], int | None],
    ) -> int | None:
        """Extract the index of a single element from the pipeline.

        Parameters
        ----------
        element_name : str | None
            The name of the element to extract.
        get_index_function : Callable[[Pipeline], int | None]
            A function that returns the index of the element to extract.

        Returns
        -------
        Any | None
            The index of the extracted element or None if the element was not found.
        """
        if element_name is not None:
            return self._get_index_of_element_by_name(element_name)
        return get_index_function(self.pipeline)

    def _extract_single_element(
        self,
        element_name: str | None,
        get_index_function: Callable[[Pipeline], int | None],
    ) -> Any | None:
        """Extract a single element from the pipeline.

        Parameters
        ----------
        element_name : str | None
            The name of the element to extract.
        get_index_function : Callable[[Pipeline], int | None]
            A function that returns the index of the element to extract.

        Returns
        -------
        Any | None
            The extracted element or None if the element was not found.
        """
        if element_name is not None:
            # if a name is provided, access the element by name
            return self.pipeline.named_steps[element_name]
        element_index = self._extract_single_element_index(None, get_index_function)
        if element_index is None:
            return None
        return self.pipeline.steps[element_index][1]

    def get_molecule_reader_element(
        self, element_name: str | None = None
    ) -> AnyToMolPipelineElement | None:
        """Get the molecule reader element from the pipeline, e.g. a SmilesToMol element.

        Parameters
        ----------
        element_name : str | None
            The name of the element to extract.

        Returns
        -------
        AnyToMolPipelineElement | None
            The extracted molecule reader element or None if the element was not found.
        """
        return self._extract_single_element(
            element_name,
            _get_molecule_reading_position_from_pipeline,
        )

    def get_featurization_element(
        self, element_name: str | None = None
    ) -> BaseEstimator | None:
        """Get the featurization element from the pipeline, e.g., a MolToMorganFP element.

        Parameters
        ----------
        element_name : str | None
            The name of the element to extract.

        Returns
        -------
        BaseEstimator | None
            The extracted featurization element or None if the element was not found.
        """
        return self._extract_single_element(
            element_name, _get_featurization_element_position_from_pipeline
        )

    def get_model_element(
        self, element_name: str | None = None
    ) -> BaseEstimator | None:
        """Get the machine learning model element from the pipeline, e.g. a RandomForestClassifier.

        Parameters
        ----------
        element_name : str | None
            The name of the element to extract.

        Returns
        -------
        BaseEstimator | None
            The extracted model element or None if the element was not found.
        """
        return self._extract_single_element(
            element_name, _get_model_element_position_from_pipeline
        )

    def _get_subpipline_from_start(
        self,
        element_name: str | None,
        start_get_index_function: Callable[[Pipeline], int | None],
    ) -> Pipeline | None:
        """Get a subpipeline up to a specific element starting from the first element of the original pipeline.

        Parameters
        ----------
        element_name : str | None
            The name of the element to extract.
        start_get_index_function : Callable[[Pipeline], int | None]
            A function that returns the index of the subpipline's last element.

        Returns
        -------
        Pipeline | None
            The extracted subpipeline or None if the corresponding last element was not found.
        """
        element_index = self._extract_single_element_index(
            element_name, start_get_index_function
        )
        if element_index is None:
            return None
        return Pipeline(steps=self.pipeline.steps[: element_index + 1])

    def get_molecule_reader_subpipeline(
        self, element_name: str | None = None
    ) -> Pipeline | None:
        """Get a subpipeline up to the molecule reading element.

        Note that standardization steps, like salt removal, are not guaranteed to be included.

        Parameters
        ----------
        element_name : str | None
            The name of the last element in the subpipeline to extract.

        Returns
        -------
        Pipeline | None
            The extracted subpipeline or None if the corresponding last element was not found.
        """
        return self._get_subpipline_from_start(
            element_name, _get_molecule_reading_position_from_pipeline
        )

    def get_featurization_subpipeline(
        self, element_name: str | None = None
    ) -> Pipeline | None:
        """Get a subpipeline up to the featurization element.

        Parameters
        ----------
        element_name : str | None
            The name of the last element in the subpipeline to extract.

        Returns
        -------
        Pipeline | None
            The extracted subpipeline or None if the corresponding last element was not found.
        """
        return self._get_subpipline_from_start(
            element_name, _get_featurization_element_position_from_pipeline
        )

    def get_model_subpipeline(self, element_name: str | None = None) -> Pipeline | None:
        """Get a subpipeline up to the machine learning model element.

        Parameters
        ----------
        element_name : str | None
            The name of the last element in the subpipeline to extract.

        Returns
        -------
        Pipeline | None
            The extracted subpipeline or None if the corresponding last element was not found.
        """
        return self._get_subpipline_from_start(
            element_name, _get_model_element_position_from_pipeline
        )

    def get_subpipeline(
        self,
        first_element: Any,
        second_element: Any,
        first_offset: int = 0,
        second_offset: int = 0,
    ) -> Pipeline | None:
        """Get a subpipeline between two elements.

        This function only checks the names of the elements.
        If the elements are not found or the second element is before the first element, a ValueError is raised.

        Parameters
        ----------
        first_element : Any
            The first element of the subpipeline.
        second_element : Any
            The second element of the subpipeline.
        first_offset : int
            The offset to apply to the first element.
        second_offset : int
            The offset to apply to the second element.

        Returns
        -------
        Pipeline | None
            The extracted subpipeline or None if the elements were not found.
        """
        first_element_index = self._get_index_of_element_by_id(first_element)
        if first_element_index is None:
            raise ValueError(f"Element {first_element} not found in pipeline.")
        second_element_index = self._get_index_of_element_by_id(second_element)
        if second_element_index is None:
            raise ValueError(f"Element {second_element} not found in pipeline.")

        # apply user-defined offsets
        first_element_index += first_offset
        second_element_index += second_offset

        if second_element_index < first_element_index:
            raise ValueError(
                f"Element {second_element} must be after element {first_element}."
            )
        return Pipeline(
            steps=self.pipeline.steps[first_element_index : second_element_index + 1]
        )

    def get_all_filter_reinserter_fill_values(self) -> list[Any]:
        """Get all fill values for FilterReinserter elements in the pipeline.

        Returns
        -------
        list[Any]
            The fill values for all FilterReinserter elements in the pipeline.
        """
        fill_values = set()
        for _, step in self.pipeline.steps:
            if isinstance(step, FilterReinserter):
                fill_values.add(step.fill_value)
            if isinstance(step, PostPredictionWrapper) and isinstance(
                step.wrapped_estimator, FilterReinserter
            ):
                fill_values.add(step.wrapped_estimator.fill_value)
        return list(fill_values)
