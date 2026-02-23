"""Defines a pipeline is exposed to the user, accessible via pipeline."""

# pylint: disable=too-many-lines

from collections.abc import Generator, Iterable, Sequence
from copy import deepcopy
from itertools import islice
from typing import Any, Literal, Self, TypeIs

import joblib
import numpy as np
from joblib import Parallel, delayed
from rdkit.Chem.rdchem import MolSanitizeException
from rdkit.rdBase import BlockLogs
from sklearn.base import _fit_context  # noqa: PLC2701
from sklearn.utils._tags import Tags, get_tags  # noqa: PLC2701
from sklearn.utils.metaestimators import available_if

from molpipeline.abstract_pipeline_elements.core import (
    ABCPipelineElement,
    InvalidInstance,
    RemovedInstance,
    SingleInstanceTransformerMixin,
    TransformingPipelineElement,
)
from molpipeline.error_handling import (
    ErrorFilter,
    FilterReinserter,
    _MultipleErrorFilter,
)
from molpipeline.pipeline._skl_adapter_pipeline import AdapterPipeline
from molpipeline.utils.molpipeline_types import (
    AnyElement,
    AnyPredictor,
    AnyStep,
    AnyTransformer,
    TypeFixedVarSeq,
)

__all__ = ["Pipeline"]


_IndexedStep = tuple[int, str, AnyElement]
_AggStep = tuple[list[int], list[str], "Pipeline"]


class Pipeline(AdapterPipeline, TransformingPipelineElement):  # pylint: disable=too-many-ancestors
    """Defines the pipeline which handles pipeline elements."""

    steps: list[AnyStep]

    @property
    def _filter_elements(self) -> list[ErrorFilter]:
        """Get the elements which filter the input."""
        return [step[1] for step in self.steps if isinstance(step[1], ErrorFilter)]

    @property
    def _filter_elements_agg(self) -> _MultipleErrorFilter:
        """Get the aggregated filter element."""
        return _MultipleErrorFilter(self._filter_elements)

    @property
    def supports_single_instance(self) -> bool:
        """Check if the pipeline supports single instance."""
        return all(
            isinstance(step[1], SingleInstanceTransformerMixin)
            for step in self._modified_steps
        )

    def __init__(
        self,
        steps: list[AnyStep],
        *,
        memory: str | joblib.Memory | None = None,
        verbose: bool = False,
        n_jobs: int = 1,
    ) -> None:
        """Initialize Pipeline.

        Parameters
        ----------
        steps: list[tuple[str, AnyTransformer | AnyPredictor | ABCPipelineElement]]
            List of (name, Estimator) tuples.
        memory: str | joblib.Memory | None, optional
            Path to cache transformers.
        verbose: bool, optional
            If True, print additional information.
        n_jobs: int, optional
            Number of cores used for aggregated steps.

        """
        super().__init__(steps, memory=memory, verbose=verbose)
        self.n_jobs = n_jobs
        self._set_error_resinserter()

    def _iter(
        self,
        with_final: bool = True,
        filter_passthrough: bool = True,
    ) -> Generator[
        tuple[list[int], list[str], "Pipeline"] | tuple[int, str, AnyElement],
        Any,
        None,
    ]:
        """Iterate over all non post-processing steps.

        Steps where children of a ABCPipelineElement are aggregated to a MolPipeline.

        Parameters
        ----------
        with_final: bool, optional
            If True, the final estimator is included.
        filter_passthrough: bool, optional
            If True, passthrough steps are filtered out.

        Yields
        ------
        _AggregatedPipelineStep
            The _AggregatedPipelineStep is composed of the index, the name and the
            transformer.

        """
        if self.supports_single_instance:
            yield from super()._iter(
                with_final=with_final,
                filter_passthrough=filter_passthrough,
            )
            return
        transformers_to_agg: list[tuple[int, str, AnyTransformer]]
        transformers_to_agg = []
        final_transformer_list: list[
            tuple[list[int], list[str], Pipeline] | tuple[int, str, AnyElement]
        ] = []
        for i, (name_i, step_i) in enumerate(super()._modified_steps):
            if isinstance(step_i, SingleInstanceTransformerMixin):
                transformers_to_agg.append((i, name_i, step_i))
            else:
                if transformers_to_agg:
                    if len(transformers_to_agg) == 1:
                        final_transformer_list.append(transformers_to_agg[0])
                    else:
                        final_transformer_list.append(
                            _agg_transformers(transformers_to_agg, self.n_jobs),
                        )
                    transformers_to_agg = []
                final_transformer_list.append((i, name_i, step_i))

        # yield last step if anything remains
        if transformers_to_agg:
            final_transformer_list.append(
                _agg_transformers(transformers_to_agg, self.n_jobs),
            )
        stop = len(final_transformer_list)
        if not with_final:
            stop -= 1

        for step in islice(final_transformer_list, 0, stop):
            if not filter_passthrough or (
                step[2] is not None and step[2] != "passthrough"
            ):
                yield step

    @property
    def _final_estimator(
        self,
    ) -> Literal["passthrough"] | AnyTransformer | AnyPredictor | ABCPipelineElement:
        """Return the lst estimator which is not a PostprocessingTransformer."""
        element_list = list(self._iter(with_final=True))
        steps = [s for s in element_list if not isinstance(s[2], ErrorFilter)]
        last_element = steps[-1]
        return last_element[2]

    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False,
    )
    def fit(
        self,
        X: Any,  # noqa: N803
        y: Any = None,
        **fit_params: Any,
    ) -> Self:
        """Fit the model.

        Fit all the transformers one after the other and transform the
        data. Finally, fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : object
            Pipeline with fitted steps.

        """
        if self.supports_single_instance:
            return self
        super().fit(X, y, **fit_params)
        return self

    # pylint: disable=duplicate-code
    def _can_fit_transform(self) -> bool:
        """Check if the final estimator can fit_transform or is passthrough.

        Returns
        -------
        bool
            True if the final estimator can fit_transform or is passthrough.

        """
        return (
            self._final_estimator == "passthrough"
            or hasattr(self._final_estimator, "transform")
            or hasattr(self._final_estimator, "fit_transform")
        )

    # pylint: enable=duplicate-code

    @available_if(_can_fit_transform)
    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False,
    )
    def fit_transform(
        self,
        X: Any,  # noqa: N803
        y: Any = None,
        **params: Any,
    ) -> Any:
        """Fit the model and transform with the final estimator.

        Fits all the transformers one after the other and transform the
        data. Then uses `fit_transform` on transformed data with the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **params : Any
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Raises
        ------
        AssertionError
            If PipelineElement is not a SingleInstanceTransformerMixin or Pipeline.
        AssertionError
            If PipelineElement is not a TransformingPipelineElement.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.

        """
        if not self.supports_single_instance:
            return super().fit_transform(X, **params)
        iter_input = X
        removed_row_dict: dict[ErrorFilter, list[int]] = {
            error_filter: [] for error_filter in self._filter_elements
        }
        iter_idx_array = np.arange(len(iter_input))

        # The meta elements merge steps which do not require fitting
        for idx, _, i_element in self._iter(with_final=True):
            if not isinstance(i_element, TransformingPipelineElement):
                raise AssertionError(
                    "PipelineElement is not a TransformingPipelineElement.",
                )
            if not check_single_instance_support(i_element):
                raise AssertionError(
                    "PipelineElement is not a SingleInstanceTransformerMixin or"
                    " Pipeline with signle instance support.",
                )
            if isinstance(i_element, ErrorFilter):
                continue
            i_element.n_jobs = self.n_jobs
            iter_input = i_element.pretransform(iter_input)
            for error_filter in self._filter_elements:
                iter_input = error_filter.transform(iter_input)
                for idx in error_filter.error_indices:
                    removed_row_dict[error_filter].append(int(iter_idx_array[idx]))
                iter_idx_array = error_filter.co_transform(iter_idx_array)
            iter_input = i_element.assemble_output(iter_input)
            i_element.n_jobs = 1

        # Set removed rows to filter elements to allow for correct co_transform
        iter_idx_array = np.arange(len(X))
        for error_filter in self._filter_elements:
            removed_idx_list = removed_row_dict[error_filter]
            error_filter.error_indices = []
            for new_idx, idx_ in enumerate(iter_idx_array):
                if idx_ in removed_idx_list:
                    error_filter.error_indices.append(new_idx)
            error_filter.n_total = len(iter_idx_array)
            iter_idx_array = error_filter.co_transform(iter_idx_array)
        error_replacer_list = [
            ele for _, ele in self.steps if isinstance(ele, FilterReinserter)
        ]
        for error_replacer in error_replacer_list:
            error_replacer.select_error_filter(self._filter_elements)
            iter_input = error_replacer.transform(iter_input)
        for _, post_element in self._post_processing_steps:
            iter_input = post_element.fit_transform(iter_input, y)
        return iter_input

    def _can_transform(self) -> bool:
        """Check if the final estimator can transform or is passthrough.

        Returns
        -------
        bool
            True if the final estimator can transform or is passthrough.

        """
        return self._final_estimator == "passthrough" or hasattr(
            self._final_estimator,
            "transform",
        )

    @available_if(_can_transform)
    def transform(
        self,
        X: Any,  # noqa: N803
        **params: Any,
    ) -> Any:
        """Transform the data, and apply `transform` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `transform` method. Only valid if the final estimator
        implements `transform`.

        This also works where final estimator is `None` in which case all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.
        **params : Any
            Parameters to the ``transform`` method of each estimator.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.

        """
        if self.supports_single_instance:
            output_generator = self._transform_iterator(X)
            iter_input = self.assemble_output(output_generator)
        else:
            iter_input = super().transform(X, **params)
        for _, post_element in self._post_processing_steps:
            iter_input = post_element.transform(iter_input, **params)
        return iter_input

    def _can_decision_function(self) -> bool:
        """Check if the final estimator implements decision_function.

        Returns
        -------
        bool
            True if the final estimator implements decision_function.

        """
        return hasattr(self._final_estimator, "decision_function")

    @available_if(_can_decision_function)
    def decision_function(
        self,
        X: Any,  # noqa: N803
        **params: Any,
    ) -> Any:
        """Transform the data, and apply `decision_function` with the final estimator.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.
        **params : Any
            Parameters to the ``decision_function`` method of the final estimator.

        Returns
        -------
        Any
            Result of calling `decision_function` on the final estimator.

        """
        iter_input = super().decision_function(X, **params)
        for _, post_element in self._post_processing_steps:
            iter_input = post_element.transform(iter_input)
        return iter_input

    def __sklearn_tags__(self) -> Tags:  # noqa: PLW3201
        """Return the sklearn tags.

        Notes
        -----
        This method is copied from the original sklearn implementation.
        Changes are marked with a comment.

        Returns
        -------
        Tags
            The sklearn tags.

        """
        tags = super().__sklearn_tags__()

        if not self.steps:
            return tags

        try:
            if self.steps[0][1] is not None and self.steps[0][1] != "passthrough":
                tags.input_tags.pairwise = get_tags(
                    self.steps[0][1],
                ).input_tags.pairwise
            # WARNING: the sparse tag can be incorrect.
            # Some Pipelines accepting sparse data are wrongly tagged sparse=False.
            # For example Pipeline([PCA(), estimator]) accepts sparse data
            # even if the estimator doesn't as PCA outputs a dense array.
            tags.input_tags.sparse = all(
                get_tags(step).input_tags.sparse
                for name, step in self.steps
                if step != "passthrough"
            )
        except (ValueError, AttributeError, TypeError):
            # This happens when the `steps` is not a list of (name, estimator)
            # tuples and `fit` is not called yet to validate the steps.
            pass

        try:
            # Only the _final_estimator is changed from the original implementation is
            # changed in the following 2 lines
            if (
                self._final_estimator is not None
                and self._final_estimator != "passthrough"
            ):
                last_step_tags = get_tags(self._final_estimator)
                tags.estimator_type = last_step_tags.estimator_type
                tags.target_tags.multi_output = last_step_tags.target_tags.multi_output
                tags.classifier_tags = deepcopy(last_step_tags.classifier_tags)
                tags.regressor_tags = deepcopy(last_step_tags.regressor_tags)
                tags.transformer_tags = deepcopy(last_step_tags.transformer_tags)
        except (ValueError, AttributeError, TypeError):
            # This happens when the `steps` is not a list of (name, estimator)
            # tuples and `fit` is not called yet to validate the steps.
            pass

        return tags

    def transform_single(self, input_value: Any) -> Any:
        """Transform a single input according to the sequence of PipelineElements.

        Parameters
        ----------
        input_value: Any
            Molecular representation which is subsequently transformed.

        Raises
        ------
        AssertionError
            If the PipelineElement is not a SingleInstanceTransformerMixin.

        Returns
        -------
        Any
            Transformed molecular representation.

        """
        log_block = BlockLogs()
        iter_value = input_value
        for _, p_element in self._modified_steps:
            if not isinstance(p_element, SingleInstanceTransformerMixin):
                raise AssertionError(
                    "PipelineElement is not a SingleInstanceTransformerMixin.",
                )
            if not isinstance(p_element, ABCPipelineElement):
                raise AssertionError(
                    "PipelineElement is not a ABCPipelineElement.",
                )
            try:
                if not isinstance(iter_value, RemovedInstance) or isinstance(
                    p_element,
                    FilterReinserter,
                ):
                    iter_value = p_element.transform_single(iter_value)
            except MolSanitizeException as err:
                iter_value = InvalidInstance(
                    p_element.uuid,
                    f"RDKit MolSanitizeException: {err.args}",
                    p_element.name,
                )
        del log_block
        return iter_value

    def pretransform(self, value_list: Any) -> Any:
        """Transform the input according to the sequence without assemble_output step.

        Parameters
        ----------
        value_list: Any
            Molecular representations which are subsequently transformed.

        Returns
        -------
        Any
            Transformed molecular representations.

        """
        return list(self._transform_iterator(value_list))

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
        final_estimator = self._final_estimator
        if isinstance(final_estimator, TransformingPipelineElement):
            return final_estimator.assemble_output(value_list)
        return list(value_list)

    def _transform_iterator(self, x_input: Any) -> Any:
        """Transform the input according to the sequence of provided PipelineElements.

        Parameters
        ----------
        x_input: Any
            Molecular representations which are subsequently transformed.

        Yields
        ------
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

    def co_transform(self, x_input: TypeFixedVarSeq) -> TypeFixedVarSeq:
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


def _agg_transformers(
    transformer_list: Sequence[tuple[int, str, AnyElement]],
    n_jobs: int = 1,
) -> tuple[list[int], list[str], Pipeline] | tuple[int, str, AnyElement]:
    """Aggregate transformers to a single step.

    Parameters
    ----------
    transformer_list: list[tuple[int, str, AnyElement]]
        List of transformers to aggregate.
    n_jobs: int, optional
        Number of cores used for aggregated steps.

    Returns
    -------
    tuple[list[int], list[str], Pipeline] | tuple[int, str, AnyElement]
        Aggregated transformer.
        If the list contains only one transformer, it is returned as is.

    """
    index_list = [step[0] for step in transformer_list]
    name_list = [step[1] for step in transformer_list]
    if len(transformer_list) == 1:
        return transformer_list[0]
    return (
        index_list,
        name_list,
        Pipeline(
            [(step[1], step[2]) for step in transformer_list],
            n_jobs=n_jobs,
        ),
    )


def check_single_instance_support(
    estimator: Any,
) -> TypeIs[SingleInstanceTransformerMixin | Pipeline]:
    """Check if the estimator supports single instance processing.

    Parameters
    ----------
    estimator: Any
        Estimator to check.

    Returns
    -------
    TypeIs[SingleInstanceTransformerMixin | Pipeline]
        True if the estimator supports single instance processing.

    """
    if isinstance(estimator, SingleInstanceTransformerMixin):
        return True
    return isinstance(estimator, Pipeline) and estimator.supports_single_instance
