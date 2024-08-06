"""Defines a pipeline is exposed to the user, accessible via pipeline."""

from __future__ import annotations

from typing import Any, Iterable, List, Literal, Optional, Tuple, TypeVar, Union

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import joblib
import numpy as np
import numpy.typing as npt
from loguru import logger
from sklearn.base import _fit_context  # pylint: disable=protected-access
from sklearn.base import clone
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.pipeline import _final_estimator_has, _fit_transform_one
from sklearn.utils import Bunch
from sklearn.utils.metadata_routing import (
    _routing_enabled,  # pylint: disable=protected-access
)
from sklearn.utils.metadata_routing import process_routing
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_memory

from molpipeline.abstract_pipeline_elements.core import ABCPipelineElement
from molpipeline.error_handling import ErrorFilter, FilterReinserter
from molpipeline.pipeline._molpipeline import _MolPipeline
from molpipeline.post_prediction import (
    PostPredictionTransformation,
    PostPredictionWrapper,
)
from molpipeline.utils.logging import print_elapsed_time
from molpipeline.utils.molpipeline_types import (
    AnyElement,
    AnyPredictor,
    AnyStep,
    AnyTransformer,
)
from molpipeline.utils.value_checks import is_empty

__all__ = ["Pipeline"]

# Type definitions
_T = TypeVar("_T")
# Cannot be moved to utils.molpipeline_types due to circular imports


_IndexedStep = Tuple[int, str, AnyElement]
_AggStep = Tuple[List[int], List[str], _MolPipeline]
_AggregatedPipelineStep = Union[_IndexedStep, _AggStep]


class Pipeline(_Pipeline):
    """Defines the pipeline which handles pipeline elements."""

    steps: list[AnyStep]
    #  * Adapted methods from sklearn.pipeline.Pipeline *

    def __init__(
        self,
        steps: list[AnyStep],
        *,
        memory: Optional[Union[str, joblib.Memory]] = None,
        verbose: bool = False,
        n_jobs: int = 1,
    ):
        """Initialize Pipeline.

        Parameters
        ----------
        steps: list[tuple[str, Union[AnyTransformer, AnyPredictor, ABCPipelineElement]]]
            List of (name, Estimator) tuples.
        memory: str, optional
            Path to cache transformers.
        verbose: bool, optional
            If True, print additional information.
        n_jobs: int, optional
            Number of cores used for aggregated steps.
        """
        super().__init__(steps, memory=memory, verbose=verbose)
        self.n_jobs = n_jobs
        self._set_error_resinserter()

    def _set_error_resinserter(self) -> None:
        """Connect the error resinserters with the error filters."""
        error_replacer_list = [
            e_filler
            for _, e_filler in self.steps
            if isinstance(e_filler, FilterReinserter)
        ]
        error_filter_list = [
            n_filter for _, n_filter in self.steps if isinstance(n_filter, ErrorFilter)
        ]
        for step in self.steps:
            if isinstance(step[1], PostPredictionWrapper):
                if isinstance(step[1].wrapped_estimator, FilterReinserter):
                    error_replacer_list.append(step[1].wrapped_estimator)
        for error_replacer in error_replacer_list:
            error_replacer.select_error_filter(error_filter_list)

    def _validate_steps(self) -> None:
        """Validate the steps."""
        names = [name for name, _ in self.steps]

        # validate names
        self._validate_names(names)

        # validate estimators
        non_post_processing_steps = [e for _, _, e in self._agg_non_postpred_steps()]
        transformer_list = non_post_processing_steps[:-1]
        estimator = non_post_processing_steps[-1]

        for transformer in transformer_list:
            if transformer is None or transformer == "passthrough":
                continue
            if not (
                hasattr(transformer, "fit") or hasattr(transformer, "fit_transform")
            ) or not hasattr(transformer, "transform"):
                raise TypeError(
                    f"All intermediate steps should be "
                    f"transformers and implement fit and transform "
                    f"or be the string 'passthrough' "
                    f"'{transformer}' (type {type(transformer)}) doesn't"
                )

        # We allow last estimator to be None as an identity transformation
        if (
            estimator is not None
            and estimator != "passthrough"
            and not hasattr(estimator, "fit")
        ):
            raise TypeError(
                f"Last step of Pipeline should implement fit "
                f"or be the string 'passthrough'. "
                f"'{estimator}' (type {type(estimator)}) doesn't"
            )

        # validate post-processing steps
        # Calling steps automatically validates them
        _ = self._post_processing_steps()

    def _iter(
        self, with_final: bool = True, filter_passthrough: bool = True
    ) -> Iterable[_AggregatedPipelineStep]:
        """Iterate over all non post-processing steps.

        Steps which are children of a ABCPipelineElement were aggregated to a MolPipeline.

        Parameters
        ----------
        with_final: bool, optional
            If True, the final estimator is included.
        filter_passthrough: bool, optional
            If True, passthrough steps are filtered out.

        Returns
        ------
        Iterable[_AggregatedPipelineStep]
            The _AggregatedPipelineStep is composed of the index, the name and the transformer.
        """
        last_element: Optional[_AggregatedPipelineStep] = None

        # This loop delays the output by one in order to identify the last step
        for step in self._agg_non_postpred_steps():
            # Only happens for the first step
            if last_element is None:
                last_element = step
                continue
            if not filter_passthrough:
                yield last_element
            elif step[2] is not None and step[2] != "passthrough":
                yield last_element
            last_element = step

        # This can only happen if no steps are set.
        if last_element is None:
            raise AssertionError("Pipeline needs to have at least one step!")

        if with_final and last_element[2] is not None:
            if last_element[2] != "passthrough":
                yield last_element

    @property
    def _estimator_type(self) -> Any:
        """Return the estimator type."""
        if self._final_estimator is None or self._final_estimator == "passthrough":
            return None
        if hasattr(self._final_estimator, "_estimator_type"):
            # pylint: disable=protected-access
            return self._final_estimator._estimator_type
        return None

    @property
    def _final_estimator(
        self,
    ) -> Union[
        Literal["passthrough"],
        AnyTransformer,
        AnyPredictor,
        _MolPipeline,
        ABCPipelineElement,
    ]:
        """Return the lst estimator which is not a PostprocessingTransformer."""
        element_list = list(self._agg_non_postpred_steps())
        last_element = element_list[-1]
        return last_element[2]

    # pylint: disable=too-many-locals,too-many-branches
    def _fit(
        self,
        X: Any,  # pylint: disable=invalid-name
        y: Any = None,  # pylint: disable=invalid-name
        routed_params: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """Fit the model by fitting all transformers except the final estimator.

        Data can be subsetted by the transformers.

        Parameters
        ----------
        X : Any
            Training data.
        y : Any, optional (default=None)
            Training objectives.
        routed_params : dict[str, Any], optional
            Parameters for each step as returned by process_routing.

        Returns
        -------
        tuple[Any, Any]
            The transformed data and the transformed objectives.
        """
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()

        # Set up the memory
        memory: joblib.Memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)
        for step in self._iter(with_final=False, filter_passthrough=False):
            step_idx, name, transformer = step
            if transformer is None or transformer == "passthrough":
                with print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)
            if isinstance(cloned_transformer, _MolPipeline):
                if routed_params:
                    fit_parameter = {
                        "element_parameters": [routed_params[n] for n in name]
                    }
                else:
                    fit_parameter = {}
            elif isinstance(name, list):
                raise AssertionError()
            else:
                if routed_params:
                    fit_parameter = routed_params[name]
                else:
                    fit_parameter = {}

            # Fit or load from cache the current transformer
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                params=fit_parameter,
            )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            if isinstance(fitted_transformer, _MolPipeline):
                ele_list = fitted_transformer.element_list
                if not isinstance(name, list) or not isinstance(step_idx, list):
                    raise AssertionError()
                if not len(name) == len(step_idx) == len(ele_list):
                    raise AssertionError()
                for idx_i, name_i, ele_i in zip(step_idx, name, ele_list):
                    self.steps[idx_i] = (name_i, ele_i)
                if y is not None:
                    y = fitted_transformer.co_transform(y)
                for idx_i, name_i, ele_i in zip(step_idx, name, ele_list):
                    self.steps[idx_i] = (name_i, ele_i)
                self._set_error_resinserter()
            elif isinstance(name, list) or isinstance(step_idx, list):
                raise AssertionError()
            else:
                self.steps[step_idx] = (name, fitted_transformer)
            if is_empty(X):
                return np.array([]), np.array([])
        return X, y

    def _transform(
        self, X: Any, routed_params: Bunch  # pylint: disable=invalid-name
    ) -> Any:
        """Transform the data, and skip final estimator.

        Call `transform` of each transformer in the pipeline except the last one,

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        routed_params: Bunch
            parameters for each step as returned by process_routing

        Returns
        -------
        Any
            Result of calling `transform` on the second last estimator.
        """
        iter_input = X
        do_routing = _routing_enabled()
        if do_routing:
            logger.warning("Routing is enabled and NOT fully tested!")

        for _, name, transform in self._iter(with_final=False):
            if is_empty(iter_input):
                if isinstance(transform, _MolPipeline):
                    _ = transform.transform(iter_input)
                iter_input = []
                break
            if transform == "passthrough":
                raise AssertionError("Passthrough should have been filtered out.")
            if hasattr(transform, "transform"):
                if do_routing:
                    iter_input = transform.transform(  # type: ignore[call-arg]
                        iter_input, routed_params[name].transform
                    )
                else:
                    iter_input = transform.transform(iter_input)
            else:
                raise AssertionError(
                    f"Non transformer ocurred in transformation step: {transform}."
                )
        return iter_input

    # * New implemented methods *
    def _non_post_processing_steps(
        self,
    ) -> list[AnyStep]:
        """Return all steps before the first PostPredictionTransformation.

        Returns
        -------
        list[AnyStep]
            List of steps before the first PostPredictionTransformation.
        """
        non_post_processing_steps: list[AnyStep] = []
        start_adding = False
        for step_name, step_estimator in self.steps[::-1]:
            if not isinstance(step_estimator, PostPredictionTransformation):
                start_adding = True
            if start_adding:
                if isinstance(step_estimator, PostPredictionTransformation):
                    raise AssertionError(
                        "PipelineElement of type PostPredictionTransformation occured before the last step."
                    )
                non_post_processing_steps.append((step_name, step_estimator))
        return list(non_post_processing_steps[::-1])

    def _post_processing_steps(self) -> list[tuple[str, PostPredictionTransformation]]:
        """Return last steps which are PostPredictionTransformation.

        Returns
        -------
        list[tuple[str, PostPredictionTransformation]]
            List of tuples containing the name and the PostPredictionTransformation.
        """
        post_processing_steps = []
        for step_name, step_estimator in self.steps[::-1]:
            if isinstance(step_estimator, PostPredictionTransformation):
                post_processing_steps.append((step_name, step_estimator))
            else:
                break
        return list(post_processing_steps[::-1])

    def _agg_non_postpred_steps(
        self,
    ) -> Iterable[_AggregatedPipelineStep]:
        """Generate (idx, (name, trans)) tuples from self.steps.

        When filter_passthrough is True, 'passthrough' and None transformers
        are filtered out.

        Returns
        ------
        Iterable[_AggregatedPipelineStep]
            The _AggregatedPipelineStep is composed of the index, the name and the transformer.
        """
        aggregated_transformer_list = []
        for i, (name_i, step_i) in enumerate(self._non_post_processing_steps()):
            if isinstance(step_i, ABCPipelineElement):
                aggregated_transformer_list.append((i, name_i, step_i))
            else:
                if aggregated_transformer_list:
                    index_list = [step[0] for step in aggregated_transformer_list]
                    name_list = [step[1] for step in aggregated_transformer_list]
                    transformer_list = [step[2] for step in aggregated_transformer_list]
                    if len(aggregated_transformer_list) == 1:
                        yield index_list[0], name_list[0], transformer_list[0]
                    else:
                        pipeline = _MolPipeline(transformer_list, n_jobs=self.n_jobs)
                        yield index_list, name_list, pipeline
                    aggregated_transformer_list = []
                yield i, name_i, step_i

        # yield last step if anything remains
        if aggregated_transformer_list:
            index_list = [step[0] for step in aggregated_transformer_list]
            name_list = [step[1] for step in aggregated_transformer_list]
            transformer_list = [step[2] for step in aggregated_transformer_list]

            if len(aggregated_transformer_list) == 1:
                yield index_list[0], name_list[0], transformer_list[0]

            elif len(aggregated_transformer_list) > 1:
                pipeline = _MolPipeline(transformer_list, n_jobs=self.n_jobs)
                yield index_list, name_list, pipeline

    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> Self:
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
        routed_params = self._check_method_params(method="fit", props=fit_params)
        Xt, yt = self._fit(X, y, routed_params)  # pylint: disable=invalid-name
        with print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                if is_empty(Xt):
                    logger.warning(
                        "All input rows were filtered out! Model is not fitted!"
                    )
                else:
                    fit_params_last_step = routed_params[self.steps[-1][0]]
                    self._final_estimator.fit(Xt, yt, **fit_params_last_step["fit"])

        return self

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

    def _can_decision_function(self) -> bool:
        """Check if the final estimator implements decision_function.

        Returns
        -------
        bool
            True if the final estimator implements decision_function.
        """
        return hasattr(self._final_estimator, "decision_function")

    @available_if(_can_fit_transform)
    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_transform(self, X: Any, y: Any = None, **params: Any) -> Any:
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

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        routed_params = self._check_method_params(method="fit_transform", props=params)
        iter_input, iter_label = self._fit(X, y, routed_params)
        last_step = self._final_estimator
        with print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                pass
            elif is_empty(iter_input):
                logger.warning("All input rows were filtered out! Model is not fitted!")
            else:
                last_step_params = routed_params[self.steps[-1][0]]
                if hasattr(last_step, "fit_transform"):
                    iter_input = last_step.fit_transform(
                        iter_input, iter_label, **last_step_params["fit_transform"]
                    )
                elif hasattr(last_step, "transform") and hasattr(last_step, "fit"):
                    last_step.fit(iter_input, iter_label, **last_step_params["fit"])
                    iter_input = last_step.transform(
                        iter_input, **last_step_params["transform"]
                    )
                else:
                    raise TypeError(
                        f"fit_transform of the final estimator"
                        f" {last_step.__class__.__name__} {last_step_params} does not "
                        f"match fit_transform of Pipeline {self.__class__.__name__}"
                    )
            for _, post_element in self._post_processing_steps():
                iter_input = post_element.fit_transform(iter_input, iter_label)
        return iter_input

    @available_if(_final_estimator_has("predict"))
    def predict(self, X: Any, **params: Any) -> Any:
        """Transform the data, and apply `predict` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls `predict`
        method. Only valid if the final estimator implements `predict`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

            .. versionadded:: 0.20

        Returns
        -------
        y_pred : ndarray
            Result of calling `predict` on the final estimator.
        """
        if _routing_enabled():
            routed_params = process_routing(self, "predict", **params)
        else:
            routed_params = process_routing(self, "predict", **{})

        iter_input = self._transform(X, routed_params)

        if self._final_estimator == "passthrough":
            pass
        elif is_empty(iter_input):
            iter_input = []
        elif hasattr(self._final_estimator, "predict"):
            if _routing_enabled():
                iter_input = self._final_estimator.predict(
                    iter_input, **routed_params[self._final_estimator].predict
                )
            else:
                iter_input = self._final_estimator.predict(iter_input, **params)
        else:
            raise AssertionError(
                "Final estimator does not implement predict, hence this function should not be available."
            )
        for _, post_element in self._post_processing_steps():
            iter_input = post_element.transform(iter_input)
        return iter_input

    @available_if(_final_estimator_has("fit_predict"))
    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_predict(self, X: Any, y: Any = None, **params: Any) -> Any:
        """Transform the data, and apply `fit_predict` with the final estimator.

        Call `fit_transform` of each transformer in the pipeline. The
        transformed data are finally passed to the final estimator that calls
        `fit_predict` method. Only valid if the final estimator implements
        `fit_predict`.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : ndarray
            Result of calling `fit_predict` on the final estimator.
        """
        routed_params = self._check_method_params(method="fit_predict", props=params)
        iter_input, iter_label = self._fit(
            X, y, routed_params
        )  # pylint: disable=invalid-name

        params_last_step = routed_params[self.steps[-1][0]]
        with print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator == "passthrough":
                y_pred = iter_input
            elif is_empty(iter_input):
                logger.warning("All input rows were filtered out! Model is not fitted!")
                iter_input = []
                y_pred = []
            elif hasattr(self._final_estimator, "fit_predict"):
                y_pred = self._final_estimator.fit_predict(
                    iter_input, iter_label, **params_last_step.get("fit_predict", {})
                )
            else:
                raise AssertionError(
                    "Final estimator does not implement fit_predict, hence this function should not be available."
                )
            for _, post_element in self._post_processing_steps():
                y_pred = post_element.fit_transform(y_pred, iter_label)
        return y_pred

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X: Any, **params: Any) -> Any:
        """Transform the data, and apply `predict_proba` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls `predict_proba`
        method. Only valid if the final estimator implements `predict_proba`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

            .. versionadded:: 0.20

        Returns
        -------
        y_pred : ndarray
            Result of calling `predict_proba` on the final estimator.
        """
        routed_params = process_routing(self, "predict_proba", **params)
        iter_input = self._transform(X, routed_params)

        if self._final_estimator == "passthrough":
            pass
        elif is_empty(iter_input):
            iter_input = []
        elif hasattr(self._final_estimator, "predict_proba"):
            if _routing_enabled():
                iter_input = self._final_estimator.predict_proba(
                    iter_input, **routed_params[self.steps[-1][0]].predict_proba
                )
            else:
                iter_input = self._final_estimator.predict_proba(iter_input, **params)
        else:
            raise AssertionError(
                "Final estimator does not implement predict_proba, hence this function should not be available."
            )
        for _, post_element in self._post_processing_steps():
            iter_input = post_element.transform(iter_input)
        return iter_input

    def _can_transform(self) -> bool:
        """Check if the final estimator can transform or is passthrough.

        Returns
        -------
        bool
            True if the final estimator can transform or is passthrough.
        """
        return self._final_estimator == "passthrough" or hasattr(
            self._final_estimator, "transform"
        )

    @available_if(_can_transform)
    def transform(self, X: Any, **params: Any) -> Any:
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
        routed_params = process_routing(self, "transform", **params)
        iter_input = X
        for _, name, transform in self._iter():
            if transform == "passthrough":
                continue
            if is_empty(iter_input):
                # This is done to prime the error filters
                if isinstance(transform, _MolPipeline):
                    _ = transform.transform(iter_input)
                iter_input = []
                break
            if hasattr(transform, "transform"):
                iter_input = transform.transform(
                    iter_input, **routed_params[name].transform
                )
            else:
                raise AssertionError(
                    "Non transformer ocurred in transformation step. This should have been caught in the validation step."
                )
        for _, post_element in self._post_processing_steps():
            iter_input = post_element.transform(iter_input, **params)
        return iter_input

    @available_if(_can_decision_function)
    def decision_function(self, X: Any, **params: Any) -> Any:
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
        if _routing_enabled():
            routed_params = process_routing(self, "decision_function", **params)
        else:
            routed_params = process_routing(self, "decision_function", **{})

        iter_input = self._transform(X, routed_params)
        if self._final_estimator == "passthrough":
            pass
        elif is_empty(iter_input):
            iter_input = []
        elif hasattr(self._final_estimator, "decision_function"):
            if _routing_enabled():
                iter_input = self._final_estimator.decision_function(
                    iter_input, **routed_params[self._final_estimator].predict
                )
            else:
                iter_input = self._final_estimator.decision_function(
                    iter_input, **params
                )
        else:
            raise AssertionError(
                "Final estimator does not implement `decision_function`, hence this function should not be available."
            )
        for _, post_element in self._post_processing_steps():
            iter_input = post_element.transform(iter_input)
        return iter_input

    @property
    def classes_(self) -> list[Any] | npt.NDArray[Any]:
        """Return the classes of the last element, which is not a PostPredictionTransformation."""
        check_last = [
            step
            for step in self.steps
            if not isinstance(step[1], PostPredictionTransformation)
        ]
        last_step = check_last[-1][1]
        if last_step == "passthrough":
            raise ValueError("Last step is passthrough.")
        if hasattr(last_step, "classes_"):
            return last_step.classes_
        raise ValueError("Last step has no classes_ attribute.")
