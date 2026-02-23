"""Module to change functions of the sklearn pipeline."""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt
from loguru import logger
from sklearn.base import _fit_context, clone  # noqa: PLC2701
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.pipeline import (
    _final_estimator_has,  # noqa: PLC2701
    _fit_transform_one,  # noqa: PLC2701
    _raise_or_warn_if_not_fitted,  # noqa: PLC2701
)
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _routing_enabled,  # noqa: PLC2701
    process_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_memory

from molpipeline.error_handling import ErrorFilter, FilterReinserter
from molpipeline.post_prediction import (
    PostPredictionTransformation,
    PostPredictionWrapper,
)
from molpipeline.utils.logging import print_elapsed_time
from molpipeline.utils.molpipeline_types import (
    AnyElement,
    AnyStep,
)
from molpipeline.utils.value_checks import is_empty

if TYPE_CHECKING:
    from collections.abc import Generator

    import joblib
    from sklearn.utils import Bunch


_IndexedStep = tuple[int, str, AnyElement]


class AdapterPipeline(_Pipeline):
    """Defines the pipeline which handles pipeline elements."""

    steps: list[AnyStep]
    #  * Adapted methods from sklearn.pipeline.Pipeline *

    @property
    def _estimator_type(self) -> Any:
        """Return the estimator type."""
        if self._final_estimator is None or self._final_estimator == "passthrough":
            return None
        if hasattr(self._final_estimator, "_estimator_type"):
            return self._final_estimator._estimator_type  # noqa: SLF001  # pylint: disable=protected-access
        return None

    @property
    def _modified_steps(
        self,
    ) -> list[AnyStep]:
        """Return modified version of steps.

        Returns only steps before the first PostPredictionTransformation.

        Raises
        ------
        AssertionError
            If a PostPredictionTransformation is found before the last step.

        """
        non_post_processing_steps: list[AnyStep] = []
        start_adding = False
        for step_name, step_estimator in self.steps[::-1]:
            if not isinstance(step_estimator, PostPredictionTransformation):
                start_adding = True
            if start_adding:
                if isinstance(step_estimator, PostPredictionTransformation):
                    raise AssertionError(
                        "PipelineElement of type PostPredictionTransformation occured "
                        "before the last step.",
                    )
                non_post_processing_steps.append((step_name, step_estimator))
        return list(non_post_processing_steps[::-1])

    @property
    def _post_processing_steps(self) -> list[tuple[str, PostPredictionTransformation]]:
        """Return last steps which are PostPredictionTransformation."""
        post_processing_steps = []
        for step_name, step_estimator in self.steps[::-1]:
            if isinstance(step_estimator, PostPredictionTransformation):
                post_processing_steps.append((step_name, step_estimator))
            else:
                break
        return list(post_processing_steps[::-1])

    @property
    def classes_(self) -> list[Any] | npt.NDArray[Any]:
        """Return the classes of the last element.

        PostPredictionTransformation elements are not considered as last element.

        Raises
        ------
        ValueError
            If the last step is passthrough or has no classes_ attribute.

        """
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

    def __init__(
        self,
        steps: list[AnyStep],
        *,
        memory: str | joblib.Memory | None = None,
        verbose: bool = False,
        n_jobs: int = 1,
    ):
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
            if isinstance(step[1], PostPredictionWrapper):  # noqa: SIM102
                if isinstance(step[1].wrapped_estimator, FilterReinserter):
                    error_replacer_list.append(step[1].wrapped_estimator)
        for error_replacer in error_replacer_list:
            error_replacer.select_error_filter(error_filter_list)

    def _validate_steps(self) -> None:
        """Validate the steps.

        Raises
        ------
        TypeError
            If the steps do not implement fit and transform or are not 'passthrough'.

        """
        names = [name for name, _ in self.steps]

        # validate names
        self._validate_names(names)

        # validate estimators
        estimator = self._modified_steps[-1][1]

        for _, transformer in self._modified_steps[:-1]:
            if transformer is None or transformer == "passthrough":
                continue
            if not (
                hasattr(transformer, "fit") or hasattr(transformer, "fit_transform")
            ) or not hasattr(transformer, "transform"):
                raise TypeError(
                    f"All intermediate steps should be "
                    f"transformers and implement fit and transform "
                    f"or be the string 'passthrough' "
                    f"'{transformer}' (type {type(transformer)}) doesn't",
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
                f"'{estimator}' (type {type(estimator)}) doesn't",
            )

        # validate post-processing steps
        # Calling steps automatically validates them
        _ = self._post_processing_steps

    def _iter(
        self,
        with_final: bool = True,
        filter_passthrough: bool = True,
    ) -> Generator[
        tuple[int, str, AnyElement] | tuple[list[int], list[str], AdapterPipeline],
        Any,
        None,
    ]:
        """Iterate over all non post-processing steps.

        Steps which are children of a ABCPipelineElement were aggregated to a
        MolPipeline.

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
        stop = len(self._modified_steps)
        if not with_final:
            stop -= 1

        for idx, (name, trans) in enumerate(islice(self._modified_steps, 0, stop)):
            if not filter_passthrough or (trans is not None and trans != "passthrough"):
                yield idx, name, trans

    # pylint: disable=too-many-locals,too-many-branches
    def _fit(  # noqa: PLR0912
        self,
        X: Any,  # noqa: N803
        y: Any = None,
        routed_params: dict[str, Any] | None = None,
        raw_params: dict[str, Any] | None = None,
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
            Although this is marked as optional, it should not be None.
            The awkward (argward?) typing is due to inheritance from sklearn.
            Can be an empty dictionary.
        raw_params : dict[str, Any], optional
            Parameters passed by the user, used when `transform_input`

        Raises
        ------
        AssertionError
            If routed_params is None or if the transformer is 'passthrough'.
        AssertionError
            If the names are a list and the step is not a Pipeline.

        Returns
        -------
        tuple[Any, Any]
            The transformed data and the transformed objectives.

        """
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        if routed_params is None:
            raise AssertionError("routed_params should not be None.")

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
            if isinstance(name, list):
                if not isinstance(transformer, _Pipeline):
                    raise AssertionError(
                        "If the name is a list, the transformer must be a Pipeline.",
                    )
                if routed_params:
                    step_params = {
                        "element_parameters": [routed_params[n] for n in name],
                    }
                else:
                    step_params = {}
            else:
                step_params = self._get_metadata_for_step(
                    step_idx=step_idx,
                    step_params=routed_params[name],
                    all_params=raw_params,
                )

            # Fit or load from cache the current transformer
            X, fitted_transformer = fit_transform_one_cached(  # noqa: N806  # type: ignore
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                params=step_params,
            )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            if isinstance(fitted_transformer, AdapterPipeline):
                ele_list = [step[1] for step in fitted_transformer.steps]
                if not isinstance(name, list) or not isinstance(step_idx, list):
                    raise AssertionError()
                if not len(name) == len(step_idx) == len(ele_list):
                    raise AssertionError()
                for idx_i, name_i, ele_i in zip(step_idx, name, ele_list, strict=True):
                    self.steps[idx_i] = (name_i, ele_i)
                if y is not None:
                    y = fitted_transformer.co_transform(y)
                for idx_i, name_i, ele_i in zip(step_idx, name, ele_list, strict=True):
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
        self,
        X: Any,  # pylint: disable=invalid-name  # noqa: N803
        routed_params: Bunch,
    ) -> Any:
        """Transform the data, and skip final estimator.

        Call `transform` of each transformer in the pipeline except the last one,

        Parameters
        ----------
        X : iterable
            Data to predict on.
            Must fulfill input requirements of first step of the pipeline.

        routed_params: Bunch
            Parameters for each step as returned by process_routing `transform`.

        Raises
        ------
        AssertionError
            If one of the transformers is 'passthrough' or does not implement.

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
            if transform == "passthrough":
                raise AssertionError("Passthrough should have been filtered out.")
            if hasattr(transform, "transform"):
                if do_routing:
                    iter_input = transform.transform(  # type: ignore[call-arg]
                        iter_input,
                        routed_params[name].transform,
                    )
                else:
                    iter_input = transform.transform(iter_input)
            else:
                raise AssertionError(
                    f"Non transformer ocurred in transformation step: {transform}.",
                )
        return iter_input

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
        routed_params = self._check_method_params(method="fit", props=fit_params)
        xt, yt = self._fit(X, y, routed_params)
        with print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                if is_empty(xt):
                    logger.warning(
                        "All input rows were filtered out! Model is not fitted!",
                    )
                else:
                    fit_params_last_step = routed_params[self._modified_steps[-1][0]]
                    self._final_estimator.fit(xt, yt, **fit_params_last_step["fit"])

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
        TypeError
            If the last step does not implement `fit_transform` or `fit` and
            `transform`.

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
                last_step_params = routed_params[self._modified_steps[-1][0]]
                if hasattr(last_step, "fit_transform"):
                    iter_input = last_step.fit_transform(
                        iter_input,
                        iter_label,
                        **last_step_params["fit_transform"],
                    )
                elif hasattr(last_step, "transform") and hasattr(last_step, "fit"):
                    last_step.fit(iter_input, iter_label, **last_step_params["fit"])
                    iter_input = last_step.transform(
                        iter_input,
                        **last_step_params["transform"],
                    )
                else:
                    raise TypeError(
                        f"fit_transform of the final estimator"
                        f" {last_step.__class__.__name__} {last_step_params} does not "
                        f"match fit_transform of Pipeline {self.__class__.__name__}",
                    )
            for _, post_element in self._post_processing_steps:
                iter_input = post_element.fit_transform(iter_input, iter_label)
        return iter_input

    @available_if(_final_estimator_has("fit_predict"))
    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False,
    )
    def fit_predict(
        self,
        X: Any,  # noqa: N803
        y: Any = None,
        **params: Any,
    ) -> Any:
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

        Raises
        ------
        AssertionError
            If the final estimator does not implement `fit_predict`.
            In this case this function should not be available.

        Returns
        -------
        y_pred : ndarray
            Result of calling `fit_predict` on the final estimator.

        """
        routed_params = self._check_method_params(method="fit_predict", props=params)
        iter_input, iter_label = self._fit(X, y, routed_params)

        params_last_step = routed_params[self._modified_steps[-1][0]]
        with print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator == "passthrough":
                y_pred = iter_input
            elif is_empty(iter_input):
                logger.warning("All input rows were filtered out! Model is not fitted!")
                iter_input = []
                y_pred = []
            elif hasattr(self._final_estimator, "fit_predict"):
                y_pred = self._final_estimator.fit_predict(
                    iter_input,
                    iter_label,
                    **params_last_step.get("fit_predict", {}),
                )
            else:
                raise AssertionError(
                    "Final estimator does not implement fit_predict, "
                    "hence this function should not be available.",
                )
            for _, post_element in self._post_processing_steps:
                y_pred = post_element.fit_transform(y_pred, iter_label)
        return y_pred

    @available_if(_final_estimator_has("predict"))
    def predict(
        self,
        X: npt.NDArray[Any] | list[Any],  # noqa: N803
        **params: Any,
    ) -> npt.NDArray[Any] | list[Any]:
        """Transform the data, and apply `predict` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls `predict`
        method. Only valid if the final estimator implements `predict`.

        Parameters
        ----------
        X : npt.NDArray[Any]
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **params : Any
            If `enable_metadata_routing=False` (default): Parameters to the
            ``predict`` called at the end of all transformations in the pipeline.
            If `enable_metadata_routing=True`: Parameters requested and accepted by
            steps. Each step must have requested certain metadata for these parameters
            to be forwarded to them.

        Raises
        ------
        AssertionError
            If a step before the final estimator is 'passthrough' or does not
            implement `transform`.

        Returns
        -------
        y_pred : npt.NDArray[Any]
            Result of calling `predict` on the final estimator.

        """
        iter_input = X
        with _raise_or_warn_if_not_fitted(self):
            if not _routing_enabled():
                for _, name, transform in self._iter(with_final=False):
                    if (
                        not hasattr(transform, "transform")
                        or transform == "passthrough"
                    ):
                        raise AssertionError(
                            f"Non transformer occurred in transformation step: {name}.",
                        )
                    iter_input = transform.transform(iter_input)
                if is_empty(iter_input):
                    iter_input = []
                else:
                    iter_input = self._final_estimator.predict(iter_input, **params)
            else:
                # metadata routing enabled
                routed_params = process_routing(self, "predict", **params)
                for _, name, transform in self._iter(with_final=False):
                    if (
                        not hasattr(transform, "transform")
                        or transform == "passthrough"
                    ):
                        raise AssertionError(
                            f"Non transformer occurred in transformation step: {name}.",
                        )
                    iter_input = transform.transform(
                        iter_input,
                        **routed_params[name].transform,
                    )
                if is_empty(iter_input):
                    iter_input = []
                else:
                    iter_input = self._final_estimator.predict(
                        iter_input,
                        **routed_params[self.steps[-1][0]].predict,
                    )
        for _, post_element in self._post_processing_steps:
            iter_input = post_element.transform(iter_input, **params)
        return iter_input

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(
        self,
        X: Any,  # noqa: N803
        **params: Any,
    ) -> Any:
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

        Raises
        ------
        AssertionError
            If the final estimator does not implement `predict_proba`.
            In this case this function should not be available.

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
                    iter_input,
                    **routed_params[self._modified_steps[-1][0]].predict_proba,
                )
            else:
                iter_input = self._final_estimator.predict_proba(iter_input, **params)
        else:
            raise AssertionError(
                "Final estimator does not implement predict_proba, "
                "hence this function should not be available.",
            )
        for _, post_element in self._post_processing_steps:
            iter_input = post_element.transform(iter_input)
        return iter_input

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Notes
        -----
        This method is copied from the original sklearn implementation.
        Changes are marked with a comment.

        Returns
        -------
        MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.

        """
        router = MetadataRouter(owner=self.__class__.__name__)

        # first we add all steps except the last one
        for _, name, trans in self._iter(with_final=False, filter_passthrough=True):
            method_mapping = MethodMapping()
            # fit, fit_predict, and fit_transform call fit_transform if it
            # exists, or else fit and transform
            if hasattr(trans, "fit_transform"):
                (
                    method_mapping.add(caller="fit", callee="fit_transform")
                    .add(caller="fit_transform", callee="fit_transform")
                    .add(caller="fit_predict", callee="fit_transform")
                )
            else:
                (
                    method_mapping.add(caller="fit", callee="fit")
                    .add(caller="fit", callee="transform")
                    .add(caller="fit_transform", callee="fit")
                    .add(caller="fit_transform", callee="transform")
                    .add(caller="fit_predict", callee="fit")
                    .add(caller="fit_predict", callee="transform")
                )

            (
                method_mapping.add(caller="predict", callee="transform")
                .add(caller="predict", callee="transform")
                .add(caller="predict_proba", callee="transform")
                .add(caller="decision_function", callee="transform")
                .add(caller="predict_log_proba", callee="transform")
                .add(caller="transform", callee="transform")
                .add(caller="inverse_transform", callee="inverse_transform")
                .add(caller="score", callee="transform")
            )

            router.add(method_mapping=method_mapping, **{name: trans})

        # Only the _non_post_processing_steps is changed from the original
        # implementation is changed in the following line
        final_name, final_est = self._modified_steps[-1]
        if final_est is None or final_est == "passthrough":
            return router

        # then we add the last step
        method_mapping = MethodMapping()
        if hasattr(final_est, "fit_transform"):
            method_mapping.add(caller="fit_transform", callee="fit_transform")
        else:
            method_mapping.add(caller="fit", callee="fit").add(
                caller="fit",
                callee="transform",
            )
        (
            method_mapping.add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict")
            .add(caller="fit_predict", callee="fit_predict")
            .add(caller="predict_proba", callee="predict_proba")
            .add(caller="decision_function", callee="decision_function")
            .add(caller="predict_log_proba", callee="predict_log_proba")
            .add(caller="transform", callee="transform")
            .add(caller="inverse_transform", callee="inverse_transform")
            .add(caller="score", callee="score")
        )

        router.add(method_mapping=method_mapping, **{final_name: final_est})
        return router
