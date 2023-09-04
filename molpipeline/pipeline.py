"""Defines the pipeline which handles pipeline elements."""
from __future__ import annotations

from multiprocessing import Pool
from typing import Any, Iterable, List, Literal, Optional, Union, Tuple, TypeVar

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import joblib
import numpy as np
from sklearn.base import (
    _fit_context,  # pylint: disable=protected-access
    clone,
)
from sklearn.utils.metaestimators import available_if
from sklearn.pipeline import (
    Pipeline as _Pipeline,
    _fit_transform_one,
    _final_estimator_has,
)
from sklearn.utils import (
    _print_elapsed_time,
)
from sklearn.utils.validation import check_memory
from rdkit.rdBase import BlockLogs

from molpipeline.abstract_pipeline_elements.core import (
    ABCPipelineElement,
    RemovedInstance,
    TransformingPipelineElement,
)
from molpipeline.utils.multi_proc import check_available_cores
from molpipeline.utils.molpipeline_types import (
    AnyPredictor,
    AnyTransformer,
    NumberIterable,
)
from molpipeline.pipeline_elements.none_handling import (
    NoneFilter,
    NoneFiller,
    _MultipleNoneFilter,
)
from molpipeline.pipeline_elements.post_prediction import (
    PostPredictionTransformation,
    PostPredictionWrapper,
)


__all__ = ["Pipeline"]


_T = TypeVar("_T")


class _MolPipeline:
    """Contains the PipeElements which describe the functionality of the pipeline."""

    _n_jobs: int
    _element_list: list[ABCPipelineElement]
    _requires_fitting: bool
    _removed_rows: tuple[int, dict[int, str]]  # (total rows in array, {index: reason})

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
    def _filter_elements(self) -> list[NoneFilter]:
        return [
            element for element in self._element_list if isinstance(element, NoneFilter)
        ]

    @property
    def _filter_elements_agg(self) -> _MultipleNoneFilter:
        return _MultipleNoneFilter(self._filter_elements)

    @property
    def _transforming_elements(
        self,
    ) -> list[Union[TransformingPipelineElement, _MolPipeline]]:
        return [
            element
            for element in self._element_list
            if isinstance(element, TransformingPipelineElement)
            or isinstance(element, _MolPipeline)
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
        self.set_params(parameter_dict)

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
        if "name" in parameter_dict:
            self.name = parameter_dict["name"]
        if "raise_nones" in parameter_dict:
            self.raise_nones = parameter_dict["raise_nones"]
        return self

    @property
    def element_list(self) -> list[ABCPipelineElement]:
        """Get a shallow copy from the list of pipeline elements."""
        return self._element_list[:]  # [:] to create shallow copy.

    def _get_meta_element_list(
        self,
    ) -> list[Union[ABCPipelineElement, _MolPipeline]]:
        """Merge elements which do not require fitting to a meta element which improves parallelization."""
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
        y: Any = None,
        **fit_params: dict[Any, Any],
    ) -> Self:
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
        _ = y  # Making pylint happy
        _ = fit_params  # Making pylint happy
        if self.requires_fitting:
            self.fit_transform(x_input)
        return self

    def fit_transform(
        self,
        x_input: Any,
        y: Any = None,
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
        removed_rows: dict[NoneFilter, list[int]] = {}
        for none_filter in self._filter_elements:
            removed_rows[none_filter] = []
        iter_idx_array = np.arange(len(iter_input))
        for i_element in iter_element_list:
            if not isinstance(i_element, (TransformingPipelineElement, _MolPipeline)):
                continue
            i_element.n_jobs = self.n_jobs
            iter_input = i_element.pretransform(iter_input)
            for none_filter in self._filter_elements:
                iter_input = none_filter.transform(iter_input)
                for idx in none_filter.none_indices:
                    original_idx = iter_idx_array[idx]
                    removed_rows[none_filter].append(original_idx)
                iter_idx_array = none_filter.co_transform(iter_idx_array)
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
        for none_filter in self._filter_elements:
            removed_idx_list = removed_rows[none_filter]
            none_filter.none_indices = []
            for new_idx, old_idx in enumerate(iter_idx_array):
                if old_idx in removed_idx_list:
                    none_filter.none_indices.append(new_idx)
            none_filter.n_total = len(iter_idx_array)
            iter_idx_array = none_filter.co_transform(iter_idx_array)
        none_filler_list = [ele for ele in self._element_list if isinstance(ele, NoneFiller)]
        for none_filler in none_filler_list:
            none_filler.select_none_filter(self._filter_elements)
            iter_input = none_filler.transform(iter_input)
        return iter_input

    def transform_single(self, input_value: Any) -> Any:
        iter_value = input_value
        for p_element in self._element_list:
            iter_value = p_element.transform_single(iter_value)
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
        output_generator: Iterable[Any]
            Generator which yields the output of the pipeline.

        Returns
        -------
        Any
            Assembled output.
        """
        last_element = self._transforming_elements[-1]
        if hasattr(last_element, "assemble_output"):
            return last_element.assemble_output(value_list)
        else:
            return list(value_list)

    def apply_to_all(self, x_input: Any) -> list[Any]:
        return list(self._transform_iterator(x_input))

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
        log_block = BlockLogs()
        agg_filter = self._filter_elements_agg
        for filter_element in self._filter_elements:
            filter_element.none_indices = []
        if self.n_jobs > 1:
            with Pool(self.n_jobs) as pool:
                iter_func = pool.imap(self.transform_single, x_input)
                for i, transformed_value in enumerate(iter_func):
                    if isinstance(transformed_value, RemovedInstance):
                        agg_filter.register_removed(i, transformed_value)
                    else:
                        yield transformed_value
        else:
            for i, value in enumerate(x_input):
                transformed_value = self.transform_single(value)
                if isinstance(transformed_value, RemovedInstance):
                    if isinstance(transformed_value, RemovedInstance):
                        agg_filter.register_removed(i, transformed_value)
                else:
                    yield transformed_value
        self._finish()
        del log_block

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
        return self._filter_elements_agg.transform(x_input)


# Cannot be moved to utils.molpipeline_types due to circular imports
_Step = Tuple[int, str, Union[AnyTransformer, AnyPredictor]]
_AggStep = Tuple[List[int], List[str], _MolPipeline]
_AggregatedPipelineStep = Union[_Step, _AggStep]


class Pipeline(_Pipeline):
    """Defines the pipeline which handles pipeline elements."""

    steps: list[
        tuple[
            str,
            Union[AnyTransformer, AnyPredictor, ABCPipelineElement],
        ],
    ]
    #  * Adapted methods from sklearn.pipeline.Pipeline *

    def __init__(
        self,
        steps: list[
            tuple[str, Union[AnyTransformer, AnyPredictor, ABCPipelineElement]]
        ],
        *,
        memory: Optional[Union[str, joblib.Memory]] = None,
        verbose: bool = False,
        n_jobs: int = 1,
        raise_nones: bool = False,
    ) -> None:
        """Initialize Pipeline.

        Parameters
        ----------
        steps: list[tuple[str, Union[AnyTransformer, AnyPredictor, TransformingPipelineElement]]]
            List of (name, Estimator) tuples.
        memory: str, optional
            Path to cache transformers.
        verbose: bool, optional
            If True, print additional information.
        n_jobs: int, optional
            Number of cores used for aggregated steps.

        Returns
        -------
        None
        """
        super().__init__(steps, memory=memory, verbose=verbose)
        self.n_jobs = n_jobs
        self.raise_nones = raise_nones

        none_filler_list = [
            n_filler for _, n_filler in self.steps if isinstance(n_filler, NoneFiller)
        ]
        none_filter_list = [
            n_filter for _, n_filter in self.steps if isinstance(n_filter, NoneFilter)
        ]
        for step in self.steps:
            if isinstance(step[1], PostPredictionWrapper):
                if isinstance(step[1].estimator, NoneFiller):
                    none_filler_list.append(step[1].estimator)
        for none_filler in none_filler_list:
            none_filler.select_none_filter(none_filter_list)

    def _validate_steps(self) -> None:
        names = [name for name, _ in self.steps]

        # validate names
        self._validate_names(names)

        # validate estimators
        non_post_processing_steps = [e for _, _, e in self._agg_non_postpred_steps()]
        transformers = non_post_processing_steps[:-1]
        estimator = non_post_processing_steps[-1]

        for t in transformers:
            if t is None or t == "passthrough":
                continue
            if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(
                t, "transform"
            ):
                raise TypeError(
                    "All intermediate steps should be "
                    "transformers and implement fit and transform "
                    "or be the string 'passthrough' "
                    "'%s' (type %s) doesn't" % (t, type(t))
                )

        # We allow last estimator to be None as an identity transformation
        if (
            estimator is not None
            and estimator != "passthrough"
            and not hasattr(estimator, "fit")
        ):
            raise TypeError(
                "Last step of Pipeline should implement fit "
                "or be the string 'passthrough'. "
                "'%s' (type %s) doesn't" % (estimator, type(estimator))
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

        Yields
        ------
        _AggregatedPipelineStep
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
            return self._final_estimator._estimator_type
        return None

    @property
    def _final_estimator(
        self,
    ) -> Union[Literal["passthrough"], AnyTransformer, AnyPredictor, _MolPipeline]:
        """Return the lst estimator which is not a PostprocessingTransformer."""
        element_list = list(self._agg_non_postpred_steps())
        last_element = element_list[-1]
        return last_element[2]

    def _fit(self, X: Any, y: Any = None, **fit_params_steps: Any) -> tuple[Any, Any]:
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()

        # Set up the memory
        memory: joblib.Memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)
        for step in self._iter(with_final=False, filter_passthrough=False):
            step_idx, name, transformer = step
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)
            if isinstance(cloned_transformer, _MolPipeline):
                fit_parameter = {
                    "element_parameters": [fit_params_steps[n] for n in name]
                }
            else:
                if isinstance(name, list):
                    raise AssertionError()
                fit_parameter = fit_params_steps[name]
            # Fit or load from cache the current transformer
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                **fit_parameter,
            )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            if isinstance(fitted_transformer, _MolPipeline):
                ele_list = fitted_transformer.element_list
                if not isinstance(name, list):
                    raise AssertionError()
                if not isinstance(step_idx, list):
                    raise AssertionError()
                if not len(name) == len(step_idx) == len(ele_list):
                    raise AssertionError()
                for idx_i, name_i, ele_i in zip(step_idx, name, ele_list):
                    self.steps[idx_i] = (name_i, ele_i)
                y = fitted_transformer.co_transform(y)
            else:
                if isinstance(name, list):
                    raise AssertionError()
                if isinstance(step_idx, list):
                    raise AssertionError()
                self.steps[step_idx] = (name, fitted_transformer)
        return X, y

    # * New implemented methods *
    def _non_post_processing_steps(
        self,
    ) -> list[tuple[str, Union[AnyTransformer, AnyPredictor, ABCPipelineElement]]]:
        """Return all steps before the first PostPredictionTransformation."""
        non_post_processing_steps = []
        start_adding = False
        for step_name, step_estimator in self.steps[::-1]:
            if not isinstance(step_estimator, PostPredictionTransformation):
                start_adding = True
            if start_adding:
                non_post_processing_steps.append((step_name, step_estimator))
        return list(non_post_processing_steps[::-1])

    def _post_processing_steps(self) -> list[tuple[str, PostPredictionTransformation]]:
        """Return last steps which are PostPredictionTransformation."""
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
                    pipeline = _MolPipeline(transformer_list, n_jobs=self.n_jobs)
                    yield index_list, name_list, pipeline
                    aggregated_transformer_list = []
                yield i, name_i, step_i
        # yield last step if anything remains
        if aggregated_transformer_list:
            index_list = [step[0] for step in aggregated_transformer_list]
            name_list = [step[1] for step in aggregated_transformer_list]
            transformer_list = [step[2] for step in aggregated_transformer_list]
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
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, yt, **fit_params_last_step)

        return self

    def _can_fit_transform(self) -> bool:
        return (
            self._final_estimator == "passthrough"
            or hasattr(self._final_estimator, "transform")
            or hasattr(self._final_estimator, "fit_transform")
        )

    @available_if(_can_fit_transform)
    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
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

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)
        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                Xt = Xt
            else:
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                if hasattr(last_step, "fit_transform"):
                    Xt = last_step.fit_transform(Xt, yt, **fit_params_last_step)
                elif hasattr(last_step, "transform") and hasattr(last_step, "fit"):
                    last_step.fit(Xt, yt, **fit_params_last_step)
                    Xt = last_step.transform(Xt)
                else:
                    raise TypeError(
                        "fit_transform of the final estimator"
                        " {} {} does not "
                        "match fit_transform of Pipeline {}".format(
                            last_step.__class__.__name__,
                            fit_params_last_step,
                            self.__class__.__name__,
                        )
                    )
            for post_name, post_element in self._post_processing_steps():
                Xt = post_element.transform(Xt)
        return Xt

    @available_if(_final_estimator_has("predict"))
    def predict(self, X: Any, **predict_params: Any) -> Any:
        """Transform the data, and apply `predict` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls `predict`
        method. Only valid if the final estimator implements `predict`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_params : dict of string -> object
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
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            if hasattr(transform, "transform"):
                Xt = transform.transform(Xt)
            else:
                raise AssertionError(
                    f"Non transformer ocurred in transformation step: {transform}."
                )
        if self._final_estimator == "passthrough":
            return Xt
        if hasattr(self._final_estimator, "predict"):
            return self._final_estimator.predict(Xt, **predict_params)
        else:
            raise AssertionError(
                "Final estimator does not implement predict, hence this function should not be available."
            )

    @available_if(_final_estimator_has("fit_predict"))
    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_predict(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
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

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : ndarray
            Result of calling `fit_predict` on the final estimator.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)

        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator == "passthrough":
                y_pred = Xt
            elif hasattr(self._final_estimator, "fit_predict"):
                y_pred = self._final_estimator.fit_predict(
                    Xt, yt, **fit_params_last_step
                )
            else:
                raise AssertionError(
                    "Final estimator does not implement fit_predict, hence this function should not be available."
                )
            for post_name, post_element in self._post_processing_steps():
                y_pred = post_element.fit_transform(y_pred, yt)
        return y_pred

    def _can_transform(self) -> bool:
        return self._final_estimator == "passthrough" or hasattr(
            self._final_estimator, "transform"
        )

    @available_if(_can_transform)
    def transform(self, X: Any) -> Any:
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

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        Xt = X
        for _, _, transform in self._iter():
            if hasattr(transform, "transform"):
                Xt = transform.transform(Xt)
            else:
                raise AssertionError(
                    "Non transformer ocurred in transformation step. This should have been caught in the validation step."
                )
        for post_name, post_element in self._post_processing_steps():
            Xt = post_element.transform(Xt)
        return Xt
