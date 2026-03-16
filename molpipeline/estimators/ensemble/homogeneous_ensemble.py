"""The base class for molpipeline ensemble models."""

import abc
from collections.abc import Iterator
from typing import Any, Generic, Literal, Self, TypeVar, overload

import joblib
import numpy as np
import numpy.typing as npt
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.metaestimators import available_if

from molpipeline.experimental.model_selection.splitter.bootstrap_splitter import (
    BootstrapSplit,
)
from molpipeline.utils.json_operations import get_init_params
from molpipeline.utils.molpipeline_types import (
    AnyPredictor,
    XType,
    YType,
)

_T = TypeVar("_T", BaseEstimator, AnyPredictor)

_ModelVar = TypeVar("_ModelVar", bound=BaseEstimator | AnyPredictor)


class HomogeneousEnsemble(abc.ABC, BaseEstimator, Generic[_ModelVar]):
    """Base class for ensemble models composed of the same type of model.

    This class does not inherit from sklearn's BaseEnsemble which is designed for
    bagging ensembles, which would be a special case of the here implemented design.

    """

    estimator: _ModelVar
    estimators_: list[_ModelVar]

    def __init__(
        self,
        estimator: _ModelVar,
        sampler: int | BaseCrossValidator = 100,
        random_state: int | None = None,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the SplitEnsemble.

        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator to be cloned for each split.
        sampler: int | BaseCrossValidator, default=100
            The sampler to be used for creating the splits.
            If an int is passed, a bootstrap sample is used.
        random_state: int | None, optional
            The random state to be used for the sampler if it is an int.
        n_jobs : int, default=1
            The number of jobs to run in parallel when fitting the estimators.
        kwargs : Any
            Additional keyword arguments to be passed to the base estimator.

        """
        self.estimator = estimator
        self.sampler = sampler
        self.random_state = random_state
        self.estimators_ = []
        self.n_jobs = n_jobs
        self.set_params(**kwargs)
        super().__init__()

    @staticmethod
    def _fit_clone(
        model: _T,
        model_input: XType,
        y: YType,
        **kwargs: Any,
    ) -> _T:
        """Clone the model and fit it on the given data.

        Parameters
        ----------
        model : BaseEstimator
            The model to be fitted.
        model_input : npt.NDArray | scipy.sparse.csr_matrix
            The input data.
        y : npt.NDArray | None
            The target values.
        kwargs : Any
            Additional keyword arguments to be passed to the fit method of the model.

        Returns
        -------
        _T
            The fitted model.

        """
        model_clone: _T = clone(model)  # type: ignore
        return model_clone.fit(model_input, y, **kwargs)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get the parameters for the estimator.

        Parameters
        ----------
        deep: bool, default=True
            If True, will return the parameters for this estimator and its subobjects.

        Returns
        -------
        dict[str, Any]
            The parameters of the estimator.

        """
        params = super().get_params(deep=deep)
        sampler = self.sampler
        if not deep or isinstance(sampler, int):
            return params

        sampler_params = get_init_params(sampler, validation="raise")
        sampler_params = {f"sampler__{k}": v for k, v in sampler_params.items()}
        params.update(sampler_params)
        return params

    def set_params(self, **params: Any) -> Self:
        """Set the parameters for this estimator.

        Parameters
        ----------
        **params: Any
            The parameters to be set.

        Returns
        -------
        Self
            The updated estimator.

        """
        sampler_keys = {k for k in params if str(k).startswith("sampler__")}
        if not sampler_keys:
            return super().set_params(**params)

        params = dict(params)
        sampler_params = {
            k.replace("sampler__", ""): params.pop(k) for k in sampler_keys
        }

        sampler = params.pop("sampler", self.sampler)
        sampler_new_params = get_init_params(sampler, validation="raise")
        sampler_new_params.update(sampler_params)
        self.sampler = sampler.__class__(**sampler_new_params)
        return super().set_params(**params)

    def fit(
        self,
        X: XType,  # noqa: N803,  # pylint: disable=invalid-name
        y: YType = None,
        groups: YType = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the ensemble of estimators on the data.

        Parameters
        ----------
        X :  npt.NDArray | scipy.sparse.csr_matrix
            The input data.
        y :  npt.NDArray | None, optional
            The target values.
        groups: npt.ArrayLike | None, optional
            Group labels for the samples used while splitting the dataset into
            train/test.
        kwargs : Any
            Additional keyword arguments to be passed to the fit method of the base
            estimator.

        Returns
        -------
        self
            The fitted SplitEnsemble instance.

        """
        fit_clone_parallel = joblib.delayed(self._fit_clone)
        with joblib.Parallel(n_jobs=self.n_jobs) as parallel:
            self.estimators_ = parallel(
                fit_clone_parallel(self.estimator, feat_mat, target, **kwargs)
                for feat_mat, target in self._iter_model_inputs(X, y, groups)
            )
        return self

    def _iter_model_inputs(
        self,
        X: XType,  # noqa: N803,  # pylint: disable=invalid-name
        y: YType,
        groups: npt.ArrayLike | None = None,
    ) -> Iterator[tuple[XType, YType]]:
        """Iterate over the model inputs for each estimator in the ensemble.

        Parameters
        ----------
        X :  npt.NDArray | scipy.sparse.csr_matrix
            The input data.
        y :  npt.NDArray | None, optional
            The target values.
        groups: npt.ArrayLike | None, optional
            Group labels for the samples used while splitting the dataset into
            the datasets for the individual estimators.

        Yields
        ------
        Iterator[tuple[_X, _Y]]
            An iterator over the model inputs for each estimator in the ensemble.

        """
        sampler = self.sampler
        if isinstance(sampler, int):
            sampler = BootstrapSplit(sampler, random_state=self.random_state)
        features = X if sparse.issparse(X) else np.asarray(X)
        for train_idx, _ in sampler.split(X, y, groups):
            y_iter = np.asarray(y)[train_idx] if y is not None else None
            yield features[train_idx], y_iter

    @abc.abstractmethod
    def predict(
        self,
        X: XType,  # noqa: N803
        **params: Any,
    ) -> npt.NDArray[Any]:
        """Predict using the ensemble of estimators.

        Parameters
        ----------
        X : npt.NDArray | scipy.sparse.csr_matrix
            The input data to predict.
        params : Any
            Additional keyword arguments to be passed to the predict method of the
            individual estimators.

        Returns
        -------
        npt.NDArray[Any]
            The predicted values.

        """


class HomogeneousEnsembleRegressor(HomogeneousEnsemble[_ModelVar], RegressorMixin):  # pylint: disable=too-many-ancestors
    """Ensemble regressor that averages the predictions of the individual estimators."""

    estimators_: list[_ModelVar]

    @overload
    def predict(
        self,
        X: XType,  # noqa: N803
        return_std: Literal[False] = False,
        **params: Any,
    ) -> npt.NDArray[np.float64]: ...

    @overload
    def predict(
        self,
        X: XType,  # noqa: N803,  # pylint: disable=invalid-name
        return_std: Literal[True] = True,
        **params: Any,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

    @overload
    def predict(
        self,
        X: XType,  # noqa: N803,  # pylint: disable=invalid-name
        return_std: bool = False,
        **params: Any,
    ) -> (
        npt.NDArray[np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ): ...

    def predict(
        self,
        X: XType,  # noqa: N803,  # pylint: disable=invalid-name
        return_std: bool = False,
        **params: Any,
    ) -> (
        npt.NDArray[np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ):
        """Predict using the ensemble of estimators.

        Parameters
        ----------
        X : npt.NDArray | scipy.sparse.csr_matrix
            The input data.
        return_std : bool, default=False
            Whether to return the standard deviation of the predictions.
        params : Any
            Additional keyword arguments to be passed to the predict method of the
            individual estimators.

        Returns
        -------
        npt.NDArray[np.float64]
            The predicted values.

        """
        predictions = np.array(
            [estimator.predict(X, **params) for estimator in self.estimators_],  # type: ignore
        )
        if return_std:
            return np.mean(predictions, axis=0), np.std(predictions, axis=0)
        return np.mean(predictions, axis=0)


class HomogeneousEnsembleClassifier(HomogeneousEnsemble[_ModelVar], ClassifierMixin):  # pylint: disable=too-many-ancestors
    """Ensemble classifier that supports both hard and soft voting."""

    voting: Literal["hard", "soft"]

    def __init__(
        self,
        estimator: _ModelVar,
        sampler: int | BaseCrossValidator = 100,
        voting: Literal["hard", "soft"] = "hard",
        random_state: int | None = None,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the SplitEnsembleClassifier.

        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator to be cloned for each split.
        sampler: int | BaseCrossValidator, default=100
            The sampler to be used for creating the splits.
            If an int is passed, a bootstrap sample is used.
        voting: Literal["hard", "soft"], default="hard"
            The voting strategy to be used in the ensemble.
        random_state: int | None, optional
            The random state to be used for the sampler if it is an int.
        n_jobs : int, default=1
            The number of jobs to run in parallel when fitting the estimators.
        kwargs : Any
            Additional keyword arguments to be passed to the base estimator.


        """
        self.voting = voting
        super().__init__(
            estimator=estimator,
            sampler=sampler,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _can_predict_proba(self) -> bool:
        """Check if all estimators in the ensemble support probability prediction.

        Returns
        -------
        bool
            True if all estimators in the ensemble support probability prediction
            False otherwise.

        """
        return all(
            hasattr(estimator, "predict_proba") for estimator in self.estimators_
        )

    @available_if(_can_predict_proba)
    def predict_proba(
        self,
        X: XType,  # noqa: N803,  # pylint: disable=invalid-name
        **params: Any,  # pylint: disable=unused-argument
    ) -> npt.NDArray[Any]:
        """Predict class probabilities using the ensemble of estimators.

        Parameters
        ----------
        X : array-like
            The input data.
        params: Any
            Additional keyword arguments to be passed to the predict_proba method of the
            individual estimators.

        Returns
        -------
        np.ndarray
            The predicted class probabilities.

        """
        predictions = np.array(
            [estimator.predict_proba(X, **params) for estimator in self.estimators_],  # type: ignore
        )
        return np.mean(predictions, axis=0)

    def predict(
        self,
        X: XType,  # noqa: N803,  # pylint: disable=invalid-name
        **params: Any,
    ) -> npt.NDArray[Any]:
        """Predict using the ensemble of estimators.

        Parameters
        ----------
        X : array-like
            The input data.
        params: Any
            Additional keyword arguments to be passed to the predict method
            (hard voting) or predict_proba method (soft voting) of the individual
            estimators.

        Returns
        -------
        np.ndarray
            The predicted class labels.

        Raises
        ------
        AttributeError
            If voting is "soft" but not all estimators in the ensemble support
            probability prediction.
        ValueError
            If voting is "hard" but the predictions of the estimators are not integer
            values.

        """
        if self.voting == "soft":
            if not self._can_predict_proba():
                raise AttributeError(
                    "Estimators in the ensemble do not support probability prediction.",
                )
            return np.argmax(self.predict_proba(X, **params), axis=1)
        predictions = np.array(
            [estimator.predict(X, **params) for estimator in self.estimators_],  # type: ignore
        )
        if not np.issubdtype(predictions.dtype, np.integer):
            converted_predictions = predictions.astype(int)
            if not np.allclose(converted_predictions, predictions):
                raise ValueError(
                    "Predictions are not integer values, cannot perform hard voting.",
                )
            predictions = converted_predictions

        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions,
        )
