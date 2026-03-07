"""The base class for molpipeline ensemble models."""

import abc
from collections.abc import Iterator
from typing import Any, Generic, Literal, Self, TypeVar, overload

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils.metaestimators import available_if

from molpipeline.utils.molpipeline_types import (
    AnyPredictor,
    XType,
    YType,
)

_T = TypeVar("_T", BaseEstimator, AnyPredictor)

_ModelVar = TypeVar("_ModelVar", bound=BaseEstimator | AnyPredictor)


class MolPipelineBaseEnsemble(abc.ABC, BaseEstimator, Generic[_ModelVar]):
    """Base class for ensemble models.

    The class is named "MolPipelineBaseEnsemble" to avoid confusion with the sklearn
    "BaseEnsemble" class, which is not compatible with the here used design.

    """

    estimator: _ModelVar
    estimators_: list[_ModelVar]

    def __init__(
        self,
        estimator: _ModelVar,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the SplitEnsemble.

        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator to be cloned for each split.
        n_jobs : int, default=1
            The number of jobs to run in parallel when fitting the estimators.
        kwargs : Any
            Additional keyword arguments to be passed to the base estimator.

        """
        self.estimator = estimator
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
        model_input : XType
            The input data.
        y : YType
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
        X : npt.ArrayLike
            The input data.
        y : npt.ArrayLikee, optional
            The target values.
        groups : npt.ArrayLike, optional
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
        parallel = joblib.Parallel(n_jobs=self.n_jobs)
        fit_clone_parallel = joblib.delayed(self._fit_clone)

        self.estimators_ = parallel(
            fit_clone_parallel(self.estimator, feat_mat, target, **kwargs)
            for feat_mat, target in self._iter_model_inputs(X, y, groups)
        )
        return self

    @abc.abstractmethod
    def _iter_model_inputs(
        self,
        X: XType,  # noqa: N803,  # pylint: disable=invalid-name
        y: YType,
        groups: npt.ArrayLike | None = None,
    ) -> Iterator[tuple[XType, YType]]:
        """Iterate over the model inputs for each estimator in the ensemble.

        Parameters
        ----------
        X : _X
            The input data.
        y : _Y, optional
            The target values.
        groups: npt.ArrayLike, optional
            Group labels for the samples used while splitting the dataset into
            the datasets for the individual estimators.

        Yields
        ------
        Iterator[tuple[_X, _Y]]
            An iterator over the model inputs for each estimator in the ensemble.

        """


class EnsembleRegressorMixIn(abc.ABC, RegressorMixin, Generic[_ModelVar]):
    """Base class for regression ensemble models."""

    estimators_: list[_ModelVar]

    @overload
    def predict(
        self,
        X: XType,  # noqa: N803,  # pylint: disable=invalid-name
        return_std: Literal[False] = False,
        **params: Any,
    ) -> npt.NDArray[np.float64]: ...

    @overload
    def predict(
        self,
        X: XType,  # noqa: N803,  # pylint: disable=invalid-name
        return_std: Literal[True],
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
        X : array-like
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


class EnsembleClassifierMixIn(abc.ABC, ClassifierMixin, Generic[_ModelVar]):
    """Base class for classification ensemble models."""

    estimators_: list[_ModelVar]
    voting: Literal["hard", "soft"]

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
