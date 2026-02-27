"""Ensemble Models where each model is trained on a different subset of the data."""

import abc
from typing import Any, Literal, Self, TypeVar

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from sklearn.utils.metaestimators import available_if
from typing_extensions import override

from molpipeline.utils.molpipeline_types import AnyPredictor

_T = TypeVar("_T", BaseEstimator, AnyPredictor)


class SplitEnsemble(abc.ABC, BaseEstimator):
    """Base class for ensemble models from sklearn splitters."""

    estimators_: list[BaseEstimator | AnyPredictor]

    def __init__(
        self,
        estimator: BaseEstimator | AnyPredictor,
        cv: int | BaseCrossValidator = 5,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the SplitEnsemble.

        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator to be cloned for each split.
        cv : int, default=5
            The spliter to be used for creating the splits.
            If an integer is provided, the splitter is a (Stratified)KFold with the
            given number of splits.
        n_jobs : int, default=1
            The number of jobs to run in parallel when fitting the estimators.
        kwargs : Any
            Additional keyword arguments to be passed to the base estimator.

        """
        self.estimator = estimator
        self.cv = cv
        self.estimators_ = []
        self.n_jobs = n_jobs
        self.set_params(**kwargs)
        super().__init__()

    @abc.abstractmethod
    def _get_splitter(self) -> BaseCrossValidator:
        """Return the splitter to be used for creating the splits.

        Returns
        -------
        BaseCrossValidator
            The splitter to be used for creating the splits.

        """

    @staticmethod
    def _fit_clone(
        model: _T,
        model_input: npt.NDArray[Any],
        y: npt.NDArray[Any],
        **kwargs: Any,
    ) -> BaseEstimator:
        """Clone the model and fit it on the given data.

        Parameters
        ----------
        model : BaseEstimator
            The model to be fitted.
        model_input : npt.NDArray[Any]
            The input data.
        y : npt.NDArray[Any]
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
        X: npt.ArrayLike,  # noqa: N803,  # pylint: disable=invalid-name
        y: npt.ArrayLike | None = None,
        groups: npt.ArrayLike | None = None,
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
        splitter = self._get_splitter()
        features = np.asarray(X)

        parallel = joblib.Parallel(n_jobs=self.n_jobs)
        fit_clone_parallel = joblib.delayed(self._fit_clone)

        self.estimators_ = parallel(
            fit_clone_parallel(
                self.estimator,
                features[train_index],
                np.asarray(y)[train_index] if y is not None else None,
                **kwargs,
            )
            for train_index, _ in splitter.split(X, y, groups)
        )
        return self

    def predict(
        self,
        X: npt.ArrayLike,  # noqa: N803,  # pylint: disable=invalid-name
    ) -> npt.NDArray[Any]:
        """Predict using the ensemble of estimators.

        Parameters
        ----------
        X : array-like
            The input data.

        Returns
        -------
        np.ndarray
            The predicted values.

        """
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])  # type: ignore
        return np.mean(predictions, axis=0)


class SplitEnsembleRegressor(SplitEnsemble, RegressorMixin):
    """SplitEnsemble for regression tasks."""

    def _get_splitter(self) -> BaseCrossValidator:
        """Return the splitter to be used for creating the splits.

        Returns
        -------
        BaseCrossValidator
            The splitter to be used for creating the splits.

        """
        cv = self.cv
        if isinstance(cv, int):
            return KFold(n_splits=cv, shuffle=True, random_state=42)
        return cv


class SplitEnsembleClassifier(SplitEnsemble, ClassifierMixin):
    """SplitEnsemble for classification tasks."""

    def __init__(
        self,
        estimator: BaseEstimator,
        cv: int | BaseCrossValidator = 5,
        voting: Literal["hard", "soft"] = "hard",
        **kwargs: Any,
    ) -> None:
        """Initialize the SplitEnsembleClassifier.

        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator to be cloned for each split.
        cv : int | BaseCrossValidator, default=5
            The spliter to be used for creating the splits.
            If an integer is provided, the splitter is a StratifiedKFold with the
            given number of splits.
        voting : Literal["hard"] | Literal["soft"], default="hard"
            Voting strategy to use for the prediction.
            The param "hard" results in majority voting.
            The param "soft" returns the class with the highest average predicted
            probability.
        kwargs : Any
            Additional keyword arguments to be passed to the base estimator.

        """
        self.voting = voting
        super().__init__(estimator=estimator, cv=cv, **kwargs)

    def _get_splitter(self) -> BaseCrossValidator:
        """Return the splitter to be used for creating the splits.

        Returns
        -------
        BaseCrossValidator
            The splitter to be used for creating the splits.

        """
        cv = self.cv
        if isinstance(cv, int):
            return StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        return cv

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
        X: npt.ArrayLike,  # noqa: N803,  # pylint: disable=invalid-name
    ) -> npt.NDArray[Any]:
        """Predict class probabilities using the ensemble of estimators.

        Parameters
        ----------
        X : array-like
            The input data.

        Returns
        -------
        np.ndarray
            The predicted class probabilities.

        """
        predictions = np.array(
            [estimator.predict_proba(X) for estimator in self.estimators_],  # type: ignore
        )
        return np.mean(predictions, axis=0)

    @override
    def predict(
        self,
        X: npt.ArrayLike,
    ) -> npt.NDArray[Any]:
        """Predict using the ensemble of estimators.

        Parameters
        ----------
        X : array-like
            The input data.

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
            return np.argmax(self.predict_proba(X), axis=1)
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])  # type: ignore
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
