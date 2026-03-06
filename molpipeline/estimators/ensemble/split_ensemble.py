"""Ensemble Models where each model is trained on a different subset of the data."""

import abc
from collections.abc import Iterator
from typing import Any, Literal

import numpy.typing as npt
from estimators.ensemble._ensemble_base import _X, _Y
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from typing_extensions import override

from molpipeline.estimators.ensemble._ensemble_base import (
    EnsembleClassifierMixIn,
    EnsembleRegressorMixIn,
    MolPipelineBaseEnsemble,
)
from molpipeline.utils.molpipeline_types import AnyPredictor


class BaseSplitEnsemble(MolPipelineBaseEnsemble):
    """Base class for ensemble models from sklearn splitters."""

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

    @override
    def _iter_model_inputs(
        self,
        X: _X,
        y: _Y = None,
        groups: npt.ArrayLike | None = None,
    ) -> Iterator[tuple[_X, _Y]]:
        """Iterate over the model inputs for each split.

        Parameters
        ----------
        X : array-like
            The input data.
        y : array-like, optional
            The target values. Default is None.
        groups : array-like, optional
            The group labels for the samples used while splitting the dataset into
            train/test set. Default is None.

        Yields
        ------
        tuple[_X, _Y]
            The input data and target values for the current split.

        """
        splitter = self._get_splitter()
        for train_index, _ in splitter.split(X, y, groups):
            yield X[train_index], y[train_index] if y is not None else None


class SplitEnsembleRegressor(BaseSplitEnsemble, EnsembleRegressorMixIn):
    """SplitEnsemble for regression tasks."""

    @override
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


class SplitEnsembleClassifier(BaseSplitEnsemble, EnsembleClassifierMixIn):
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
