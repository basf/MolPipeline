"""Ensembles with clones of the same estimator, trained on the same data.

These might only be useful where the estimator has variability in the fit process, e.g.
due to random initialization or stochasticity in the training process.

"""

from collections.abc import Iterator
from typing import Any, Literal

import numpy.typing as npt
from typing_extensions import override

from molpipeline.estimators.ensemble._ensemble_base import (
    EnsembleClassifierMixIn,
    EnsembleRegressorMixIn,
    MolPipelineBaseEnsemble,
    _ModelVar,
)
from molpipeline.utils.molpipeline_types import (
    XVar,
    YVar,
)

__all__ = [
    "CloneEnsembleClassifier",
    "CloneEnsembleRegressor",
]


class BaseCloneEnsemble(MolPipelineBaseEnsemble[_ModelVar]):
    """Base class for ensemble models with cloned models."""

    def __init__(
        self,
        estimator: _ModelVar,
        n_estimators: int = 5,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the CloneEnsemble.

        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator to be cloned for each split.
        n_estimators : int, default=5
            The number of clones to be created.
        n_jobs : int, default=1
            The number of jobs to run in parallel when fitting the estimators.
        kwargs : Any
            Additional keyword arguments to be passed to the base estimator.

        """
        self.n_estimators = n_estimators
        super().__init__(estimator=estimator, n_jobs=n_jobs, **kwargs)

    @override
    def _iter_model_inputs(
        self,
        X: XVar,
        y: YVar,
        groups: npt.ArrayLike | None = None,
    ) -> Iterator[tuple[XVar, YVar]]:
        """Iterate over the model inputs and targets.

        Returns the same data, over and over, for each clone.

        Parameters
        ----------
        X : XVarType
            The input data.
        y : YVarType
            The target data.
        groups : npt.ArrayLike | None, optional
            The group labels for the samples, if any. Default is None.

        Yields
        ------
        XVarType
            The input data for the current clone.
        YVarType
            The target data for the current clone.

        """
        for _ in range(self.n_estimators):
            yield X, y


class CloneEnsembleClassifier(  # pylint: disable=too-many-ancestors
    EnsembleClassifierMixIn[_ModelVar],
    BaseCloneEnsemble[_ModelVar],
):
    """Ensemble classifier that creates clones of the same estimator."""

    def __init__(
        self,
        estimator: _ModelVar,
        n_estimators: int = 5,
        voting: Literal["hard", "soft"] = "hard",
        **kwargs: Any,
    ) -> None:
        """Initialize the CloneEnsembleClassifier.

        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator to be cloned for each model.
        n_estimators : int, default=5
            The number of cloned estimators.
        voting : Literal["hard", "soft"], default="hard"
            Voting strategy used during prediction.
        kwargs : Any
            Additional keyword arguments forwarded to the base class.

        """
        self.voting = voting
        super().__init__(estimator=estimator, n_estimators=n_estimators, **kwargs)


class CloneEnsembleRegressor(  # pylint: disable=too-many-ancestors
    EnsembleRegressorMixIn[_ModelVar],
    BaseCloneEnsemble[_ModelVar],
):
    """Ensemble regressor that creates clones of the same estimator."""
