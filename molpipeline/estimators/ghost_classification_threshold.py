"""Classification threshold adjustment with GHOST."""

from typing import Any, Self, Literal, override

import ghostml
from sklearn.utils._random import check_random_state
from sklearn.base import BaseEstimator, TransformerMixin

from molpipeline import FilterReinserter
from molpipeline.post_prediction import PostPredictionWrapper
import numpy as np
import numpy.typing as npt

from molpipeline.utils.molpipeline_types import AnyPredictor


class Ghost(BaseEstimator, TransformerMixin):
    """GHOST estimator wrapper for classification threshold adjustment.

    Applies the GHOST (Generalized tHreshOld ShifTing) algorithm to adjust the
    classification threshold. See the paper and repository for more details:

    Esposito, C., Landrum, G.A., Schneider, N., Stiefl, N. and Riniker, S., 2021.
    "GHOST: adjusting the decision threshold to handle imbalanced data in machine
     learning."
    Journal of Chemical Information and Modeling, 61(6), pp.2623-2640.
    https://doi.org/10.1021/acs.jcim.1c00160

    https://github.com/rinikerlab/GHOST

    """

    # TODO muss ich die Params fuer GHOST auf MolPipeline Seite eigentlich validieren? mach das vllt ghost besser?
    #      - Wenn ichs mache passiert es at construction time.
    #      - Dafuer muss ich die Params und validation in sync mit ghostml halten.
    def __init__(
        self,
        thresholds: list[float] | None = None,
        optimization_metric: Literal["Kappa", "ROC"] = "Kappa",
        random_state: int | None = None,
    ):
        """
        Initialize the GHOST post-prediction wrapper.

        Parameters
        ----------
        threshold : float, optional
            Classification threshold to apply, by default 0.5.

        """
        if thresholds is None:
            # use default bins from GHOST paper
            thresholds = list(np.round(np.arange(0.05, 0.55, 0.05), 2))
        self._check_thresholds(thresholds)
        self.thresholds = thresholds
        self._check_optimization_metric(optimization_metric)
        self.optimization_metric = optimization_metric
        self.random_seed = self._get_random_seed_from_input(random_state)
        self.decision_threshold: float | None = None

    @staticmethod
    def _check_optimization_metric(
        optimization_metric: Literal["Kappa", "ROC"],
    ) -> None:
        """Check if the optimization metric is valid."""
        if optimization_metric not in {"Kappa", "ROC"}:
            raise ValueError(
                "optimization_metric must be either 'Kappa' or 'ROC'",
            )

    @staticmethod
    def _get_random_seed_from_input(random_state: int | None) -> int:
        """Get a random seed from the input data.

        GHOST expects an integer random seed, so we generate one if not provided.
        """
        rng = check_random_state(random_state)
        return rng.randint(0, np.iinfo(np.int32).max)

    @staticmethod
    def _check_thresholds(thresholds: list[float]) -> None:
        """Check if the thresholds are valid."""
        if len(thresholds) == 0:
            raise ValueError("Thresholds must not be empty.")
        if not all(0 <= t <= 1 for t in thresholds):
            raise ValueError("All thresholds must be between 0 and 1.")
        if len(set(thresholds)) != len(thresholds):
            raise ValueError("Thresholds must be unique.")

    def _check_and_process_X(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Check and process the input predictions."""
        y_pred = X
        if y_pred.ndim == 2:
            # assume binary classification output when it's a 2D array
            # take class probabilities for class 1
            y_pred = y_pred[:, 1]
        if y_pred.ndim != 1:
            raise ValueError("X must be a 1D or 2D array.")
        if not np.all((y_pred >= 0) & (y_pred <= 1)):
            raise ValueError("All values in X must be between 0 and 1.")
        return y_pred

    def fit(
        self,
        X: npt.NDArray[Any],  # pylint: disable=invalid-name
        y: npt.NDArray[Any] | None = None,
    ) -> Self:
        """Fit the GHOST post-prediction wrapper.

        Prepares the decision threshold based on the predictions.

        Parameters
        ----------
        X : npt.NDArray[Any]
            Input data. The predictions.
        y : npt.NDArray[Any] | None, optional
            Target data. The true labels.

        """
        # TODO vielleicht noch nan etc auf y_pred_proba rauswerfen?

        y_pred = X
        y_true = y

        y_pred = self._check_and_process_X(y_pred)
        if y_true is None:
            raise ValueError("y must be provided for fitting the GHOST wrapper.")
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y must be binary (0 or 1).")

        self.decision_threshold = ghostml.optimize_threshold_from_predictions(
            y_true,
            y_pred,
            thresholds=self.thresholds,
            ThOpt_metrics=self.optimization_metric,
            random_seed=self.random_seed,
        )

        return self

    def transform(
        self,
        X: npt.NDArray[Any],  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.int64]:
        if self.decision_threshold is None:
            raise ValueError("Call fit first before calling transform.")

        y_pred = X
        y_pred = self._check_and_process_X(y_pred)

        # TODO vielleicht noch nan etc auf y_pred_proba rauswerfen?
        return (y_pred > self.decision_threshold).astype(np.int64)


class GhostPostPredictionWrapper(PostPredictionWrapper):
    """Post-prediction wrapper for GHOST classification threshold adjustment."""

    def __init__(
        self,
        thresholds: list[float] | None = None,
        optimization_metric: Literal["Kappa", "ROC"] = "Kappa",
        random_state: int | None = None,
    ):
        super().__init__(
            wrapped_estimator=Ghost(
                thresholds=thresholds,
                optimization_metric=optimization_metric,
                random_state=random_state,
            )
        )

    @staticmethod
    def _check_estimator(estimator: AnyPredictor) -> None:
        """Check if the wrapped estimator has a predict_proba method."""
        if not hasattr(estimator, "predict_proba"):
            raise ValueError(
                f"GHOST requires an estimator with a predict_proba method. Got: {estimator}"
            )

    @override
    def prepare_input(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        final_estimator: AnyPredictor,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Prepare input data for fitting."""
        self._check_estimator(final_estimator)
        y_pred_proba = final_estimator.predict_proba(X)
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            # binary classification, take probabilities for class 1
            y_pred_proba = y_pred_proba[:, 1]
        elif y_pred_proba.ndim != 1:
            raise ValueError("predict_proba must return a 1D or 2D array.")

        return y_pred_proba, y
