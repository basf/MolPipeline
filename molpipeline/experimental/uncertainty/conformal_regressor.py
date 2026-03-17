"""Conformal prediction wrappers for regression using crepes."""
# pylint: disable=duplicate-code

from typing import Any

import numpy as np
import numpy.typing as npt
from crepes import WrapRegressor
from crepes.extras import DifficultyEstimator, MondrianCategorizer
from sklearn.base import BaseEstimator, RegressorMixin, clone
from typing_extensions import Self

from molpipeline.experimental.model_selection.splitter import (
    PercentileStratifiedKFold,
)
from molpipeline.experimental.uncertainty.conformal_base import (
    BaseConformalPredictor,
    NonconformityFunctor,
)


class ConformalRegressor(BaseConformalPredictor, RegressorMixin):
    """Conformal prediction wrapper for regressors.

    This class uses composition with crepes to provide full sklearn compatibility.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        mondrian: bool = False,
        difficulty_estimator: DifficultyEstimator | None = None,
        binning_bins: int = 10,
        nonconformity: (str | NonconformityFunctor | None) = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ConformalRegressor.

        Parameters
        ----------
        estimator : BaseEstimator
            The base regressor to wrap.
        mondrian : bool, optional
            Whether to use Mondrian conformal prediction (default: False).
        difficulty_estimator : DifficultyEstimator | None, optional
            Difficulty estimator for normalized conformal prediction.
        binning_bins : int, optional
            Number of bins for Mondrian categorization (default: 10).
        nonconformity : str | NonconformityFunctor | None, optional
            Nonconformity function to use. For regression, this is typically
            a callable that computes residual-based nonconformity scores.
        **kwargs : Any
            Additional keyword arguments passed to crepes calibration.

        """
        super().__init__(estimator, nonconformity=nonconformity, **kwargs)
        self.mondrian = mondrian
        self.difficulty_estimator = difficulty_estimator
        self.binning_bins = binning_bins
        self._crepes_wrapper: WrapRegressor | None = None

    def fit(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        **fit_params: Any,
    ) -> Self:
        """Fit the conformal regressor.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Training features.
        y: npt.NDArray[Any]
            Training targets.
        **fit_params : Any
            Additional parameters passed to estimator fit method.

        Returns
        -------
        ConformalRegressor
            Self.

        Raises
        ------
        RuntimeError
            If internal crepes wrapper initialization fails.

        """
        self._crepes_wrapper = WrapRegressor(clone(self.estimator))
        if self._crepes_wrapper is None:
            raise RuntimeError(
                "Internal error: _crepes_wrapper is None after initialization.",
            )
        self._crepes_wrapper.fit(x, y, **fit_params)
        return self

    def calibrate(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        **calib_params: Any,
    ) -> Self:
        """Calibrate the conformal regressor.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Calibration features.
        y: npt.NDArray[Any]
            Calibration targets.
        **calib_params : Any
            Additional calibration parameters.

        Returns
        -------
        ConformalRegressor
            Self.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        """
        if self._crepes_wrapper is None:
            raise ValueError("Must fit before calibrating")

        kwargs = {**self.kwargs, **calib_params}

        if self.difficulty_estimator is not None:
            kwargs["de"] = self.difficulty_estimator

        if self.mondrian:
            mc = MondrianCategorizer()
            mc.fit(x, learner=self._crepes_wrapper.learner, no_bins=self.binning_bins)
            kwargs["mc"] = mc

        if self.nonconformity_func is not None:
            kwargs["nc"] = self.nonconformity_func

        self._crepes_wrapper.calibrate(x, y, **kwargs)
        return self

    def predict_int(
        self,
        x: npt.NDArray[Any],
        confidence: float = 0.9,
        **kwargs: Any,
    ) -> npt.NDArray[Any]:
        """Predict intervals.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Features to predict.
        confidence : float, optional
            Confidence level (default: 0.9).
        **kwargs : Any
            Additional parameters passed to crepes.

        Returns
        -------
        npt.NDArray[Any]
            Prediction intervals of shape (n_samples, 2) with columns [lower, upper].

        Raises
        ------
        ValueError
            If the model has not been fitted or confidence level is invalid.

        """
        if self._crepes_wrapper is None:
            raise ValueError("Must fit and calibrate before predicting")

        conf = self._validate_confidence_level(confidence)

        return self._crepes_wrapper.predict_int(x, confidence=conf, **kwargs)

    def evaluate(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        confidence: float = 0.9,
        metrics: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate conformal regressor performance.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Test features.
        y: npt.NDArray[Any]
            True test targets.
        confidence : float, optional
            Confidence level for evaluation (default: 0.9).
        metrics : list[str] | None, optional
            Metrics to compute. If None, uses default metrics.
            Available: 'error', 'eff_mean', 'eff_med', 'ks_test',
                       'time_fit', 'time_evaluate'.
        **kwargs : Any
            Additional parameters passed to crepes evaluate method.

        Returns
        -------
        dict[str, Any]
            Dictionary containing evaluation metrics.

        Raises
        ------
        ValueError
            If the model has not been fitted and calibrated or confidence level is
            invalid.

        """
        if self._crepes_wrapper is None:
            raise ValueError("Must fit and calibrate before evaluating")

        conf = self._validate_confidence_level(confidence)

        if metrics is None:
            metrics = ["error", "eff_mean", "eff_med", "ks_test"]

        return self._crepes_wrapper.evaluate(
            x,
            y,
            confidence=conf,
            metrics=metrics,
            **kwargs,
        )


class CrossConformalRegressor(BaseConformalPredictor, RegressorMixin):
    """Cross-conformal prediction wrapper for regressors.

    This class manages multiple ConformalRegressor instances using cross-validation
    and aggregates their predictions.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        n_folds: int = 5,
        mondrian: bool = False,
        difficulty_estimator: DifficultyEstimator | None = None,
        binning_bins: int = 10,
        nonconformity: (str | NonconformityFunctor | None) = None,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CrossConformalRegressor.

        Parameters
        ----------
        estimator : BaseEstimator
            The base regressor to wrap.
        n_folds : int, optional
            Number of cross-validation folds (default: 5).
        mondrian : bool, optional
            Whether to use Mondrian conformal prediction (default: False).
        difficulty_estimator : DifficultyEstimator | None, optional
            Difficulty estimator for normalized conformal prediction.
        binning_bins : int, optional
            Number of bins for Mondrian categorization (default: 10).
        nonconformity : str | NonconformityFunctor | None, optional
            Nonconformity function to use for all individual regressors.
        random_state : int | None, optional
            Random state for reproducibility (default: None).
        **kwargs : Any
            Additional keyword arguments.

        """
        super().__init__(
            estimator,
            nonconformity=nonconformity,
            n_folds=n_folds,
            mondrian=mondrian,
            random_state=random_state,
            **kwargs,
        )
        self.difficulty_estimator = difficulty_estimator
        self.binning_bins = binning_bins

    def fit(  # pylint: disable=too-many-locals
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        **fit_params: Any,
    ) -> Self:
        """Fit cross-conformal models (without calibration).

        This method trains one underlying estimator per fold on the respective
        training split and stores the fold indices for later calibration.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Training features.
        y: npt.NDArray[Any]
            Training targets.
        fit_params : Any
            Additional parameters passed to the underlying estimator fit.

        Returns
        -------
        CrossConformalRegressor
            Self.

        """
        self.models_ = []
        self.cv_splits_ = []

        y_array = np.asarray(y)
        splitter = PercentileStratifiedKFold(
            n_splits=self.n_folds,
            n_groups=self.binning_bins,
            shuffle=True,
            random_state=self.random_state,
        )

        x_array = np.asarray(x)
        splits = splitter.split(X=np.zeros(len(y_array)), y=y_array)

        for train_idx, calib_idx in splits:
            x_train = x_array[train_idx]
            y_train = y_array[train_idx]

            model = ConformalRegressor(
                clone(self.estimator),
                mondrian=self.mondrian,
                difficulty_estimator=self.difficulty_estimator,
                binning_bins=self.binning_bins,
                nonconformity=self.nonconformity_func,
                **self.kwargs,
            )
            model.fit(x_train, y_train, **fit_params)
            self.models_.append(model)
            self.cv_splits_.append((train_idx, calib_idx))

        return self

    def calibrate(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        **calib_params: Any,
    ) -> Self:
        """Calibrate already-fitted cross-conformal models.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Features used to extract per-fold calibration splits.
        y: npt.NDArray[Any]
            Targets used to extract per-fold calibration splits.
        **calib_params : Any
            Additional parameters passed to each fold's `ConformalRegressor.calibrate`.

        Returns
        -------
        CrossConformalRegressor
            Self.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        """
        if not self.models_ or not self.cv_splits_:
            raise ValueError("Must fit before calibrating")

        if len(self.models_) != len(self.cv_splits_):
            raise ValueError("Internal error: models_ and cv_splits_ mismatch")

        x_array = np.asarray(x)
        y_array = np.asarray(y)

        for model, (_, calib_idx) in zip(self.models_, self.cv_splits_, strict=True):
            x_calib = x_array[calib_idx]
            y_calib = y_array[calib_idx]
            model.calibrate(x_calib, y_calib, **calib_params)

        return self

    def predict_int(
        self,
        x: npt.NDArray[Any],
        confidence: float = 0.9,
        **kwargs: Any,
    ) -> npt.NDArray[Any]:
        """Predict intervals using aggregated models.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Features to predict.
        confidence : float, optional
            Confidence level (default: 0.9).
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        npt.NDArray[Any]
            Aggregated prediction intervals.

        Raises
        ------
        ValueError
            If the model has not been fitted or confidence level is invalid.

        """
        if not self.models_:
            raise ValueError("Must fit before predicting")

        conf = self._validate_confidence_level(confidence)

        intervals_list = [
            model.predict_int(x, confidence=conf, **kwargs) for model in self.models_
        ]
        return np.mean(intervals_list, axis=0)

    def evaluate(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        confidence: float = 0.9,
        metrics: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate cross-conformal regressor performance using aggregated models.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Test features.
        y: npt.NDArray[Any]
            True test targets.
        confidence : float, optional
            Confidence level for evaluation (default: 0.9).
        metrics : list[str] | None, optional
            Metrics to compute. If None, uses default metrics.
            Available: 'error', 'eff_mean', 'eff_med', 'ks_test',
                       'time_fit', 'time_evaluate'.
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing aggregated evaluation metrics.

        Raises
        ------
        ValueError
            If the model has not been fitted or confidence level is invalid.

        """
        if not self.models_:
            raise ValueError("Must fit before evaluating")

        conf = self._validate_confidence_level(confidence)

        if metrics is None:
            metrics = ["error", "eff_mean", "eff_med", "ks_test"]

        # Aggregate results from all models
        all_results = []
        for model in self.models_:
            result = model.evaluate(x, y, confidence=conf, metrics=metrics, **kwargs)
            all_results.append(result)

        # Compute mean and std of metrics across models
        aggregated_results = {}
        for metric in metrics:
            values = [result[metric] for result in all_results if metric in result]
            if values:
                aggregated_results[f"{metric}_mean"] = float(np.mean(values))
                aggregated_results[f"{metric}_std"] = float(np.std(values))

        return aggregated_results
