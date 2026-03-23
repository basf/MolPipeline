"""Conformal prediction wrappers for classification using crepes."""

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from crepes import WrapClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from typing_extensions import Self

from molpipeline.experimental.uncertainty.conformal.conformal_base import (
    BaseConformalPredictor,
    NonconformityFunctor,
    _apply_antitonic_regressors,
    _fit_antitonic_regressors,
)

if TYPE_CHECKING:
    from sklearn.isotonic import IsotonicRegression


class ConformalClassifier(BaseConformalPredictor, ClassifierMixin):
    """Conformal prediction wrapper for classifiers.

    This class uses composition with crepes to provide sklearn compatibility.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        mondrian: bool = False,
        calibrate_probs: bool = False,
        nonconformity: (str | NonconformityFunctor | None) = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ConformalClassifier.

        Parameters
        ----------
        estimator : BaseEstimator
            The base classifier to wrap.
        mondrian : bool, optional
            Whether to use Mondrian (class-conditional) conformal prediction
            (default: False).
        calibrate_probs : bool, default=False
            Default behavior for probability calibration during `calibrate()`.
            If True, antitonic probability calibration is applied unless
            overridden in `calibrate()`.
        nonconformity : str | NonconformityFunctor | None, optional
            Nonconformity function to use. Can be:
            - String: 'hinge', 'margin' (built-in functions)
            - NonconformityFunctor instance
            - None: Use crepes default
        **kwargs : Any
            Additional keyword arguments passed to crepes calibration.

        Raises
        ------
        TypeError
            If nonconformity_func is not a NonconformityFunctor or None.

        """
        super().__init__(estimator, nonconformity=nonconformity, **kwargs)
        self.mondrian = mondrian
        self.calibrate_probs = calibrate_probs
        if not (self.nonconformity_func is None or callable(self.nonconformity_func)):
            raise TypeError(
                f"nonconformity_func must be a NonconformityFunctor or None, got "
                f"{type(self.nonconformity_func).__name__}",
            )
        self._crepes_wrapper: WrapClassifier | None = None
        self._isotonic_regressors: list[IsotonicRegression] | None = None
        self.n_classes_: int | None = None

    def fit(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        **fit_params: Any,
    ) -> Self:
        """Fit the conformal classifier.

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
        ConformalClassifier
            Self.

        Raises
        ------
        RuntimeError
            If internal crepes wrapper initialization fails.

        """
        self._crepes_wrapper = WrapClassifier(clone(self.estimator))
        if self._crepes_wrapper is None:
            raise RuntimeError(
                "Internal error: _crepes_wrapper is None after initialization.",
            )
        self._crepes_wrapper.fit(x, y, **fit_params)
        self.n_classes_ = len(np.unique(y))
        return self

    def calibrate(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        calibrate_probs: bool | None = None,
        **calib_params: Any,
    ) -> Self:
        """Calibrate the conformal classifier.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Calibration features.
        y: npt.NDArray[Any]
            Calibration targets.
        calibrate_probs : bool | None, optional
            If True, also calibrate probabilities via antitonic mapping using
            isotonic regression. If None, uses the value configured in `__init__`
            (default: None).
        **calib_params : Any
            Additional calibration parameters.

        Returns
        -------
        ConformalClassifier
            Self.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        """
        if self._crepes_wrapper is None:
            raise ValueError("Must fit before calibrating")

        kwargs = {**self.kwargs, **calib_params}

        if self.mondrian:
            kwargs.setdefault("class_cond", True)

        if self.nonconformity_func is not None:
            kwargs["nc"] = self.nonconformity_func

        self._crepes_wrapper.calibrate(x, y, **kwargs)
        # Store number of classes
        self.n_classes_ = len(np.unique(y))

        if calibrate_probs is None:
            calibrate_probs = self.calibrate_probs

        # Optionally calibrate probabilities via isotonic regression
        if calibrate_probs:
            self._fit_isotonic_regressors(x)

        return self

    def predict_proba(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict probabilities using the conformal classifier.

        If isotonic regression calibration was performed (via calibrate_probs=True),
        returns calibrated probabilities. Otherwise returns standard probabilities.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Features to predict.

        Returns
        -------
        npt.NDArray[Any]
            Predicted probabilities (calibrated if isotonic regressors were fitted).

        Raises
        ------
        ValueError
            If the model has not been fitted.

        """
        if self._crepes_wrapper is None:
            raise ValueError("Must fit before predicting")

        # Use isotonic-calibrated probabilities if available
        if self._isotonic_regressors is not None:
            return self._apply_isotonic_calibration(x)

        return self._crepes_wrapper.predict_proba(x)

    def predict_set(
        self,
        x: npt.NDArray[Any],
        confidence: float = 0.9,
        **kwargs: Any,
    ) -> npt.NDArray[np.int_]:
        """Predict conformal sets.

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
        npt.NDArray[np.int_]
            Conformal prediction sets as binary array of shape (n_samples, n_classes).

        Raises
        ------
        ValueError
            If the model has not been fitted or confidence level is invalid.

        """
        if self._crepes_wrapper is None:
            raise ValueError("Must fit and calibrate before predicting")

        conf = self._validate_confidence_level(confidence)

        return self._crepes_wrapper.predict_set(x, confidence=conf, **kwargs)

    def predict_p(self, x: npt.NDArray[Any], **kwargs: Any) -> npt.NDArray[Any]:
        """Predict p-values.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Features to predict.
        **kwargs : Any
            Additional parameters passed to crepes.

        Returns
        -------
        npt.NDArray[Any]
            p-values.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        """
        if self._crepes_wrapper is None:
            raise ValueError("Must fit and calibrate before predicting")
        return self._crepes_wrapper.predict_p(x, **kwargs)

    def _fit_isotonic_regressors(self, x_calib: npt.NDArray[Any]) -> None:
        """Fit isotonic regressors for probability calibration.

        Parameters
        ----------
        x_calib : npt.NDArray[Any]
            Calibration features used to fit isotonic regressors.

        Raises
        ------
        ValueError
            If n_classes_ is not set.

        """
        # Get p-values for calibration data
        p_values_calib = self.predict_p(x_calib)
        epsilon = 1e-10

        if self.n_classes_ is None:
            raise ValueError("n_classes_ not set. Call calibrate() first.")

        self._isotonic_regressors = _fit_antitonic_regressors(
            p_values_calib.astype(np.float64, copy=False),
            epsilon=epsilon,
        )

    def _apply_isotonic_calibration(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Apply isotonic calibration to predictions.

        Parameters
        ----------
        x : npt.NDArray[Any]
            Test features.

        Returns
        -------
        npt.NDArray[Any]
            Calibrated probabilities.

        Raises
        ------
        ValueError
            If isotonic regressors are not fitted or n_classes_ is not set.

        """
        if self._isotonic_regressors is None:
            raise ValueError("Isotonic regressors not fitted")

        if self.n_classes_ is None:
            raise ValueError("n_classes_ not set. Call calibrate() first.")

        p_values_test = self.predict_p(x).astype(np.float64, copy=False)
        epsilon = 1e-10
        return _apply_antitonic_regressors(
            p_values_test,
            self._isotonic_regressors,
            epsilon=epsilon,
        )

    def evaluate(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        confidence: float = 0.9,
        metrics: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate conformal classifier performance.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Test features.
        y: npt.NDArray[Any]
            True test labels.
        confidence : float, optional
            Confidence level for evaluation (default: 0.9).
        metrics : list[str] | None, optional
            Metrics to compute. If None, uses default metrics.
            Available: 'error', 'avg_c', 'one_c', 'empty', 'ks_test',
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
            metrics = ["error", "avg_c", "one_c", "empty", "ks_test"]

        return self._crepes_wrapper.evaluate(
            x,
            y,
            confidence=conf,
            metrics=metrics,
            **kwargs,
        )


class CrossConformalClassifier(BaseConformalPredictor, ClassifierMixin):
    """Cross-conformal prediction wrapper for classifiers.

    This class manages multiple ConformalClassifier instances using cross-validation
    and aggregates their predictions.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        estimator: BaseEstimator,
        *,
        n_folds: int = 5,
        mondrian: bool = False,
        calibrate_probs: bool = False,
        nonconformity: (str | NonconformityFunctor | None) = None,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CrossConformalClassifier.

        Parameters
        ----------
        estimator : BaseEstimator
            The base classifier to wrap.
        n_folds : int, optional
            Number of cross-validation folds (default: 5).
        mondrian : bool, optional
            Whether to use Mondrian conformal prediction (default: False).
        calibrate_probs : bool, default=False
            Default behavior for probability calibration in each fold during
            `calibrate()`. If True, antitonic probability calibration is applied
            unless overridden in `calibrate()`.
        nonconformity : str | NonconformityFunctor | None, optional
            Nonconformity function to use for all individual classifiers.
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
        self.calibrate_probs = calibrate_probs
        self.n_classes_: int | None = None

    def fit(
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
        **fit_params : Any
            Additional parameters passed to the underlying estimator fit.

        Returns
        -------
        CrossConformalClassifier
            Self.

        Raises
        ------
        ValueError
            If n_folds is None.

        """
        self.models_ = []
        self.cv_splits_ = []
        self.n_classes_ = len(np.unique(y))

        if self.n_folds is None:
            raise ValueError("n_folds must be set before fitting.")

        splitter = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        for train_idx, calib_idx in splitter.split(x, y):
            x_train = x[train_idx]
            y_train = y[train_idx]

            model = ConformalClassifier(
                clone(self.estimator),
                mondrian=self.mondrian,
                calibrate_probs=self.calibrate_probs,
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
        calibrate_probs: bool | None = None,
        **calib_params: Any,
    ) -> Self:
        """Calibrate already-fitted cross-conformal models.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Features used to extract per-fold calibration splits.
        y: npt.NDArray[Any]
            Labels used to extract per-fold calibration splits.
        calibrate_probs : bool | None, optional
            If True, also calibrate probabilities via antitonic mapping using
            isotonic regression for each fold. If None, uses the value
            configured in `__init__` (default: None).
        **calib_params : Any
            Additional parameters passed to each fold's `ConformalClassifier.calibrate`.

        Returns
        -------
        CrossConformalClassifier
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

        if calibrate_probs is None:
            calibrate_probs = self.calibrate_probs

        for model, (_, calib_idx) in zip(self.models_, self.cv_splits_, strict=True):
            x_calib = x[calib_idx]
            y_calib = y[calib_idx]
            model.calibrate(
                x_calib,
                y_calib,
                calibrate_probs=calibrate_probs,
                **calib_params,
            )

        return self

    def predict_proba(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict probabilities using aggregated models.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Features to predict.

        Returns
        -------
        npt.NDArray[Any]
            Aggregated probability predictions.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        """
        if not self.models_:
            raise ValueError("Must fit before predicting")

        probas = np.array([model.predict_proba(x) for model in self.models_])
        return np.mean(probas, axis=0)

    def predict_set(
        self,
        x: npt.NDArray[Any],
        confidence: float = 0.9,
        **kwargs: Any,
    ) -> npt.NDArray[np.int_]:
        """Predict conformal sets using aggregated models.

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
        npt.NDArray[np.int_]
            Aggregated conformal prediction sets.

        Raises
        ------
        ValueError
            If the model has not been fitted or confidence level is invalid.

        """
        if not self.models_:
            raise ValueError("Must fit before predicting")

        conf = self._validate_confidence_level(confidence)

        p_values_list = [model.predict_p(x, **kwargs) for model in self.models_]
        aggregated_p_values = np.median(p_values_list, axis=0)

        return (aggregated_p_values >= (1 - conf)).astype(int)

    def predict_p(self, x: npt.NDArray[Any], **kwargs: Any) -> npt.NDArray[Any]:
        """Predict p-values using aggregated models.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Features to predict.
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        npt.NDArray[Any]
            Aggregated p-values.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        """
        if not self.models_:
            raise ValueError("Must fit before predicting")

        p_values_list = [model.predict_p(x, **kwargs) for model in self.models_]
        return np.median(p_values_list, axis=0)

    def evaluate(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        confidence: float = 0.9,
        metrics: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate cross-conformal classifier performance using aggregated models.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Test features.
        y: npt.NDArray[Any]
            True test labels.
        confidence : float, optional
            Confidence level for evaluation (default: 0.9).
        metrics : list[str] | None, optional
            Metrics to compute. If None, uses default metrics.
            Available: 'error', 'avg_c', 'one_c', 'empty', 'ks_test',
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
            metrics = ["error", "avg_c", "one_c", "empty", "ks_test"]

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
