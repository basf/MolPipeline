"""Conformal prediction wrappers for classification and regression using crepes."""
# pylint: disable=too-many-lines

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from crepes import WrapClassifier, WrapRegressor
from crepes.extras import DifficultyEstimator, MondrianCategorizer
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state

from molpipeline.experimental.model_selection.splitter import (
    create_continuous_stratified_folds,
)
from molpipeline.experimental.uncertainty.utils import (
    NonconformityFunctor,
    create_nonconformity_function,
)


class BaseConformalPredictor(BaseEstimator, ABC):
    """Base class for conformal predictors providing common functionality."""

    def __init__(
        self,
        estimator: BaseEstimator,
        nonconformity: str | NonconformityFunctor | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize BaseConformalPredictor.

        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator to wrap.
        nonconformity : str | NonconformityFunctor | None, optional
            Nonconformity function to use.
        **kwargs : Any
            Additional keyword arguments for configuration.

        """
        self.estimator = estimator
        self.nonconformity = nonconformity
        self.kwargs = kwargs

    @property
    def nonconformity(self) -> str | NonconformityFunctor | None:
        """Get the nonconformity parameter value.

        This property is needed for sklearn's get_params/set_params compatibility.
        """
        if self.nonconformity_func is None:
            return None
        if isinstance(self.nonconformity_func, NonconformityFunctor):
            return self.nonconformity_func.get_name()
        return self.nonconformity_func

    @nonconformity.setter
    def nonconformity(self, value: str | NonconformityFunctor | None) -> None:
        """Set the nonconformity function.

        Parameters
        ----------
        value : str | NonconformityFunctor | None
            The nonconformity function to set.
        """
        self.nonconformity_func = create_nonconformity_function(value)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Handle unpickling with backward compatibility.

        Parameters
        ----------
        state : dict[str, Any]
            The object's state dictionary.
        """
        if "nonconformity_func" not in state:
            warnings.warn(
                "Loading a model with the old 'nonconformity' attribute format. "
                "This backward compatibility will be removed in version 0.13.0. "
                "Please re-save your models with the current version.",
                DeprecationWarning,
                stacklevel=2,
            )
            fixed_state = dict(state)  # Shallow Copy
            if "nonconformity" in state:
                fixed_state["nonconformity_func"] = create_nonconformity_function(
                    fixed_state.pop("nonconformity")
                )
            else:
                fixed_state["nonconformity_func"] = None
            return super().__setstate__(fixed_state)
        return super().__setstate__(state)

    @abstractmethod
    def evaluate(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate the conformal predictor. Must be implemented by subclasses.

        Parameters
        ----------
        x: np.ndarray
            Features to evaluate.
        y: np.ndarray
            True labels/targets.
        **kwargs : Any
            Additional parameters for evaluation.

        Returns
        -------
        dict[str, Any]
            Dictionary of evaluation metrics.
        """

    @staticmethod
    def _validate_confidence_level(confidence_level: float) -> float:
        """Validate confidence level parameter.

        Parameters
        ----------
        confidence_level : float
            Confidence level to validate.

        Returns
        -------
        float
            Validated confidence level.

        Raises
        ------
        ValueError
            If confidence level is not between 0 (exclusive) and 1 (inclusive).
        """
        if not isinstance(confidence_level, (int, float)):
            raise ValueError(
                f"confidence_level must be a number, got {type(confidence_level).__name__}"
            )

        if not 0 < confidence_level <= 1:
            raise ValueError(
                f"confidence_level must be between 0 (exclusive) and 1 (inclusive), got {confidence_level}"
            )

        if confidence_level < 0.5:
            warnings.warn(
                f"Confidence level {confidence_level} is less than 0.5 (50%). "
                "This represents weak confidence and may produce unreliable prediction sets/intervals.",
                UserWarning,
                stacklevel=3,
            )

        return confidence_level

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional
            Whether to return parameters of sub-estimators (default: True).

        Returns
        -------
        dict[str, Any]
            Parameter dictionary.

        """
        params = {
            "estimator": self.estimator,
            "nonconformity": self.nonconformity,
        }

        # Get other parameters from parent class (exclude handled keys)
        parent_params = {
            k: v
            for k, v in super().get_params(deep=deep).items()
            if k not in {"estimator", "kwargs", "nonconformity_func"}
        }
        params.update(parent_params)

        params.update(self.kwargs)

        if deep and hasattr(self.estimator, "get_params"):
            estimator_params = self.estimator.get_params(deep=True)
            params.update({f"estimator__{k}": v for k, v in estimator_params.items()})

        return params

    def set_params(self, **params: Any) -> "BaseConformalPredictor":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : Any
            Parameters to set.

        Returns
        -------
        BaseConformalPredictor
            Self.

        """
        # Make a copy to avoid modifying the input dictionary
        params_copy = params.copy()

        # Convert nonconformity parameter to nonconformity_func attribute
        if "nonconformity" in params_copy:
            params_copy["nonconformity_func"] = create_nonconformity_function(
                params_copy.pop("nonconformity")  # Remove original key
            )
        super().set_params(**params_copy)
        return self


class ConformalClassifier(BaseConformalPredictor, ClassifierMixin):
    """Conformal prediction wrapper for classifiers.

    This class uses composition with crepes to provide sklearn compatibility.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        mondrian: bool = False,
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
        if not (self.nonconformity_func is None or callable(self.nonconformity_func)):
            raise TypeError(
                f"nonconformity_func must be a NonconformityFunctor or None, got {type(self.nonconformity_func).__name__}"
            )
        self._crepes_wrapper: WrapClassifier | None = None

    def fit(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        **fit_params: Any,
    ) -> "ConformalClassifier":
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
                "Internal error: _crepes_wrapper is None after initialization."
            )
        self._crepes_wrapper.fit(x, y, **fit_params)
        return self

    def calibrate(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        **calib_params: Any,
    ) -> "ConformalClassifier":
        """Calibrate the conformal classifier.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Calibration features.
        y: npt.NDArray[Any]
            Calibration targets.
        **calib_params : Any
            Additional calibration parameters.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Returns
        -------
        ConformalClassifier
            Self.

        """
        if self._crepes_wrapper is None:
            raise ValueError("Must fit before calibrating")

        kwargs = {**self.kwargs, **calib_params}

        if self.mondrian:
            kwargs.setdefault("class_cond", True)

        if self.nonconformity_func is not None:
            kwargs["nc"] = self.nonconformity_func

        self._crepes_wrapper.calibrate(x, y, **kwargs)
        return self

    def predict(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict using the conformal classifier.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Features to predict.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Returns
        -------
        npt.NDArray[Any]
            Predictions.

        """
        if self._crepes_wrapper is None:
            raise ValueError("Must fit before predicting")
        return self._crepes_wrapper.predict(x)

    def predict_proba(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict probabilities using the conformal classifier.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Features to predict.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Returns
        -------
        npt.NDArray[Any]
            Predicted probabilities.

        """
        if self._crepes_wrapper is None:
            raise ValueError("Must fit before predicting")
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

        Raises
        ------
        ValueError
            If the model has not been fitted or confidence level is invalid.

        Returns
        -------
        npt.NDArray[np.int_]
            Conformal prediction sets as binary array of shape (n_samples, n_classes).

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

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Returns
        -------
        npt.NDArray[Any]
            p-values.

        """
        if self._crepes_wrapper is None:
            raise ValueError("Must fit and calibrate before predicting")
        return self._crepes_wrapper.predict_p(x, **kwargs)

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
            If the model has not been fitted and calibrated or confidence level is invalid.

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

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        n_folds: int = 5,
        mondrian: bool = False,
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
        nonconformity : str | NonconformityFunctor | None, optional
            Nonconformity function to use for all individual classifiers.
        random_state : int | None, optional
            Random state for reproducibility (default: None).
        **kwargs : Any
            Additional keyword arguments.

        """
        super().__init__(estimator, nonconformity=nonconformity, **kwargs)
        self.n_folds = n_folds
        self.mondrian = mondrian
        self.random_state = random_state
        self.models_: list[ConformalClassifier] = []

    def fit_and_calibrate(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
    ) -> "CrossConformalClassifier":
        """Fit and calibrate the cross-conformal classifier.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Training features.
        y: npt.NDArray[Any]
            Training targets.

        Returns
        -------
        CrossConformalClassifier
            Self.

        """
        self.models_ = []
        rng = check_random_state(self.random_state)

        splitter = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=rng,
        )

        for train_idx, calib_idx in splitter.split(x, y):
            x_train, x_calib = x[train_idx], x[calib_idx]
            y_train, y_calib = y[train_idx], y[calib_idx]

            model = ConformalClassifier(
                clone(self.estimator),
                mondrian=self.mondrian,
                nonconformity=self.nonconformity_func,
                **self.kwargs,
            )
            model.fit(x_train, y_train)
            model.calibrate(x_calib, y_calib)
            self.models_.append(model)

        return self

    def predict(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict using aggregated models.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Features to predict.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Returns
        -------
        npt.NDArray[Any]
            Aggregated predictions.

        """
        if not self.models_:
            raise ValueError("Must fit before predicting")

        predictions = np.array([model.predict(x) for model in self.models_])
        return mode(predictions, axis=0, keepdims=False)[0]

    def predict_proba(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict probabilities using aggregated models.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Features to predict.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Returns
        -------
        npt.NDArray[Any]
            Aggregated probability predictions.

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

        Raises
        ------
        ValueError
            If the model has not been fitted or confidence level is invalid.

        Returns
        -------
        npt.NDArray[np.int_]
            Aggregated conformal prediction sets.

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

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Returns
        -------
        npt.NDArray[Any]
            Aggregated p-values.

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
    ) -> "ConformalRegressor":
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
                "Internal error: _crepes_wrapper is None after initialization."
            )
        self._crepes_wrapper.fit(x, y, **fit_params)
        return self

    def calibrate(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        **calib_params: Any,
    ) -> "ConformalRegressor":
        """Calibrate the conformal regressor.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Calibration features.
        y: npt.NDArray[Any]
            Calibration targets.
        **calib_params : Any
            Additional calibration parameters.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Returns
        -------
        ConformalRegressor
            Self.

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

    def predict(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict using the conformal regressor.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Features to predict.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Returns
        -------
        npt.NDArray[Any]
            Predictions.

        """
        if self._crepes_wrapper is None:
            raise ValueError("Must fit before predicting")
        return self._crepes_wrapper.predict(x)

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

        Raises
        ------
        ValueError
            If the model has not been fitted or confidence level is invalid.

        Returns
        -------
        npt.NDArray[Any]
            Prediction intervals of shape (n_samples, 2) with columns [lower, upper].

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
            If the model has not been fitted and calibrated or confidence level is invalid.

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
        super().__init__(estimator, nonconformity=nonconformity, **kwargs)
        self.n_folds = n_folds
        self.mondrian = mondrian
        self.difficulty_estimator = difficulty_estimator
        self.binning_bins = binning_bins
        self.random_state = random_state
        self.models_: list[ConformalRegressor] = []

    def fit_and_calibrate(  # pylint: disable=too-many-locals
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
    ) -> "CrossConformalRegressor":
        """Fit and calibrate the cross-conformal regressor.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Training features.
        y: npt.NDArray[Any]
            Training targets.

        Returns
        -------
        CrossConformalRegressor
            Self.

        """
        self.models_ = []

        splits = create_continuous_stratified_folds(
            y,
            n_splits=self.n_folds,
            n_groups=self.binning_bins,
            random_state=self.random_state,
        )

        x_array = np.asarray(x)
        y_array = np.asarray(y)

        for train_idx, calib_idx in splits:
            x_train, x_calib = x_array[train_idx], x_array[calib_idx]
            y_train, y_calib = y_array[train_idx], y_array[calib_idx]

            model = ConformalRegressor(
                clone(self.estimator),
                mondrian=self.mondrian,
                difficulty_estimator=self.difficulty_estimator,
                binning_bins=self.binning_bins,
                nonconformity=self.nonconformity_func,
                **self.kwargs,
            )
            model.fit(x_train, y_train)
            model.calibrate(x_calib, y_calib)
            self.models_.append(model)

        return self

    def predict(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict using aggregated models.

        Parameters
        ----------
        x: npt.NDArray[Any]
            Features to predict.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Returns
        -------
        npt.NDArray[Any]
            Aggregated predictions.

        """
        if not self.models_:
            raise ValueError("Must fit before predicting")

        predictions = np.array([model.predict(x) for model in self.models_])
        return np.mean(predictions, axis=0)

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

        Raises
        ------
        ValueError
            If the model has not been fitted or confidence level is invalid.

        Returns
        -------
        npt.NDArray[Any]
            Aggregated prediction intervals.

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
