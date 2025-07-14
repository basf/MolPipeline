"""Conformal prediction wrappers for classification and regression using crepes.

Provides unified and cross-conformal prediction with Mondrian and nonconformity options.
"""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from crepes import WrapClassifier, WrapRegressor
from crepes.extras import DifficultyEstimator, MondrianCategorizer
from scipy.stats import mode
from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import check_random_state


def _bin_targets(y: npt.NDArray[Any], n_bins: int = 10) -> npt.NDArray[np.int_]:
    """Bin continuous targets for stratified splitting in regression.

    Parameters
    ----------
    y : npt.NDArray[Any]
        Target values.
    n_bins : int, optional
        Number of bins (default: 10).

    Returns
    -------
    npt.NDArray[np.int_]
        Binned targets.

    """
    y = np.asarray(y)
    bins = np.linspace(np.min(y), np.max(y), n_bins + 1)
    y_binned = np.digitize(y, bins) - 1  # bins start at 1
    y_binned[y_binned == n_bins] = n_bins - 1  # edge case
    return y_binned


def _detect_estimator_type(
    estimator: BaseEstimator,
) -> Literal["classifier", "regressor"]:
    """Automatically detect whether an estimator is a classifier or regressor.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to check.

    Returns
    -------
    Literal["classifier", "regressor"]
        The detected estimator type.

    Raises
    ------
    ValueError
        If the estimator type cannot be determined.

    """
    if is_classifier(estimator):
        return "classifier"
    if is_regressor(estimator):
        return "regressor"
    raise ValueError(
        f"Could not determine if {type(estimator).__name__} is a "
        "classifier or regressor. Please specify estimator_type explicitly.",
    )


def _get_mondrian_param_classification(
    mondrian: MondrianCategorizer | Callable[..., Any] | bool,
    y_calib: npt.NDArray[Any],
) -> MondrianCategorizer | Callable[..., Any] | npt.NDArray[Any] | None:
    """Get mondrian parameter for classification calibration.

    Returns
    -------
    MondrianCategorizer | Callable[..., Any] | npt.NDArray[Any] | None
        Mondrian parameter for classification calibration.

    """
    if isinstance(mondrian, MondrianCategorizer) or callable(mondrian):
        return mondrian
    if mondrian is True:
        return y_calib
    return None


def _get_mondrian_param_regression(
    mondrian: MondrianCategorizer | Callable[..., Any] | bool,
) -> MondrianCategorizer | None:
    """Get mondrian parameter for regression calibration.

    Returns
    -------
    MondrianCategorizer | None
        Mondrian parameter for regression calibration.

    """
    if isinstance(mondrian, MondrianCategorizer) or callable(mondrian):
        return mondrian
    return None


class ConformalPredictor(BaseEstimator):  # pylint: disable=too-many-instance-attributes
    """Conformal prediction wrapper for both classifiers and regressors.

    Uses crepes under the hood.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        mondrian: MondrianCategorizer | Callable[..., Any] | bool = False,
        confidence_level: float = 0.9,
        estimator_type: Literal["classifier", "regressor", "auto"] = "auto",
        nonconformity: (
            Callable[
                [npt.NDArray[Any], npt.NDArray[Any] | None, npt.NDArray[Any] | None],
                npt.NDArray[Any],
            ]
            | None
        ) = None,
        difficulty_estimator: DifficultyEstimator | None = None,
        binning: int | MondrianCategorizer | None = None,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize ConformalPredictor.

        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator or pipeline to wrap.
        mondrian : MondrianCategorizer | Callable[..., Any] | bool, optional
            Mondrian calibration/grouping (default: False).
        confidence_level : float, optional
            Confidence level for prediction sets/intervals (default: 0.9).
        estimator_type : Literal["classifier", "regressor", "auto"], optional
            Type of estimator: 'classifier', 'regressor', or 'auto' to
            detect automatically (default: 'auto').
        nonconformity : Callable, optional
            Nonconformity function for classification that takes (X_prob, classes, y)
            and returns non-conformity scores. Examples: hinge, margin from
            crepes.extras.
        difficulty_estimator : DifficultyEstimator | None, optional
            Difficulty estimator for normalized conformal prediction (regression).
            Should be a fitted DifficultyEstimator from crepes.extras.
        binning : int | MondrianCategorizer | None, optional
            Number of bins or MondrianCategorizer for Mondrian calibration (regression).
        n_jobs : int, optional
            Number of parallel jobs (default: 1).
        **kwargs : Any
            Additional keyword arguments for crepes.

        """
        self.estimator = estimator
        self.mondrian = mondrian
        self.confidence_level = confidence_level
        if estimator_type == "auto":
            self.estimator_type = _detect_estimator_type(estimator)
        else:
            self.estimator_type = estimator_type
        self.nonconformity = nonconformity
        self.difficulty_estimator = difficulty_estimator
        self.binning = binning
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self._conformal: WrapClassifier | WrapRegressor | None = None
        self.fitted_ = False
        self.calibrated_ = False

    def fit(self, x: npt.NDArray[Any], y: npt.NDArray[Any]) -> "ConformalPredictor":
        """Fit the conformal predictor.

        Parameters
        ----------
        x : npt.NDArray[Any]
            Training features.
        y : npt.NDArray[Any]
            Training targets.

        Returns
        -------
        ConformalPredictor
            Self.

        Raises
        ------
        ValueError
            If estimator_type is not 'classifier' or 'regressor'.

        """
        if self.estimator_type == "classifier":
            self._conformal = WrapClassifier(clone(self.estimator))
        elif self.estimator_type == "regressor":
            self._conformal = WrapRegressor(clone(self.estimator))
        else:
            raise ValueError("estimator_type must be 'classifier' or 'regressor'")

        self._conformal.fit(x, y)
        self.fitted_ = True
        return self

    def calibrate(
        self,
        x_calib: npt.NDArray[Any],
        y_calib: npt.NDArray[Any],
        **calib_params: Any,
    ) -> None:
        """Calibrate the conformal predictor.

        Parameters
        ----------
        x_calib : npt.NDArray[Any]
            Calibration features.
        y_calib : npt.NDArray[Any]
            Calibration targets.
        calib_params : dict
            Additional calibration parameters.

        Raises
        ------
        ValueError
            If estimator_type is not 'classifier' or 'regressor'.
        RuntimeError
            If the estimator must be fitted before calling calibrate.

        """
        if not self.fitted_ or self._conformal is None:
            raise RuntimeError("Estimator must be fitted before calling calibrate")
        if self.estimator_type == "classifier":
            if self.mondrian is True:
                self._conformal.calibrate(
                    x_calib,
                    y_calib,
                    class_cond=True,
                    **calib_params,
                )
            elif isinstance(
                self.mondrian,
                (MondrianCategorizer, type(lambda: None)),
            ) and callable(self.mondrian):
                self._conformal.calibrate(
                    x_calib,
                    y_calib,
                    mc=self.mondrian,
                    **calib_params,
                )
            else:
                self._conformal.calibrate(x_calib, y_calib, **calib_params)
        elif self.estimator_type == "regressor":
            mc = _get_mondrian_param_regression(self.mondrian)
            self._conformal.calibrate(x_calib, y_calib, mc=mc, **calib_params)
        else:
            raise ValueError("estimator_type must be 'classifier' or 'regressor'")
        self.calibrated_ = True

    def predict(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict using the conformal predictor.

        Parameters
        ----------
        x : npt.NDArray[Any]
            Features to predict.

        Returns
        -------
        npt.NDArray[Any]
            Predictions.

        Raises
        ------
        ValueError
            If estimator must be fitted before calling predict.

        """
        if not self.fitted_ or self._conformal is None:
            raise ValueError("Estimator must be fitted before calling predict")
        return self._conformal.predict(x)

    def predict_proba(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict probabilities using the conformal predictor.

        Parameters
        ----------
        x : npt.NDArray[Any]
            Features to predict.

        Returns
        -------
        npt.NDArray[Any]
            Predicted probabilities.

        Raises
        ------
        ValueError
            If estimator must be fitted before calling predict_proba.
        NotImplementedError
            If called for a regressor.
        RuntimeError
            If the internal conformal wrapper is not of the expected type.

        """
        if not self.fitted_ or self._conformal is None:
            raise ValueError("Estimator must be fitted before calling predict_proba")
        if self.estimator_type != "classifier":
            raise NotImplementedError("predict_proba is for classifiers only.")
        if isinstance(self._conformal, WrapClassifier):
            return self._conformal.predict_proba(x)
        raise RuntimeError("Expected WrapClassifier but got different type")

    def predict_conformal_set(
        self,
        x: npt.NDArray[Any],
        confidence: float | None = None,
    ) -> list[list[int]]:
        """Predict conformal sets.

        Parameters
        ----------
        x : npt.NDArray[Any]
            Features to predict.
        confidence : float, optional
            Confidence level.

        Returns
        -------
        list[list[int]]
            Conformal prediction sets as list of lists containing class indices.

        Raises
        ------
        ValueError
            If estimator must be fitted before calling predict_conformal_set.
        NotImplementedError
            If called for a regressor.
        RuntimeError
            If the internal conformal wrapper is not of the expected type.

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must be fitted before calling predict_conformal_set",
            )
        if self._conformal is None:
            raise RuntimeError("Conformal wrapper is not initialized")
        if self.estimator_type != "classifier":
            raise NotImplementedError(
                "predict_conformal_set is only for classification.",
            )
        conf = confidence if confidence is not None else self.confidence_level
        if isinstance(self._conformal, WrapClassifier):
            prediction_sets_binary = self._conformal.predict_set(x, confidence=conf)

            prediction_sets = []
            for i in range(prediction_sets_binary.shape[0]):
                class_indices = [
                    j
                    for j in range(prediction_sets_binary.shape[1])
                    if prediction_sets_binary[i, j] == 1
                ]
                prediction_sets.append(class_indices)

            return prediction_sets
        raise RuntimeError("Expected WrapClassifier but got different type")

    def predict_p(self, x: npt.NDArray[Any], **kwargs: Any) -> npt.NDArray[Any]:
        """Predict p-values.

        Parameters
        ----------
        x : npt.NDArray[Any]
            Features to predict.
        kwargs : dict
            Additional parameters.

        Returns
        -------
        npt.NDArray[Any]
            p-values.

        Raises
        ------
        ValueError
            If estimator must be fitted before calling predict_p.
        NotImplementedError
            If called for a regressor.
        RuntimeError
            If the internal conformal wrapper is not of the expected type.

        """
        if not self.fitted_:
            raise ValueError("Estimator must be fitted before calling predict_p")
        if self._conformal is None:
            raise RuntimeError("Conformal wrapper is not initialized")
        if self.estimator_type != "classifier":
            raise NotImplementedError("predict_p is only for classification.")
        if isinstance(self._conformal, WrapClassifier):
            return self._conformal.predict_p(x, **kwargs)
        raise RuntimeError("Expected WrapClassifier but got different type")

    def predict_int(
        self,
        x: npt.NDArray[Any],
        confidence: float | None = None,
    ) -> npt.NDArray[Any]:
        """Predict intervals.

        Parameters
        ----------
        x : npt.NDArray[Any]
            Features to predict.
        confidence : float, optional
            Confidence level.

        Returns
        -------
        npt.NDArray[Any]
            Prediction intervals of shape (n_samples, 2) with columns [lower, upper].

        Raises
        ------
        ValueError
            If estimator must be fitted before calling predict_int.
        NotImplementedError
            If called for a classifier.
        RuntimeError
            If the internal conformal wrapper is not of the expected type.

        """
        if not self.fitted_:
            raise ValueError("Estimator must be fitted before calling predict_int")
        if self._conformal is None:
            raise RuntimeError("Conformal wrapper is not initialized")
        if self.estimator_type != "regressor":
            raise NotImplementedError("predict_int is only for regression.")
        conf = confidence if confidence is not None else self.confidence_level
        if isinstance(self._conformal, WrapRegressor):
            return self._conformal.predict_int(x, confidence=conf)
        raise RuntimeError("Expected WrapRegressor but got different type")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        dict[str, Any]
            Parameter names mapped to their values.

        """
        params = {
            "estimator": self.estimator,
            "mondrian": self.mondrian,
            "confidence_level": self.confidence_level,
            "estimator_type": self.estimator_type,
            "nonconformity": self.nonconformity,
            "difficulty_estimator": self.difficulty_estimator,
            "binning": self.binning,
            "n_jobs": self.n_jobs,
        }
        params.update(self.kwargs)

        if deep and hasattr(self.estimator, "get_params"):
            estimator_params = self.estimator.get_params(deep=True)
            params.update({f"estimator__{k}": v for k, v in estimator_params.items()})

        return params

    def set_params(self, **params: Any) -> "ConformalPredictor":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        ConformalPredictor
            This estimator.

        Raises
        ------
        ValueError

        """
        valid_params = self.get_params(deep=False)
        estimator_params: dict[str, Any] = {}

        for key, value in params.items():
            if key in valid_params:
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {type(self).__name__}",
                )

        if estimator_params and hasattr(self.estimator, "set_params"):
            self.estimator.set_params(**estimator_params)

        return self


class CrossConformalPredictor(BaseEstimator):  # pylint: disable=too-many-instance-attributes
    """Cross-conformal prediction using WrapClassifier/WrapRegressor."""

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        n_folds: int = 5,
        confidence_level: float = 0.9,
        mondrian: MondrianCategorizer | Callable[..., Any] | bool = False,
        nonconformity: (
            Callable[
                [npt.NDArray[Any], npt.NDArray[Any] | None, npt.NDArray[Any] | None],
                npt.NDArray[Any],
            ]
            | None
        ) = None,
        binning: int | MondrianCategorizer | None = None,
        estimator_type: Literal["classifier", "regressor", "auto"] = "auto",
        n_bins: int = 10,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CrossConformalPredictor.

        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator or pipeline to wrap.
        n_folds : int, optional
            Number of cross-validation folds (default: 5).
        confidence_level : float, optional
            Confidence level for prediction sets/intervals (default: 0.9).
        mondrian : MondrianCategorizer | Callable[..., Any] | bool, optional
            Mondrian calibration/grouping (default: False).
        nonconformity : Callable, optional
            Nonconformity function for classification that takes (X_prob, classes, y)
            and returns non-conformity scores. Examples: hinge, margin from
            crepes.extras.
        binning : int | MondrianCategorizer | None, optional
            Number of bins or MondrianCategorizer for Mondrian calibration (regression).
        estimator_type : Literal["classifier", "regressor", "auto"], optional
            Auto detects it automatically (default: 'auto').
        n_bins : int, optional
            Number of bins for stratified splitting in regression (default: 10).
        random_state : int | None, optional
            Random state for reproducibility.
        **kwargs : Any
            Additional keyword arguments for crepes.

        """
        self.estimator = estimator
        self.n_folds = n_folds
        self.confidence_level = confidence_level
        self.mondrian = mondrian
        self.nonconformity = nonconformity
        self.binning = binning
        if estimator_type == "auto":
            self.estimator_type = _detect_estimator_type(estimator)
        else:
            self.estimator_type = estimator_type
        self.n_bins = n_bins
        self.random_state = random_state  # Store the original seed/state
        self.kwargs = kwargs
        self.models_: list[WrapClassifier | WrapRegressor] = []
        self.fitted_ = False

    def _create_splitter(
        self,
        y: npt.NDArray[Any],
        rng: Any,
    ) -> tuple[KFold | StratifiedKFold, npt.NDArray[Any]]:
        """Create the appropriate splitter for cross-validation.

        Parameters
        ----------
        y : npt.NDArray[Any]
            Target values.
        rng : Any
            Random state object.

        Returns
        -------
        tuple[KFold | StratifiedKFold, npt.NDArray[Any]]
            Splitter and y values for splitting.

        Raises
        ------
        ValueError
            If estimator_type is not 'classifier' or 'regressor'.

        """
        if self.estimator_type == "classifier":
            splitter = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=rng,
            )
            y_split = y
        elif self.estimator_type == "regressor":
            splitter = KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=rng,
            )
            y_split = _bin_targets(y, n_bins=self.n_bins)
        else:
            raise ValueError("estimator_type must be 'classifier' or 'regressor'")
        return splitter, y_split

    def _create_mondrian_categorizer(
        self,
        model: WrapRegressor,
        y_calib_vals: npt.NDArray[Any],
    ) -> tuple[MondrianCategorizer, Callable[..., Any]]:
        """Create a MondrianCategorizer for regression binning.

        Parameters
        ----------
        model : WrapRegressor
            The fitted regression model.
        y_calib_vals : npt.NDArray[Any]
            Calibration target values.

        Returns
        -------
        tuple[MondrianCategorizer, Callable[..., Any]]
            Fitted MondrianCategorizer and binning function.

        """
        mc_obj = MondrianCategorizer()
        y_min, y_max = np.min(y_calib_vals), np.max(y_calib_vals)
        n_bins = self.binning

        def bin_func(
            x_test: Any,
            model: Any = model,
            y_min: Any = y_min,
            y_max: Any = y_max,
            n_bins: Any = n_bins,
        ) -> Any:
            y_pred = model.predict(x_test)
            bins = np.linspace(y_min, y_max, n_bins + 1)
            binned = np.digitize(y_pred, bins) - 1
            return np.clip(binned, 0, n_bins - 1)

        return mc_obj, bin_func

    def _fit_single_model(
        self,
        x_array: npt.NDArray[Any],
        y_array: npt.NDArray[Any],
        train_idx: npt.NDArray[np.int_],
        calib_idx: npt.NDArray[np.int_],
    ) -> WrapClassifier | WrapRegressor:
        """Fit and calibrate a single model for one fold.

        Parameters
        ----------
        x_array : npt.NDArray[Any]
            Feature array.
        y_array : npt.NDArray[Any]
            Target array.
        train_idx : npt.NDArray[np.int_]
            Training indices.
        calib_idx : npt.NDArray[np.int_]
            Calibration indices.

        Returns
        -------
        WrapClassifier | WrapRegressor
            Fitted and calibrated model.

        """
        if self.estimator_type == "classifier":
            model = WrapClassifier(clone(self.estimator))
            model.fit(x_array[train_idx], y_array[train_idx])

            if self.mondrian is True:
                model.calibrate(x_array[calib_idx], y_array[calib_idx], class_cond=True)
            elif isinstance(
                self.mondrian,
                (MondrianCategorizer, type(lambda: None)),
            ) and callable(self.mondrian):
                model.calibrate(
                    x_array[calib_idx],
                    y_array[calib_idx],
                    mc=self.mondrian,
                )
            else:
                model.calibrate(x_array[calib_idx], y_array[calib_idx])
        else:
            model = WrapRegressor(clone(self.estimator))
            model.fit(x_array[train_idx], y_array[train_idx])
            mc = _get_mondrian_param_regression(self.mondrian)
            if self.binning is not None and isinstance(self.binning, int):
                mc_obj, bin_func = self._create_mondrian_categorizer(
                    model,
                    y_array[calib_idx],
                )
                mc_obj.fit(x_array[calib_idx], f=bin_func, no_bins=self.binning)
                mc = mc_obj
            elif self.binning is not None:
                mc = self.binning
            model.calibrate(x_array[calib_idx], y_array[calib_idx], mc=mc)
        return model

    def fit(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
    ) -> "CrossConformalPredictor":
        """Fit the cross-conformal predictor.

        Parameters
        ----------
        x : npt.NDArray[Any]
            Training features.
        y : npt.NDArray[Any]
            Training targets.

        Returns
        -------
        CrossConformalPredictor
            Self.

        """
        self.models_ = []
        rng = check_random_state(self.random_state)
        splitter, y_split = self._create_splitter(y, rng)

        x_array = np.asarray(x)
        y_array = np.asarray(y)

        for train_idx, calib_idx in splitter.split(x_array, y_split):
            model = self._fit_single_model(x_array, y_array, train_idx, calib_idx)
            self.models_.append(model)

        self.fitted_ = True
        return self

    def predict(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict using the cross-conformal predictor.

        Parameters
        ----------
        x : npt.NDArray[Any]
            Features to predict.

        Returns
        -------
        npt.NDArray[Any]
            Predictions (majority vote).

        Raises
        ------
        ValueError
            If estimator must be fitted before calling predict.

        """
        if not self.fitted_:
            raise ValueError("Estimator must be fitted before calling predict")
        result = np.array([m.predict(x) for m in self.models_])
        pred_mode = mode(result, axis=0, keepdims=False)
        return np.ravel(pred_mode.mode)

    def predict_proba(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict probabilities using the cross-conformal predictor.

        Parameters
        ----------
        x : npt.NDArray[Any]
            Features to predict.

        Returns
        -------
        npt.NDArray[Any]
            Predicted probabilities (averaged).

        Raises
        ------
        ValueError
            If estimator must be fitted before calling predict_proba.
        NotImplementedError
            If called for a regressor.

        """
        if not self.fitted_:
            raise ValueError("Estimator must be fitted before calling predict_proba")
        if self.estimator_type != "classifier":
            raise NotImplementedError("predict_proba is for classifiers only.")
        result = np.array([m.predict_proba(x) for m in self.models_])
        return np.atleast_2d(np.mean(result, axis=0))

    def predict_conformal_set(
        self,
        x: npt.NDArray[Any],
        confidence: float | None = None,
    ) -> list[list[int]]:
        """Predict conformal sets using the cross-conformal predictor.

        Parameters
        ----------
        x : npt.NDArray[Any]
            Features to predict.
        confidence : float, optional
            Confidence level.

        Returns
        -------
        list[list[int]]
            Conformal prediction sets as list of lists containing class indices.

        Raises
        ------
        ValueError
            If estimator must be fitted before calling predict_conformal_set.
        NotImplementedError
            If called for a regressor.

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must be fitted before calling predict_conformal_set",
            )
        if self.estimator_type != "classifier":
            raise NotImplementedError(
                "predict_conformal_set is only for classification.",
            )
        conf = confidence if confidence is not None else self.confidence_level

        p_values_list = [m.predict_p(x) for m in self.models_]
        aggregated_p_values = np.mean(p_values_list, axis=0)

        prediction_sets_binary = (aggregated_p_values >= 1 - conf).astype(int)

        prediction_sets = []
        for i in range(prediction_sets_binary.shape[0]):
            class_indices = [
                j
                for j in range(prediction_sets_binary.shape[1])
                if prediction_sets_binary[i, j] == 1
            ]
            prediction_sets.append(class_indices)

        return prediction_sets

    def predict_p(self, x: npt.NDArray[Any], **kwargs: Any) -> npt.NDArray[Any]:
        """Predict p-values using the cross-conformal predictor.

        Parameters
        ----------
        x : npt.NDArray[Any]
            Features to predict.
        kwargs : dict
            Additional parameters.

        Returns
        -------
        npt.NDArray[Any]
            Aggregated p-values from all folds.

        Raises
        ------
        ValueError
            If estimator must be fitted before calling predict_p.
        NotImplementedError
            If called for a regressor.

        """
        if not self.fitted_:
            raise ValueError("Estimator must be fitted before calling predict_p")
        if self.estimator_type != "classifier":
            raise NotImplementedError("predict_p is only for classification.")

        p_values_list = [m.predict_p(x, **kwargs) for m in self.models_]
        return np.mean(p_values_list, axis=0)

    def predict_int(
        self,
        x: npt.NDArray[Any],
        confidence: float | None = None,
    ) -> npt.NDArray[Any]:
        """Predict intervals using the cross-conformal predictor.

        Parameters
        ----------
        x : npt.NDArray[Any]
            Features to predict.
        confidence : float, optional
            Confidence level.

        Returns
        -------
        npt.NDArray[Any]
            Prediction intervals based on aggregated predictions.

        Raises
        ------
        ValueError
            If estimator must be fitted before calling predict_int.
        NotImplementedError
            If called for a classifier.

        """
        if not self.fitted_:
            raise ValueError("Estimator must be fitted before calling predict_int")
        if self.estimator_type != "regressor":
            raise NotImplementedError("predict_int is only for regression.")

        conf = confidence if confidence is not None else self.confidence_level

        intervals_list = [m.predict_int(x, confidence=conf) for m in self.models_]

        intervals_array = np.array(intervals_list)  # shape: (n_folds, n_samples, 2)
        lower_bounds = np.nanmean(intervals_array[:, :, 0], axis=0)
        upper_bounds = np.nanmean(intervals_array[:, :, 1], axis=0)

        return np.column_stack([lower_bounds, upper_bounds])

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        dict[str, Any]
            Parameter names mapped to their values.

        """
        params = {
            "estimator": self.estimator,
            "n_folds": self.n_folds,
            "confidence_level": self.confidence_level,
            "mondrian": self.mondrian,
            "nonconformity": self.nonconformity,
            "binning": self.binning,
            "estimator_type": self.estimator_type,
            "n_bins": self.n_bins,
            "random_state": self.random_state,
        }
        params.update(self.kwargs)

        if deep and hasattr(self.estimator, "get_params"):
            estimator_params = self.estimator.get_params(deep=True)
            params.update({f"estimator__{k}": v for k, v in estimator_params.items()})

        return params

    def set_params(self, **params: Any) -> "CrossConformalPredictor":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        CrossConformalPredictor
            This estimator.

        Raises
        ------
        ValueError

        """
        valid_params = self.get_params(deep=False)
        estimator_params: dict[str, Any] = {}

        for key, value in params.items():
            if key in valid_params:
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {type(self).__name__}",
                )

        if estimator_params and hasattr(self.estimator, "set_params"):
            self.estimator.set_params(**estimator_params)

        return self
