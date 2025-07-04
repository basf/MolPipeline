"""Conformal prediction wrappers for classification and regression using crepes.

Provides unified and cross-conformal prediction with Mondrian and nonconformity options.
"""

from typing import Any, cast

import numpy as np
from crepes import WrapClassifier, WrapRegressor
from crepes.extras import MondrianCategorizer
from scipy.stats import mode
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold


def bin_targets(y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Bin continuous targets for stratified splitting in regression.

    Parameters
    ----------
    y : np.ndarray
        Target values.
    n_bins : int, optional
        Number of bins (default: 10).

    Returns
    -------
    np.ndarray
        Binned targets.

    """
    y = np.asarray(y)
    bins = np.linspace(np.min(y), np.max(y), n_bins + 1)
    y_binned = np.digitize(y, bins) - 1  # bins start at 1
    y_binned[y_binned == n_bins] = n_bins - 1  # edge case
    return y_binned


class UnifiedConformalCV(BaseEstimator):
    """One wrapper to rule them all: conformal prediction for both classifiers and regressors.

    Uses crepes under the hood, so you know it's sweet.

    Parameters
    ----------
    estimator : sklearn-like estimator
        Your favorite model (or pipeline).
    mondrian : bool/callable/MondrianCategorizer, optional
        If True, use class-conditional (Mondrian) calibration. If callable or
        MondrianCategorizer, use as custom group function/categorizer.
    confidence_level : float, optional
        How confident should we be? (default: 0.9)
    estimator_type : {'classifier', 'regressor'}, optional
        What kind of model are we wrapping?
    nonconformity : callable, optional
        Nonconformity function for classification (e.g., hinge, margin, or custom).
    difficulty_estimator : callable or DifficultyEstimator, optional
        For regression: difficulty estimator for normalized conformal prediction.
    binning : int or callable, optional
        For regression: number of bins or binning function for Mondrian calibration.
    n_jobs : int, optional
        Parallelize all the things.
    kwargs : dict
        Extra toppings for crepes.

    """

    def __init__(
        self,
        estimator: Any,
        mondrian: Any = False,
        confidence_level: float = 0.9,
        estimator_type: str = "classifier",
        nonconformity: Any | None = None,
        difficulty_estimator: Any | None = None,
        binning: Any | None = None,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize UnifiedConformalCV.

        Parameters
        ----------
        estimator : Any
            The base estimator or pipeline to wrap.
        mondrian : Any, optional
            Mondrian calibration/grouping (default: False).
        confidence_level : float, optional
            Confidence level for prediction sets/intervals (default: 0.9).
        estimator_type : str, optional
            Type of estimator: 'classifier' or 'regressor' (default: 'classifier').
        nonconformity : Any, optional
            Nonconformity function for classification.
        difficulty_estimator : Any, optional
            Difficulty estimator for normalized conformal prediction (regression).
        binning : Any, optional
            Number of bins or binning function for Mondrian calibration (regression).
        n_jobs : int, optional
            Number of parallel jobs (default: 1).
        **kwargs : Any
            Additional keyword arguments for crepes.
        """
        self.estimator = estimator
        self.mondrian = mondrian
        self.confidence_level = confidence_level
        self.estimator_type = estimator_type
        self.nonconformity = nonconformity
        self.difficulty_estimator = difficulty_estimator
        self.binning = binning
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def fit(self, x: np.ndarray, y: np.ndarray) -> "UnifiedConformalCV":
        """Fit the conformal predictor.

        Parameters
        ----------
        x : np.ndarray
            Training features.
        y : np.ndarray
            Training targets.

        Returns
        -------
        UnifiedConformalCV
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
        self, x_calib: np.ndarray, y_calib: np.ndarray, **calib_params: Any,
    ) -> None:
        """Calibrate the conformal predictor.

        Parameters
        ----------
        x_calib : np.ndarray
            Calibration features.
        y_calib : np.ndarray
            Calibration targets.
        calib_params : dict
            Additional calibration parameters.

        Raises
        ------
        ValueError
            If estimator_type is not 'classifier' or 'regressor'.

        """
        if self.estimator_type == "classifier":
            mondrian = self.mondrian
            if isinstance(mondrian, MondrianCategorizer) or callable(mondrian):
                self._conformal.calibrate(x_calib, y_calib, mc=mondrian, **calib_params)
            elif mondrian is True:
                # Use class labels as Mondrian categories
                self._conformal.calibrate(x_calib, y_calib, mc=y_calib, **calib_params)
            else:
                self._conformal.calibrate(x_calib, y_calib, **calib_params)
        elif self.estimator_type == "regressor":
            mondrian = self.mondrian
            if isinstance(mondrian, MondrianCategorizer) or callable(mondrian):
                mc = mondrian
            else:
                mc = None
            self._conformal.calibrate(x_calib, y_calib, mc=mc, **calib_params)
        else:
            raise ValueError("estimator_type must be 'classifier' or 'regressor'")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict using the conformal predictor.

        Parameters
        ----------
        x : np.ndarray
            Features to predict.

        Returns
        -------
        np.ndarray
            Predictions.

        """
        return self._conformal.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict probabilities using the conformal predictor.

        Parameters
        ----------
        x : np.ndarray
            Features to predict.

        Returns
        -------
        np.ndarray
            Predicted probabilities.

        Raises
        ------
        NotImplementedError
            If called for a regressor.

        """
        if self.estimator_type != "classifier":
            raise NotImplementedError("predict_proba is for classifiers only.")
        conformal = cast("WrapClassifier", self._conformal)
        return conformal.predict_proba(x)

    def predict_conformal_set(
        self, x: np.ndarray, confidence: float | None = None,
    ) -> Any:
        """Predict conformal sets.

        Parameters
        ----------
        x : np.ndarray
            Features to predict.
        confidence : float, optional
            Confidence level.

        Returns
        -------
        Any
            Conformal prediction sets.

        Raises
        ------
        NotImplementedError
            If called for a regressor.

        """
        if self.estimator_type != "classifier":
            raise NotImplementedError(
                "predict_conformal_set is only for classification.",
            )
        conf = confidence if confidence is not None else self.confidence_level
        conformal = cast("WrapClassifier", self._conformal)
        return conformal.predict_set(x, confidence=conf)

    def predict_p(self, x: np.ndarray, **kwargs: Any) -> Any:
        """Predict p-values.

        Parameters
        ----------
        x : np.ndarray
            Features to predict.
        kwargs : dict
            Additional parameters.

        Returns
        -------
        Any
            p-values.

        Raises
        ------
        NotImplementedError
            If called for a regressor.

        """
        if self.estimator_type != "classifier":
            raise NotImplementedError("predict_p is only for classification.")
        return self._conformal.predict_p(x, **kwargs)

    def predict_int(self, x: np.ndarray, confidence: float | None = None) -> Any:
        """Predict intervals.

        Parameters
        ----------
        x : np.ndarray
            Features to predict.
        confidence : float, optional
            Confidence level.

        Returns
        -------
        Any
            Prediction intervals.

        Raises
        ------
        NotImplementedError
            If called for a classifier.

        """
        if self.estimator_type != "regressor":
            raise NotImplementedError("predict_interval is only for regression.")
        conf = confidence if confidence is not None else self.confidence_level
        conformal = cast("WrapRegressor", self._conformal)
        return conformal.predict_int(x, confidence=conf)


class CrossConformalCV(BaseEstimator):
    """Cross-conformal prediction for both classifiers and regressors using WrapClassifier/WrapRegressor.

    Handles Mondrian (class_cond) logic as described.

    Parameters
    ----------
    estimator : sklearn-like estimator
        Your favorite model (or pipeline).
    n_folds : int, optional
        Number of cross-validation folds.
    confidence_level : float, optional
        Confidence level for prediction sets/intervals.
    mondrian : bool/callable/MondrianCategorizer, optional
        Mondrian calibration/grouping.
    nonconformity : callable, optional
        Nonconformity function for classification (e.g., hinge, margin, or custom).
    difficulty_estimator : callable or DifficultyEstimator, optional
        For regression: difficulty estimator for normalized conformal prediction.
    binning : int or callable, optional
        For regression: number of bins or binning function for Mondrian calibration.
    estimator_type : {'classifier', 'regressor'}, optional
        What kind of model are we wrapping?
    n_bins : int, optional
        Number of bins for stratified splitting in regression.
    n_jobs : int, optional
        Parallelize all the things.
    kwargs : dict
        Extra toppings for crepes.

    """

    def __init__(
        self,
        estimator: Any,
        n_folds: int = 5,
        confidence_level: float = 0.9,
        mondrian: Any = False,
        nonconformity: Any | None = None,
        binning: Any | None = None,
        estimator_type: str = "classifier",
        n_bins: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize CrossConformalCV.

        Parameters
        ----------
        estimator : Any
            The base estimator or pipeline to wrap.
        n_folds : int, optional
            Number of cross-validation folds (default: 5).
        confidence_level : float, optional
            Confidence level for prediction sets/intervals (default: 0.9).
        mondrian : Any, optional
            Mondrian calibration/grouping (default: False).
        nonconformity : Any, optional
            Nonconformity function for classification.
        binning : Any, optional
            Number of bins or binning function for Mondrian calibration (regression).
        estimator_type : str, optional
            Type of estimator: 'classifier' or 'regressor' (default: 'classifier').
        n_bins : int, optional
            Number of bins for stratified splitting in regression (default: 10).
        **kwargs : Any
            Additional keyword arguments for crepes.
        """
        self.estimator = estimator
        self.n_folds = n_folds
        self.confidence_level = confidence_level
        self.mondrian = mondrian
        self.nonconformity = nonconformity
        self.binning = binning
        self.estimator_type = estimator_type
        self.n_bins = n_bins
        self.kwargs = kwargs

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> "CrossConformalCV":
        """Fit the cross-conformal predictor.

        Parameters
        ----------
        x : np.ndarray
            Training features.
        y : np.ndarray
            Training targets.

        Returns
        -------
        CrossConformalCV
            Self.

        Raises
        ------
        ValueError
            If estimator_type is not 'classifier' or 'regressor'.

        """
        x = np.array(x)
        y = np.array(y)
        self.models_ = []
        if self.estimator_type == "classifier":
            splitter = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=42,
            )
            y_split = y
        elif self.estimator_type == "regressor":
            splitter = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            y_split = bin_targets(y, n_bins=self.n_bins)
        else:
            raise ValueError("estimator_type must be 'classifier' or 'regressor'")
        for train_idx, calib_idx in splitter.split(x, y_split):
            if self.estimator_type == "classifier":
                model = WrapClassifier(clone(self.estimator))
                model.fit(x[train_idx], y[train_idx])
                mondrian = self.mondrian
                if isinstance(mondrian, MondrianCategorizer) or callable(mondrian):
                    model.calibrate(x[calib_idx], y[calib_idx], mc=mondrian)
                elif mondrian is True:
                    model.calibrate(x[calib_idx], y[calib_idx], mc=y[calib_idx])
                else:
                    model.calibrate(x[calib_idx], y[calib_idx])
            else:
                model = WrapRegressor(clone(self.estimator))
                model.fit(x[train_idx], y[train_idx])
                mondrian = self.mondrian
                if isinstance(mondrian, MondrianCategorizer) or callable(mondrian):
                    mc = mondrian
                else:
                    mc = None
                if self.binning is not None:
                    mc_obj = MondrianCategorizer()
                    calib_idx_val = calib_idx

                    def _bin_func(
                        _: Any, calib_idx_val: Any = calib_idx_val,
                    ) -> Any:
                        return y[calib_idx_val]

                    mc_obj.fit(x[calib_idx], f=_bin_func, no_bins=self.binning)
                    mc = mc_obj
                model.calibrate(x[calib_idx], y[calib_idx], mc=mc)
            self.models_.append(model)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict using the cross-conformal predictor.

        Parameters
        ----------
        x : np.ndarray
            Features to predict.

        Returns
        -------
        np.ndarray
            Predictions (majority vote).

        """
        result = np.array([m.predict(x) for m in self.models_])
        result = np.asarray(result)
        if result.shape == ():
            result = np.full((len(self.models_), len(x)), result)
        if result.ndim == 1 and len(x) == 1:
            result = result[:, np.newaxis]
        pred_mode = mode(result, axis=0, keepdims=False)
        return np.ravel(pred_mode.mode)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict probabilities using the cross-conformal predictor.

        Parameters
        ----------
        x : np.ndarray
            Features to predict.

        Returns
        -------
        np.ndarray
            Predicted probabilities (averaged).

        Raises
        ------
        NotImplementedError
            If called for a regressor.

        """
        if self.estimator_type != "classifier":
            raise NotImplementedError("predict_proba is for classifiers only.")
        binary_class_dim = 2
        result = np.array([m.predict_proba(x) for m in self.models_])
        if (
            result.ndim == binary_class_dim
            and result.shape[1] == binary_class_dim
            and len(x) == 1
        ):
            result = result[:, np.newaxis, :]
        proba = np.atleast_2d(np.mean(result, axis=0))
        if proba.shape[0] != len(x):
            proba = np.full((len(x), proba.shape[1]), np.nan)
        return proba

    def predict_conformal_set(
        self, x: np.ndarray, confidence: float | None = None,
    ) -> list[list[Any]]:
        """Predict conformal sets using the cross-conformal predictor.

        Parameters
        ----------
        x : np.ndarray
            Features to predict.
        confidence : float, optional
            Confidence level.

        Returns
        -------
        list[list[Any]]
            Union of conformal sets from all folds.

        Raises
        ------
        NotImplementedError
            If called for a regressor.

        """
        if self.estimator_type != "classifier":
            raise NotImplementedError(
                "predict_conformal_set is only for classification.",
            )
        conf = confidence if confidence is not None else self.confidence_level
        sets = [m.predict_set(x, confidence=conf) for m in self.models_]
        n = len(x)
        union_sets = []
        for i in range(n):
            union = set()
            for s in sets:
                union.update(s[i])
            union_sets.append(list(union))
        return union_sets
