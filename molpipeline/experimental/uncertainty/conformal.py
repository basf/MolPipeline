"""
Conformal prediction wrappers for classification and regression models.

This module provides unified implementations of conformal prediction for
uncertainty quantification with both classification and regression models.
"""

from crepes import WrapClassifier, WrapRegressor
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import StratifiedKFold, KFold
from crepes.extras import hinge, margin, MondrianCategorizer, DifficultyEstimator
import numpy as np
import numpy.typing as npt
from typing import Any, Callable, Optional, Literal, List, Union
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_random_state
from scipy.stats import mode


def _bin_targets(y: npt.NDArray[Any], n_bins: int = 10) -> npt.NDArray[np.int_]:
    """
    Bin continuous targets for stratified splitting in regression.

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


class UnifiedConformalCV(BaseEstimator):
    def __init__(
        self,
        estimator: BaseEstimator,
        mondrian: bool | Callable | MondrianCategorizer = False,
        confidence_level: float = 0.9,
        estimator_type: Literal["auto", "classifier", "regressor"] = "auto",
        nonconformity: Optional[Callable] = None,
        difficulty_estimator: Optional[Callable] = None,
        binning: Optional[int | Callable] = None,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        **kwargs: Any
    ):
        """
        Unified conformal prediction wrapper for both classifiers and regressors.

        Parameters
        ----------
        estimator : BaseEstimator
            The underlying model or pipeline to wrap.
        mondrian : bool, callable, or MondrianCategorizer, optional
            If True, use class-conditional (Mondrian) calibration. If callable or MondrianCategorizer, use as custom group function/categorizer.
        confidence_level : float, optional
            Confidence level for prediction sets/intervals (default: 0.9).
        estimator_type : Literal["auto", "classifier", "regressor"], optional
            Type of estimator. If "auto", will infer using sklearn's is_classifier/is_regressor.
        nonconformity : callable, optional
            Nonconformity function for classification (e.g., hinge, margin, or custom).
        difficulty_estimator : callable, optional
            For regression: difficulty estimator for normalized conformal prediction.
        binning : int or callable, optional
            For regression: number of bins or binning function for Mondrian calibration.
        n_jobs : int, optional
            Number of parallel jobs to use.
        random_state : int or None, optional
            Random state for reproducibility.
        **kwargs : dict
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
        self.random_state = check_random_state(random_state) if random_state is not None else None
        self.fitted_ = False
        self.calibrated_ = False
        self._conformal = None
        
        # Determine estimator_type if auto
        if estimator_type == "auto":
            if is_classifier(estimator):
                self._resolved_estimator_type = "classifier"
            elif is_regressor(estimator):
                self._resolved_estimator_type = "regressor"
            else:
                raise ValueError(
                    "Could not automatically determine estimator_type. "
                    "Please specify 'classifier' or 'regressor'."
                )
        else:
            self._resolved_estimator_type = estimator_type

    def _get_mondrian_param_classification(self, mondrian, y_calib):
        if isinstance(mondrian, MondrianCategorizer) or callable(mondrian):
            return mondrian
        elif mondrian is True:
            return y_calib
        else:
            return None

    def _get_mondrian_param_regression(self, mondrian, y_calib):
        if isinstance(mondrian, MondrianCategorizer) or callable(mondrian):
            return mondrian
        elif mondrian is True:
            return y_calib
        else:
            return None
            
    def get_params(self, deep=True):
        return {
            "estimator": self.estimator,
            "mondrian": self.mondrian,
            "confidence_level": self.confidence_level,
            "estimator_type": self.estimator_type,
            "nonconformity": self.nonconformity,
            "difficulty_estimator": self.difficulty_estimator,
            "binning": self.binning,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            **self.kwargs,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any], **fit_params: Any) -> "UnifiedConformalCV":
        if self._resolved_estimator_type == "classifier":
            self._conformal = WrapClassifier(clone(self.estimator))
        elif self._resolved_estimator_type == "regressor":
            self._conformal = WrapRegressor(clone(self.estimator))
        else:
            raise ValueError("estimator_type must be 'classifier' or 'regressor'")
        self._conformal.fit(X, y, **fit_params)
        self.fitted_ = True
        self.models_ = [self._conformal]

        return self

    def calibrate(
        self,
        X_calib: npt.NDArray[Any],
        y_calib: npt.NDArray[Any],
        **calib_params: Any,
    ) -> None:
        if self._resolved_estimator_type == "classifier":
            nc = self.nonconformity if self.nonconformity is not None else hinge
            mc = self._get_mondrian_param_classification(self.mondrian, y_calib)
            self._conformal.calibrate(X_calib, y_calib, nc=nc, mc=mc, **calib_params)
            self.calibrated_ = True

        elif self._resolved_estimator_type == "regressor":
            de = self.difficulty_estimator
            mc = self._get_mondrian_param_regression(self.mondrian, y_calib)
            self._conformal.calibrate(X_calib, y_calib, de=de, mc=mc, **calib_params)
            self.calibrated_ = True
        else:
            raise ValueError("estimator_type must be 'classifier' or 'regressor'")

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return self._conformal.predict(X)

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self._resolved_estimator_type != "classifier":
            raise NotImplementedError("predict_proba is for classifiers only.")
        return self._conformal.predict_proba(X)

    def predict_conformal_set(
        self,
        X: npt.NDArray[Any],
        confidence: float | None = None,
    ) -> list[list[Any]]:
        """
        Predict conformal sets for classification.

        Parameters
        ----------
        X : npt.NDArray[Any]
            Input features.
        confidence : float or None, optional
            Confidence level for prediction set (default: self.confidence_level).

        Returns
        -------
        list[list[Any]]
            List of conformal sets (per sample), each a list of class labels.
        """
        if self._resolved_estimator_type != "classifier":
            raise NotImplementedError("predict_conformal_set is only for classification.")
        if not self.fitted_:
            raise RuntimeError("You must fit the model before calling predict_conformal_set.")
        
        # Default confidence to self.confidence_level if not provided
        confidence = confidence if confidence is not None else self.confidence_level
        
        pred_set_bin = self._conformal.predict_set(X, confidence=confidence)
        classes = self._conformal.learner.classes_
        return [list(np.array(classes)[row.astype(bool)]) for row in pred_set_bin]

    def predict_p(self, X: npt.NDArray[Any], **kwargs: Any) -> npt.NDArray[Any]:
        if self._resolved_estimator_type != "classifier":
            raise NotImplementedError("predict_p is only for classification.")
        return self._conformal.predict_p(X, **kwargs)

    def predict_int(self, X: npt.NDArray[Any], confidence: float | None = None) -> npt.NDArray[Any]:
        """
        Predict confidence intervals for regression.
        
        Parameters
        ----------
        X : npt.NDArray[Any]
            Input features.
        confidence : float or None, optional
            Confidence level for intervals (default: self.confidence_level).
            
        Returns
        -------
        npt.NDArray[Any]
            Array of prediction intervals, shape (n_samples, 2).
        """
        if self._resolved_estimator_type != "regressor":
            raise NotImplementedError("predict_int is only for regression.")
        conf = confidence if confidence is not None else self.confidence_level
        return self._conformal.predict_int(X, confidence=conf)
    


class CrossConformalCV(BaseEstimator):
    def __init__(
        self,
        estimator: BaseEstimator,
        n_folds: int = 5,
        confidence_level: float = 0.9,
        mondrian: bool | Callable | MondrianCategorizer = False,
        nonconformity: Optional[Callable] = None,
        binning: Optional[int | Callable] = None,
        estimator_type: Literal["auto", "classifier", "regressor"] = "auto",
        n_bins: int = 10,
        difficulty_estimator: Optional[Callable] = None,
        random_state: Optional[int] = None,
        **kwargs: Any
    ):
        """
        Cross-conformal prediction for both classifiers and regressors using WrapClassifier/WrapRegressor.

        Parameters
        ----------
        estimator : BaseEstimator
            The underlying model or pipeline to wrap.
        n_folds : int, optional
            Number of cross-validation folds (default: 5).
        confidence_level : float, optional
            Confidence level for prediction sets/intervals (default: 0.9).
        mondrian : bool, callable, or MondrianCategorizer, optional
            Mondrian calibration/grouping.
        nonconformity : callable, optional
            Nonconformity function for classification (e.g., hinge, margin, or custom).
        binning : int or callable, optional
            For regression: number of bins or binning function for Mondrian calibration.
        estimator_type : Literal["auto", "classifier", "regressor"], optional
            Type of estimator. If "auto", will infer using sklearn's is_classifier/is_regressor.
        n_bins : int, optional
            Number of bins for stratified splitting in regression (default: 10).
        difficulty_estimator : callable, optional
            For regression: difficulty estimator for normalized conformal prediction.
        random_state : int or None, optional
            Random state for reproducibility.
        **kwargs : dict
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
        self.difficulty_estimator = difficulty_estimator
        self.kwargs = kwargs
        self.random_state = check_random_state(random_state) if random_state is not None else None
        self.fitted_ = False
        self.calibrated_ = False
        
        # Determine estimator_type if auto
        if estimator_type == "auto":
            if is_classifier(estimator):
                self._resolved_estimator_type = "classifier"
            elif is_regressor(estimator):
                self._resolved_estimator_type = "regressor"
            else:
                raise ValueError(
                    "Could not automatically determine estimator_type. "
                    "Please specify 'classifier' or 'regressor'."
                )
        else:
            self._resolved_estimator_type = estimator_type

    def get_params(self, deep=True):
        return {
            "estimator": self.estimator,
            "n_folds": self.n_folds,
            "confidence_level": self.confidence_level,
            "mondrian": self.mondrian,
            "nonconformity": self.nonconformity,
            "binning": self.binning,
            "estimator_type": self.estimator_type,
            "n_bins": self.n_bins,
            "difficulty_estimator": self.difficulty_estimator,
            "random_state": self.random_state,
            **self.kwargs,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any], **fit_params: Any) -> "CrossConformalCV":
        X = np.asarray(X)
        y = np.asarray(y)
        self.models_ = []
        self.mondrian_categorizers_ = []  # Store categorizers for each fold
        self.calib_bins_ = []  # Store calibration bins for each fold
        
        if self._resolved_estimator_type == "classifier":
            splitter = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            y_split = y
        elif self._resolved_estimator_type == "regressor":
            splitter = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            y_split = _bin_targets(y, n_bins=self.n_bins)
        else:
            raise ValueError("estimator_type must be 'classifier' or 'regressor'")
            
        for train_idx, calib_idx in splitter.split(X, y_split):
            if self._resolved_estimator_type == "classifier":
                model = WrapClassifier(clone(self.estimator))
                model.fit(X[train_idx], y[train_idx])
                if self.mondrian:
                    model.calibrate(X[calib_idx], y[calib_idx], nc=self.nonconformity or hinge, class_cond=True)
                else:
                    model.calibrate(X[calib_idx], y[calib_idx], nc=self.nonconformity or hinge, class_cond=False)
                self.mondrian_categorizers_.append(None)
                self.calib_bins_.append(None)
            else:
                model = WrapRegressor(clone(self.estimator))
                model.fit(X[train_idx], y[train_idx])
                de = None
                if self.difficulty_estimator is not None:
                    de = DifficultyEstimator()
                    de.fit(X[calib_idx], y=y[calib_idx])
                if self.mondrian:
                    if self.binning is not None:
                        mc = MondrianCategorizer()
                        mc.fit(X[calib_idx], f=lambda X: y[calib_idx], no_bins=self.binning)
                    else:
                        mc = MondrianCategorizer()
                        mc.fit(X[calib_idx], f=lambda X: y[calib_idx])
                    model.calibrate(X[calib_idx], y[calib_idx], de=de, mc=mc)
                    self.mondrian_categorizers_.append(mc)
                    self.calib_bins_.append(None)
                else:
                    model.calibrate(X[calib_idx], y[calib_idx], de=de)
                    self.mondrian_categorizers_.append(None)
                    self.calib_bins_.append(None)
            self.models_.append(model)
        self.calibrated_ = True
        self.fitted_ = True

        return self

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        result = np.array([m.predict(X) for m in self.models_])
        if self._resolved_estimator_type == "regressor":
            return np.mean(result, axis=0)
        pred_mode = mode(result, axis=0, keepdims=False)
        return np.ravel(pred_mode.mode)

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        result = np.array([m.predict_proba(X) for m in self.models_])
        proba = np.atleast_2d(np.mean(result, axis=0))
        return proba

    def predict_conformal_set(
        self,
        X: npt.NDArray[Any],
        confidence: float | None = None,
    ) -> List[List[Union[int]]]:
        """
        Predict conformal sets for classification by union across folds.

        Parameters
        ----------
        X : npt.NDArray[Any]
            Input features.
        confidence : float or None, optional
            Confidence level for prediction set (default: self.confidence_level).

        Returns
        -------
        List[List[Union[int]]]
            List of conformal sets (per sample), each containing the class labels
            that might be the true class with the specified confidence level.
            For example, for a binary classifier with classes [0, 1], might return
            [[0, 1], [1], [0, 1]] for three samples.
        """
        if self._resolved_estimator_type != "classifier":
            raise NotImplementedError("predict_conformal_set is only for classification.")
        if not self.fitted_:
            raise RuntimeError("You must fit the model before calling predict_conformal_set.")
        
        # Default confidence to self.confidence_level if not provided
        confidence = confidence if confidence is not None else self.confidence_level

        sets = []
        for m in self.models_:
            pred_set_bin = m.predict_set(X, confidence=confidence)
            classes = getattr(m.learner, "classes_", None)
            if classes is None:
                raise AttributeError("Underlying estimator does not expose 'classes_'.")
            sets.append([list(np.array(classes)[row.astype(bool)]) for row in pred_set_bin])
        
        n = len(X)
        union_sets: list[list[Any]] = []
        for i in range(n):
            union = set()
            for s in sets:
                union.update(s[i])
            union_sets.append(list(union))
        return union_sets

    def predict_int(self, X: npt.NDArray[Any], confidence: float | None = None) -> npt.NDArray[Any]:
        """
        Predict confidence intervals for regression.
        
        Parameters
        ----------
        X : npt.NDArray[Any]
            Input features.
        confidence : float or None, optional
            Confidence level for intervals (default: self.confidence_level).
            
        Returns
        -------
        npt.NDArray[Any]
            Array of prediction intervals, shape (n_samples, 2).
        """
        if self._resolved_estimator_type != "regressor":
            raise NotImplementedError("predict_int is only for regression.")
        conf = confidence if confidence is not None else self.confidence_level
        intervals = []
        for i, model in enumerate(self.models_):
            interval = model.predict_int(X, confidence=conf)
            intervals.append(np.array(interval))
        # Return average lower/upper bounds across folds
        intervals = np.array(intervals)  # shape: (n_folds, n_samples, 2)
        avg_intervals = np.nanmean(intervals, axis=0)
        return avg_intervals


    def predict_p(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Return averaged conformal p-values across folds (classification only)."""
        if self._resolved_estimator_type != "classifier":
            raise NotImplementedError("predict_p is only for classification.")
        # Each model in self.models_ has predict_p
        pvals = np.array([m.predict_p(X) for m in self.models_])  # shape: (n_folds, n_samples, n_classes)
        avg_pvals = np.mean(pvals, axis=0)  # shape: (n_samples, n_classes)
        return avg_pvals