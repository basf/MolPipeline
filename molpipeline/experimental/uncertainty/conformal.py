from crepes import WrapClassifier, WrapRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from crepes.extras import hinge, margin, MondrianCategorizer
import numpy as np
from sklearn.base import BaseEstimator, clone
from scipy.stats import mode

def bin_targets(y, n_bins=10):
    """
    Bin continuous targets for stratified splitting in regression.
    """
    y = np.asarray(y)
    bins = np.linspace(np.min(y), np.max(y), n_bins + 1)
    y_binned = np.digitize(y, bins) - 1  # bins start at 1
    y_binned[y_binned == n_bins] = n_bins - 1  # edge case
    return y_binned

class UnifiedConformalCV(BaseEstimator):
    """
    One wrapper to rule them all: conformal prediction for both classifiers and regressors.
    Uses crepes under the hood, so you know it's sweet.

    Parameters
    ----------
    estimator : sklearn-like estimator
        Your favorite model (or pipeline).
    mondrian : bool/callable/MondrianCategorizer, optional
        If True, use class-conditional (Mondrian) calibration. If callable or MondrianCategorizer, use as custom group function/categorizer.
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
        estimator,
        mondrian=False,
        confidence_level=0.9,
        estimator_type="classifier",
        nonconformity=None,
        difficulty_estimator=None,
        binning=None,
        n_jobs=1,
        **kwargs
    ):
        self.estimator = estimator
        self.mondrian = mondrian
        self.confidence_level = confidence_level
        self.estimator_type = estimator_type
        self.nonconformity = nonconformity
        self.difficulty_estimator = difficulty_estimator
        self.binning = binning
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def fit(self, X, y, **fit_params):
        if self.estimator_type == "classifier":
            self._conformal = WrapClassifier(clone(self.estimator))
        elif self.estimator_type == "regressor":
            self._conformal = WrapRegressor(clone(self.estimator))
        else:
            raise ValueError("estimator_type must be 'classifier' or 'regressor'")
        self._conformal.fit(X, y, **fit_params)
        self.fitted_ = True
        return self

    def calibrate(self, X_calib, y_calib, **calib_params):
        # --- Classification ---
        if self.estimator_type == "classifier":
            nc = self.nonconformity if self.nonconformity is not None else hinge
            mondrian = self.mondrian
            if isinstance(mondrian, MondrianCategorizer):
                mc = mondrian
                self._conformal.calibrate(X_calib, y_calib, nc=nc, mc=mc, **calib_params)
            elif callable(mondrian):
                mc = mondrian
                self._conformal.calibrate(X_calib, y_calib, nc=nc, mc=mc, **calib_params)
            elif mondrian is True:
                self._conformal.calibrate(X_calib, y_calib, nc=nc, class_cond=True, **calib_params)
            else:
                self._conformal.calibrate(X_calib, y_calib, nc=nc, class_cond=False, **calib_params)
        # --- Regression ---
        elif self.estimator_type == "regressor":
            de = self.difficulty_estimator
            mondrian = self.mondrian
            if isinstance(mondrian, MondrianCategorizer) or callable(mondrian):
                mc = mondrian
            else:
                mc = None
            bin_opt = self.binning
            self._conformal.calibrate(
                X_calib, y_calib, de=de, mc=mc, **calib_params
            )
        else:
            raise ValueError("estimator_type must be 'classifier' or 'regressor'")

    def predict(self, X):
        return self._conformal.predict(X)

    def predict_proba(self, X):
        if self.estimator_type != "classifier":
            raise NotImplementedError("predict_proba is for classifiers only.")
        return self._conformal.predict_proba(X)

    def predict_conformal_set(self, X, confidence=None):
        if self.estimator_type != "classifier":
            raise NotImplementedError("predict_conformal_set is only for classification.")
        conf = confidence if confidence is not None else self.confidence_level
        return self._conformal.predict_set(X, confidence=conf)

    def predict_p(self, X, **kwargs):
        if self.estimator_type != "classifier":
            raise NotImplementedError("predict_p is only for classification.")
        return self._conformal.predict_p(X, **kwargs)

    def predict_int(self, X, confidence=None):
        if self.estimator_type != "regressor":
            raise NotImplementedError("predict_interval is only for regression.")
        conf = confidence if confidence is not None else self.confidence_level
        return self._conformal.predict_int(X, confidence=conf)


class CrossConformalCV(BaseEstimator):
    """
    Cross-conformal prediction for both classifiers and regressors using WrapClassifier/WrapRegressor.
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
    def __init__(self, estimator, n_folds=5, confidence_level=0.9, mondrian=False, nonconformity=None, binning=None, estimator_type="classifier", n_bins=10, **kwargs):
        self.estimator = estimator
        self.n_folds = n_folds
        self.confidence_level = confidence_level
        self.mondrian = mondrian
        self.nonconformity = nonconformity
        self.binning = binning
        self.estimator_type = estimator_type
        self.n_bins = n_bins
        self.kwargs = kwargs

    def fit(self, X, y, **fit_params):
        X = np.array(X)
        y = np.array(y)
        self.models_ = []
        if self.estimator_type == "classifier":
            splitter = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            y_split = y
        elif self.estimator_type == "regressor":
            splitter = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            y_split = bin_targets(y, n_bins=self.n_bins)
        else:
            raise ValueError("estimator_type must be 'classifier' or 'regressor'")
        for train_idx, calib_idx in splitter.split(X, y_split):
            if self.estimator_type == "classifier":
                model = WrapClassifier(clone(self.estimator))
                model.fit(X[train_idx], y[train_idx])
                # Mondrian logic: only use class_cond=True if mondrian is True
                if self.mondrian:
                    model.calibrate(X[calib_idx], y[calib_idx], nc=self.nonconformity or hinge, class_cond=True)
                else:
                    model.calibrate(X[calib_idx], y[calib_idx], nc=self.nonconformity or hinge, class_cond=False)
            else:
                model = WrapRegressor(clone(self.estimator))
                model.fit(X[train_idx], y[train_idx])
                # Mondrian logic: use MondrianCategorizer with binning if mondrian
                if self.mondrian:
                    if self.binning is not None:
                        mc = MondrianCategorizer()
                        mc.fit(X[calib_idx], f=lambda X: y[calib_idx], no_bins=self.binning)
                    else:
                        mc = MondrianCategorizer()
                        mc.fit(X[calib_idx], f=lambda X: y[calib_idx])
                    model.calibrate(X[calib_idx], y[calib_idx], mc=mc)
                else:
                    model.calibrate(X[calib_idx], y[calib_idx])
            self.models_.append(model)
        return self

    def predict(self, X):
        # Majority vote
        result = np.array([m.predict(X) for m in self.models_])
        result = np.asarray(result)
        if result.shape == ():
            result = np.full((len(self.models_), len(X)), result)
        if result.ndim == 1 and len(X) == 1:
            result = result[:, np.newaxis]
        pred_mode = mode(result, axis=0, keepdims=False)
        return np.ravel(pred_mode.mode)

    def predict_proba(self, X):
        # Average probabilities
        result = np.array([m.predict_proba(X) for m in self.models_])
        if result.ndim == 2 and result.shape[1] == 2 and len(X) == 1:
            result = result[:, np.newaxis, :]
        proba = np.atleast_2d(np.mean(result, axis=0))
        if proba.shape[0] != len(X):
            proba = np.full((len(X), proba.shape[1]), np.nan)
        return proba

    def predict_conformal_set(self, X, confidence=None):
        # Union of conformal sets from all folds.
        sets = [m.predict_set(X, confidence) for m in self.models_]
        n = len(X)
        union_sets = []
        for i in range(n):
            union = set()
            for s in sets:
                union.update(s[i])
            union_sets.append(list(union))
        return union_sets
