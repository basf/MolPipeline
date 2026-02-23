"""Unit tests for SplitEnsembleClassifier and SplitEnsembleRegressor."""

import unittest
from typing import Self

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold
from typing_extensions import override

from molpipeline.estimators.ensemble.split_ensemble import (
    SplitEnsembleClassifier,
    SplitEnsembleRegressor,
)


class MockEstimator(BaseEstimator):
    """A mock estimator that records fit arguments and returns fixed predictions."""

    fit_args: dict[str, npt.ArrayLike]

    def __init__(self, alpha: int = 0, beta: int = 0, gamma: int = 0) -> None:
        """Initialize the MockEstimator with dummy parameters.

        Parameters
        ----------
        alpha : int, default=0
            Dummy parameter alpha.
        beta : int, default=0
            Dummy parameter beta.
        gamma : int, default=0
            Dummy parameter gamma.

        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.fit_args = {}

    def fit(
        self,
        X: npt.ArrayLike,  # noqa: N803 # pylint: disable=invalid-name
        y: npt.ArrayLike,
    ) -> Self:
        """Fit the model and record the arguments.

        Parameters
        ----------
        X : npt.ArrayLike
            Training data.
        y : npt.ArrayLike
            Target values.

        Returns
        -------
        Self
            The fitted estimator.

        """
        self.fit_args = {"X": X, "y": y}
        return self

    def predict(  # noqa: PLR6301
        self,
        X: npt.ArrayLike,  # noqa: N803 # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float64]:
        """Return fixed predictions.

        Parameters
        ----------
        X : npt.ArrayLike
            Input data for prediction.

        Returns
        -------
        npt.NDArray[np.float64]
            Predicted values, which are all zeros for this mock estimator.

        """
        feature_arr = np.asarray(X)
        return np.zeros(len(feature_arr), dtype=np.float64)


class MockClassifier(MockEstimator):
    """A mock classifier that records fit arguments and returns fixed predictions."""

    @override
    def predict(  # type: ignore
        self,
        X: npt.ArrayLike,  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.int64]:
        """Return fixed class predictions.

        Parameters
        ----------
        X : npt.ArrayLike
            Input data for prediction.

        Returns
        -------
        npt.NDArray[np.int64]
            Predicted class labels.

        """
        feature_arr = np.asarray(X)
        return np.array([x[0] % 2 for x in feature_arr], dtype=np.int64)

    def predict_proba(  # noqa: PLR6301
        self,
        X: npt.ArrayLike,  # noqa: N803 # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float64]:
        """Return fake class probabilities.

        Parameters
        ----------
        X : npt.ArrayLike
            Input data for probability prediction.

        Returns
        -------
        npt.NDArray[np.float64]
            Predicted class probabilities, where class 0 has probability 0.7 and class
            1 has probability 0.3 for all samples.

        """
        feature_arr = np.asarray(X)
        proba = np.zeros((len(feature_arr), 2))
        proba[:, 0] = 0.7
        proba[:, 1] = 0.3
        return proba


class TestSplitEnsembleRegressor(unittest.TestCase):
    """Unit tests for SplitEnsembleRegressor."""

    def test_param_forwarding(self) -> None:
        """Test that parameters are correctly forwarded to the base estimator.

        Raises
        ------
        TypeError
            If the base estimator is not an instance of MockEstimator.

        """
        base = MockEstimator(alpha=1)
        ensemble = SplitEnsembleRegressor(
            estimator=base,
            cv=3,
            estimator__beta=2,
        )
        ensemble.set_params(estimator__gamma=3)
        base_est = ensemble.estimator
        if not isinstance(base_est, MockEstimator):
            raise TypeError("Expected an instance of MockEstimator")
        self.assertEqual(base_est.alpha, 1)
        self.assertEqual(base_est.beta, 2)
        self.assertEqual(base_est.gamma, 3)

    def test_fit_sample_forwarding(self) -> None:
        """Test that fit samples are correctly forwarded to each base estimator.

        Raises
        ------
        TypeError
            If any of the fitted estimators is not an instance of MockEstimator.

        """
        features = np.array([[i, i, i, i] for i in range(10)])
        y = np.arange(10)
        base = MockEstimator()
        ensemble = SplitEnsembleRegressor(estimator=base, cv=2)
        ensemble.fit(features, y)
        # Should have 2 estimators, each fit on a split
        self.assertEqual(len(ensemble.estimators_), 2)
        # Reconstruct the expected splits using KFold
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        splits = list(kf.split(features, y))
        for est, (train_idx, _) in zip(ensemble.estimators_, splits, strict=True):
            if not isinstance(est, MockEstimator):
                raise TypeError("Expected an instance of MockEstimator")
            self.assertTrue(np.array_equal(est.fit_args["X"], features[train_idx]))
            self.assertTrue(np.array_equal(est.fit_args["y"], y[train_idx]))

    def test_fit_sample_forwarding_with_lists(self) -> None:
        """Test that fit samples are correctly forwarded when X and y are lists.

        Raises
        ------
        TypeError
            If any of the fitted estimators is not an instance of MockEstimator.

        """
        features = [[i, i, i, i] for i in range(10)]
        y = [float(i) for i in range(10)]
        base = MockEstimator()
        ensemble = SplitEnsembleRegressor(estimator=base, cv=2)
        ensemble.fit(features, y)
        self.assertEqual(len(ensemble.estimators_), 2)

        features_arr = np.array(features)
        y_arr = np.array(y)
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        splits = list(kf.split(features_arr, y_arr))
        for est, (train_idx, _) in zip(ensemble.estimators_, splits, strict=True):
            if not isinstance(est, MockEstimator):
                raise TypeError("Expected an instance of MockEstimator")
            self.assertTrue(np.array_equal(est.fit_args["X"], features_arr[train_idx]))
            self.assertTrue(np.array_equal(est.fit_args["y"], y_arr[train_idx]))


class TestSplitEnsembleClassifier(unittest.TestCase):
    """Unit tests for SplitEnsembleClassifier."""

    def test_param_forwarding(self) -> None:
        """Test that parameters are correctly forwarded to the base estimator.

        Raises
        ------
        TypeError
            If the base estimator is not an instance of MockClassifier.

        """
        base = MockClassifier(alpha=1)
        ensemble = SplitEnsembleClassifier(
            estimator=base,
            cv=3,
            estimator__beta=2,
        )
        ensemble.set_params(estimator__gamma=3)
        base_est = ensemble.estimator
        if not isinstance(base_est, MockClassifier):
            raise TypeError("Expected an instance of MockClassifier")
        self.assertEqual(base_est.alpha, 1)
        self.assertEqual(base_est.beta, 2)
        self.assertEqual(base_est.gamma, 3)

    def test_fit_sample_forwarding(self) -> None:
        """Test that fit samples are correctly forwarded to each base estimator.

        Raises
        ------
        TypeError
            If the fitted estimator is not an instance of MockClassifier.

        """
        features = np.array([[i, i, i, i] for i in range(10)])
        y = np.arange(10) % 2
        base = MockClassifier()
        ensemble = SplitEnsembleClassifier(estimator=base, cv=2)
        ensemble.fit(features, y)
        self.assertEqual(len(ensemble.estimators_), 2)

        kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        splits = list(kf.split(features, y))
        for est, (train_idx, _) in zip(ensemble.estimators_, splits, strict=True):
            if not isinstance(est, MockClassifier):
                raise TypeError("Expected an instance of MockClassifier")
            self.assertTrue(np.array_equal(est.fit_args["X"], features[train_idx]))
            self.assertTrue(np.array_equal(est.fit_args["y"], y[train_idx]))

    def test_fit_sample_forwarding_with_lists(self) -> None:
        """Test that fit samples are correctly forwarded when X and y are lists.

        Raises
        ------
        TypeError
            If the fitted estimator is not an instance of MockClassifier.

        """
        features = [[i, i, i, i] for i in range(10)]
        y = [i % 2 for i in range(10)]
        base = MockClassifier()
        ensemble = SplitEnsembleClassifier(estimator=base, cv=2)
        ensemble.fit(features, y)
        self.assertEqual(len(ensemble.estimators_), 2)

        features_arr = np.array(features)
        y_arr = np.array(y)
        kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        splits = list(kf.split(features_arr, y_arr))
        for est, (train_idx, _) in zip(ensemble.estimators_, splits, strict=True):
            if not isinstance(est, MockClassifier):
                raise TypeError("Expected an instance of MockClassifier")
            # The estimator receives numpy arrays, so compare with arrays
            self.assertTrue(np.array_equal(est.fit_args["X"], features_arr[train_idx]))
            self.assertTrue(np.array_equal(est.fit_args["y"], y_arr[train_idx]))

    def test_predict_hard_voting(self) -> None:
        """Test that hard voting prediction returns the majority class."""
        features = np.array([[i, i, i, i] for i in range(6)])
        y = np.array([0, 1, 0, 1, 0, 1])
        base = MockClassifier()
        ensemble = SplitEnsembleClassifier(estimator=base, cv=2, voting="hard")
        ensemble.fit(features, y)
        preds = ensemble.predict(features)
        # Since all estimators return alternating 0,1, the majority is always 0
        self.assertTrue(np.array_equal(preds, np.array([0, 1, 0, 1, 0, 1])))

    def test_predict_soft_voting(self) -> None:
        """Test that soft voting returns the class with highest average probability."""
        features = np.array([[i, i, i, i] for i in range(6)])
        y = np.array([0, 1, 0, 1, 0, 1])
        base = MockClassifier()
        ensemble = SplitEnsembleClassifier(estimator=base, cv=2, voting="soft")
        ensemble.fit(features, y)
        preds = ensemble.predict(features)
        # Since predict_proba always returns class 0 as most probable
        self.assertTrue(np.array_equal(preds, np.zeros(len(features))))
        proba = ensemble.predict_proba(features)
        self.assertTrue(np.array_equal(proba, np.tile([0.7, 0.3], (len(features), 1))))


if __name__ == "__main__":
    unittest.main()
