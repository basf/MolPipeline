"""Unit tests for SplitEnsembleClassifier and SplitEnsembleRegressor."""

import unittest

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

from molpipeline.estimators.ensemble.split_ensemble import (
    SplitEnsembleClassifier,
    SplitEnsembleRegressor,
)
from tests.templates.test_wrapped_estimators import WrappedEstimatorBaseTestMixIn
from tests.utils.mock_estimators import (
    MockClassiferWithFloatLabels,
    MockClassifier,
    MockClassifierWithTrueFloatLabels,
    MockEstimator,
)


class TestSplitEnsembleRegressor(WrappedEstimatorBaseTestMixIn, unittest.TestCase):
    """Unit tests for SplitEnsembleRegressor."""

    @staticmethod
    def get_wrapped_estimator_type() -> type:
        """Return the SplitEnsembleRegressor class.

        Returns
        -------
        type[SplitEnsembleRegressor]
            The SplitEnsembleRegressor class.

        """
        return SplitEnsembleRegressor

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

    def test_linear_regression_dense_and_sparse(self) -> None:
        """Regressor works with both dense arrays and CSR sparse matrices.

        Raises
        ------
        TypeError
            If the predictions are not returned as numpy arrays.

        """
        features = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [1, 2], [2, 1]])
        y = np.array([0.0, 1.0, 1.0, 0.0, 2.0, 1.0])

        # Dense array
        reg = SplitEnsembleRegressor(estimator=LinearRegression(), cv=2)
        reg.fit(features, y)
        preds = reg.predict(features)
        self.assertIsInstance(preds, np.ndarray)
        if not isinstance(preds, np.ndarray):
            raise TypeError("Expected predictions to be a numpy array")
        self.assertEqual(preds.shape, (features.shape[0],))

        # Sparse matrix
        x_sparse = sp.csr_matrix(features)
        reg_sparse = SplitEnsembleRegressor(estimator=LinearRegression(), cv=2)
        reg_sparse.fit(x_sparse, y)
        preds_sparse = reg_sparse.predict(x_sparse)
        if not isinstance(preds, np.ndarray):
            raise TypeError("Expected predictions to be a numpy array")
        self.assertEqual(preds_sparse.shape, (x_sparse.shape[0],))


class TestSplitEnsembleClassifier(WrappedEstimatorBaseTestMixIn, unittest.TestCase):
    """Unit tests for SplitEnsembleClassifier."""

    @staticmethod
    def get_wrapped_estimator_type() -> type:
        """Return the SplitEnsembleClassifier class.

        Returns
        -------
        type[SplitEnsembleClassifier]
            The SplitEnsembleClassifier class.

        """
        return SplitEnsembleClassifier

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

    def test_predict_hard_voting_float_type(self) -> None:
        """Test that hard voting prediction works when y is provided as floats.

        The MockClassiferWithFloatLabels returns the same values as MockClassifier but
        as floats. The hard voting should still work and return the correct class
        labels, since they can be interpreted as integers.

        """
        features = np.array([[i, i, i, i] for i in range(6)])
        y = np.array([0, 1, 0, 1, 0, 1])
        base = MockClassiferWithFloatLabels()
        ensemble = SplitEnsembleClassifier(estimator=base, cv=2, voting="hard")
        ensemble.fit(features, y)
        preds = ensemble.predict(features)
        self.assertTrue(np.array_equal(preds, np.array([0, 1, 0, 1, 0, 1])))

    def test_predict_hard_voting_true_floats_raises_error(self) -> None:
        """Test that hard voting prediction raises an error when y has float values.

        If y contains float values that cannot be transformed to integers, the hard
        voting cannot determine the majority class and should not attempt to vote, but
        instead raise a ValueError.

        The MockClassifierWithTrueFloatLabels returns the same values as MockClassifier
        but adds 0.5, so they cannot be interpreted as integers.

        """
        features = np.array([[i, i, i, i] for i in range(6)])
        y = np.array([0, 1, 0, 1, 0, 1])
        base = MockClassifierWithTrueFloatLabels()
        ensemble = SplitEnsembleClassifier(estimator=base, cv=2, voting="hard")
        ensemble.fit(features, y)
        expected_msg = "Predictions are not integer values, cannot perform hard voting."
        with self.assertRaisesRegex(ValueError, expected_msg):
            ensemble.predict(features)

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

    def test_logistic_regression_dense_and_sparse(self) -> None:
        """Classifier works with both dense arrays and CSR sparse matrices."""
        features = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [1, 2], [2, 1]])
        y = np.array([0, 1, 1, 0, 1, 0])

        # Dense array
        clf = SplitEnsembleClassifier(
            estimator=LogisticRegression(solver="liblinear"),
            cv=2,
        )
        clf.fit(features, y)
        preds = clf.predict(features)
        self.assertEqual(preds.shape, (features.shape[0],))
        # Sparse matrix
        x_sparse = sp.csr_matrix(features)
        clf_sparse = SplitEnsembleClassifier(
            estimator=LogisticRegression(solver="liblinear"),
            cv=2,
        )
        clf_sparse.fit(x_sparse, y)
        preds_sparse = clf_sparse.predict(x_sparse)
        self.assertEqual(preds_sparse.shape, (x_sparse.shape[0],))


if __name__ == "__main__":
    unittest.main()
