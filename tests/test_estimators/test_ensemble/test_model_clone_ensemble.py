"""Unit tests for CloneEnsembleClassifier and CloneEnsembleRegressor."""

import unittest
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression, LogisticRegression

from molpipeline.estimators.ensemble.model_clone_ensemble import (
    CloneEnsembleClassifier,
    CloneEnsembleRegressor,
)
from tests.templates.test_wrapped_estimators import WrappedEstimatorBaseTestMixIn
from tests.utils.mock_estimators import (
    MockClassifier,
    MockEstimator,
)

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class TestCloneEnsembleRegressor(WrappedEstimatorBaseTestMixIn, unittest.TestCase):
    """Unit tests for CloneEnsembleRegressor."""

    @staticmethod
    def get_wrapped_estimator_type() -> type:
        """Return the CloneEnsembleRegressor class.

        Returns
        -------
        type[CloneEnsembleRegressor]
                The class of the wrapped estimator to be tested.

        """
        return CloneEnsembleRegressor

    def test_fit_sample_forwarding(self) -> None:
        """Each clone receives the full feature matrix and target vector.

        Raises
        ------
        TypeError
            If the base estimator is not an instance of MockClassifier.

        """
        features = np.array([[i, i, i, i] for i in range(10)])
        y = np.arange(10)
        base = MockEstimator()
        ensemble = CloneEnsembleRegressor(estimator=base, n_estimators=3)
        ensemble.fit(features, y)

        self.assertEqual(len(ensemble.estimators_), 3)
        for est in ensemble.estimators_:
            if not isinstance(est, MockEstimator):
                raise TypeError("Expected an instance of MockEstimator")
            self.assertTrue(np.array_equal(est.fit_args["X"], features))
            self.assertTrue(np.array_equal(est.fit_args["y"], y))

    def test_fit_sample_forwarding_with_lists(self) -> None:
        """List inputs are handled and forwarded to every clone.

        Raises
        ------
        TypeError
            If the base estimator is not an instance of MockClassifier.

        """
        features = [[i, i, i, i] for i in range(10)]
        y = [float(i) for i in range(10)]
        base = MockEstimator()
        ensemble = CloneEnsembleRegressor(estimator=base, n_estimators=3)
        ensemble.fit(features, y)

        self.assertEqual(len(ensemble.estimators_), 3)
        for est in ensemble.estimators_:
            if not isinstance(est, MockEstimator):
                raise TypeError("Expected an instance of MockEstimator")
            self.assertEqual(est.fit_args["X"], features)
            self.assertEqual(est.fit_args["y"], y)

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
        reg = CloneEnsembleRegressor(estimator=LinearRegression(), n_estimators=2)
        reg.fit(features, y)
        preds = reg.predict(features)
        self.assertIsInstance(preds, np.ndarray)
        if not isinstance(preds, np.ndarray):
            raise TypeError("Expected predictions to be a numpy array")
        self.assertEqual(preds.shape, (features.shape[0],))

        # Sparse matrix
        x_sparse = sp.csr_matrix(features)
        reg_sparse = CloneEnsembleRegressor(
            estimator=LinearRegression(),
            n_estimators=2,
        )
        reg_sparse.fit(x_sparse, y)
        preds_sparse = reg_sparse.predict(x_sparse)
        self.assertIsInstance(preds_sparse, np.ndarray)
        if not isinstance(preds, np.ndarray):
            raise TypeError("Expected predictions to be a numpy array")
        self.assertEqual(preds_sparse.shape, (x_sparse.shape[0],))


class TestCloneEnsembleClassifier(WrappedEstimatorBaseTestMixIn, unittest.TestCase):
    """Unit tests for CloneEnsembleClassifier."""

    @staticmethod
    def get_wrapped_estimator_type() -> type:
        """Return the CloneEnsembleRegressor class.

        Returns
        -------
        type[CloneEnsembleRegressor]
            The class of the wrapped estimator to be tested.

        """
        return CloneEnsembleRegressor

    def test_fit_sample_forwarding(self) -> None:
        """Each classifier clone receives the full training set.

        Raises
        ------
        TypeError
            If any of the clones is not an instance of MockClassifier.

        """
        features = np.array([[i, i, i, i] for i in range(10)])
        y = np.arange(10) % 2
        base: BaseEstimator = MockClassifier()
        ensemble = CloneEnsembleClassifier(estimator=base, n_estimators=3)
        ensemble.fit(features, y)

        self.assertEqual(len(ensemble.estimators_), 3)
        for est in ensemble.estimators_:
            if not isinstance(est, MockClassifier):
                raise TypeError("Expected an instance of MockClassifier")
            self.assertTrue(np.array_equal(est.fit_args["X"], features))
            self.assertTrue(np.array_equal(est.fit_args["y"], y))

    def test_predict_hard_voting(self) -> None:
        """Hard voting returns the most frequent class per sample."""
        features = np.array([[i, i, i, i] for i in range(6)])
        y = np.array([0, 1, 0, 1, 0, 1])
        base: BaseEstimator = MockClassifier()
        ensemble = CloneEnsembleClassifier(
            estimator=base,
            n_estimators=3,
            voting="hard",
        )
        ensemble.fit(features, y)
        preds = ensemble.predict(features)
        self.assertTrue(np.array_equal(preds, np.array([0, 1, 0, 1, 0, 1])))

    def test_predict_soft_voting(self) -> None:
        """Soft voting uses the class with highest mean predicted probability."""
        features = np.array([[i, i, i, i] for i in range(6)])
        y = np.array([0, 1, 0, 1, 0, 1])
        base = MockClassifier()
        ensemble = CloneEnsembleClassifier(
            estimator=base,
            n_estimators=3,
            voting="soft",
        )
        ensemble.fit(features, y)
        preds = ensemble.predict(features)
        self.assertTrue(np.array_equal(preds, np.zeros(len(features))))
        proba = ensemble.predict_proba(features)
        expected_proba = np.tile([0.7, 0.3], (len(features), 1))
        self.assertTrue(np.allclose(proba, expected_proba))

    def test_logistic_regression_dense_and_sparse(self) -> None:
        """Classifier works with both dense arrays and CSR sparse matrices."""
        features = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [1, 2], [2, 1]])
        y = np.array([0, 1, 1, 0, 1, 0])

        # Dense array
        clf = CloneEnsembleClassifier(
            estimator=LogisticRegression(solver="liblinear"),
            n_estimators=2,
        )
        clf.fit(features, y)
        preds = clf.predict(features)
        self.assertEqual(preds.shape, (features.shape[0],))

        # Sparse matrix
        x_sparse = sp.csr_matrix(features)
        clf_sparse = CloneEnsembleClassifier(
            estimator=LogisticRegression(solver="liblinear"),
            n_estimators=2,
        )
        clf_sparse.fit(x_sparse, y)
        preds_sparse = clf_sparse.predict(x_sparse)
        self.assertEqual(preds_sparse.shape, (x_sparse.shape[0],))


if __name__ == "__main__":
    unittest.main()
