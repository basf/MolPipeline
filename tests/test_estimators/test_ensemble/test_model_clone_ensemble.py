"""Unit tests for CloneEnsembleClassifier and CloneEnsembleRegressor."""

import unittest
from typing import Self

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from typing_extensions import override

from molpipeline.estimators.ensemble.model_clone_ensemble import (
    CloneEnsembleClassifier,
    CloneEnsembleRegressor,
)


class MockEstimator(BaseEstimator):
    """A mock estimator that records fit arguments and returns fixed predictions."""

    fit_args: dict[str, npt.ArrayLike]

    def __init__(self, alpha: int = 0, beta: int = 0, gamma: int = 0) -> None:
        """Initialize the MockEstimator with dummy parameters.

        Parameters
        ----------
        alpha : int, default=0
            A dummy parameter to test parameter forwarding.
        beta : int, default=0
            A dummy parameter to test parameter forwarding.
        gamma : int, default=0
            A dummy parameter to test parameter forwarding.

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
            The feature matrix used for fitting.
        y : npt.ArrayLike
            The target vector used for fitting.

        Returns
        -------
        Self
            The fitted estimator with recorded fit arguments.

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
            The feature matrix used for prediction.

        Returns
        -------
        npt.NDArray[np.float64]
            An array of zeros with the same length as the number of samples in X.

        """
        feature_arr = np.asarray(X)
        return np.zeros(len(feature_arr), dtype=np.float64)


class MockClassifier(MockEstimator):
    """A mock classifier that returns deterministic class predictions."""

    @override
    def predict(  # type: ignore
        self,
        X: npt.ArrayLike,  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.int64]:
        """Return alternating class predictions.

        Parameters
        ----------
        X : npt.ArrayLike
            The feature matrix used for prediction.

        Returns
        -------
        npt.NDArray[np.int64]
            An array of class predictions (0 or 1) based on the first feature of X,
            alternating between 0 and 1.

        """
        feature_arr = np.asarray(X)
        return np.array([x[0] % 2 for x in feature_arr], dtype=np.int64)

    def predict_proba(  # noqa: PLR6301
        self,
        X: npt.ArrayLike,  # noqa: N803 # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float64]:
        """Return fixed class probabilities.

        Parameters
        ----------
        X : npt.ArrayLike
            The feature matrix used for probability prediction.

        Returns
        -------
        npt.NDArray[np.float64]
            An array of shape (n_samples, 2) where the first column is 0.7 and the
            second column is 0.3 for all samples.

        """
        feature_arr = np.asarray(X)
        proba = np.zeros((len(feature_arr), 2))
        proba[:, 0] = 0.7
        proba[:, 1] = 0.3
        return proba


class TestCloneEnsembleRegressor(unittest.TestCase):
    """Unit tests for CloneEnsembleRegressor."""

    def test_param_forwarding(self) -> None:
        """Parameters are forwarded to the wrapped estimator.

        Raises
        ------
        TypeError
            If the base estimator is not an instance of MockClassifier.

        """
        base = MockEstimator(alpha=1)
        ensemble = CloneEnsembleRegressor(
            estimator=base,
            n_estimators=3,
            estimator__beta=2,
        )
        ensemble.set_params(estimator__gamma=3)
        base_est = ensemble.estimator
        if not isinstance(base_est, MockEstimator):
            raise TypeError("Expected an instance of MockEstimator")
        self.assertEqual(base_est.alpha, 1)
        self.assertEqual(base_est.beta, 2)
        self.assertEqual(base_est.gamma, 3)

    def test_get_params(self) -> None:
        """get_params exposes nested estimator parameters."""
        base = MockEstimator(alpha=1)
        ensemble = CloneEnsembleRegressor(
            estimator=base,
            n_estimators=3,
            estimator__beta=2,
        )
        ensemble.set_params(estimator__gamma=3)
        params = ensemble.get_params(deep=True)
        self.assertIn("estimator__alpha", params)
        self.assertIn("estimator__beta", params)
        self.assertIn("estimator__gamma", params)
        self.assertEqual(params["estimator__alpha"], 1)
        self.assertEqual(params["estimator__beta"], 2)
        self.assertEqual(params["estimator__gamma"], 3)

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
        """Regressor works with both dense arrays and CSR sparse matrices."""
        features = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [1, 2], [2, 1]])
        y = np.array([0.0, 1.0, 1.0, 0.0, 2.0, 1.0])

        # Dense array
        reg = CloneEnsembleRegressor(estimator=LinearRegression(), n_estimators=2)
        reg.fit(features, y)
        preds = reg.predict(features)
        self.assertEqual(preds.shape, (features.shape[0],))

        # Sparse matrix
        x_sparse = sp.csr_matrix(features)
        reg_sparse = CloneEnsembleRegressor(
            estimator=LinearRegression(),
            n_estimators=2,
        )
        reg_sparse.fit(x_sparse, y)
        preds_sparse = reg_sparse.predict(x_sparse)
        self.assertEqual(preds_sparse.shape, (x_sparse.shape[0],))


class TestCloneEnsembleClassifier(unittest.TestCase):
    """Unit tests for CloneEnsembleClassifier."""

    def test_param_forwarding(self) -> None:
        """Parameters are forwarded to the wrapped estimator.

        Raises
        ------
        TypeError
            If the base estimator is not an instance of MockClassifier.

        """
        base = MockClassifier(alpha=1)
        ensemble = CloneEnsembleClassifier(
            estimator=base,
            n_estimators=3,
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
        """Each classifier clone receives the full training set.

        Raises
        ------
        TypeError
            If any of the clones is not an instance of MockClassifier.

        """
        features = np.array([[i, i, i, i] for i in range(10)])
        y = np.arange(10) % 2
        base = MockClassifier()
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
        base = MockClassifier()
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
