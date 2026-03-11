"""Template tests for wrapped estimators."""

import abc
import unittest
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import ParameterGrid

from tests.utils.mock_estimators import (
    MockClassifier,
    MockEstimator,
)


class TestMixIn(abc.ABC):
    """Base class for test mix-ins.

    This defines the interface of unit-tests to avoid ignoring mypy and pylint.

    """

    @abc.abstractmethod
    def assertEqual(self, first: Any, second: Any, msg: str | None = None) -> None:  # noqa: N802  # pylint: disable=invalid-name
        """Assert that two values are equal.

        Parameters
        ----------
        first: Any
            The first value to compare.
        second: Any
            The second value to compare.
        msg: str | None
            Optional message to display on failure.

        """

    @abc.abstractmethod
    def assertIn(self, member: Any, container: Any, msg: str | None = None) -> None:  # noqa: N802  # pylint: disable=invalid-name
        """Assert that a value is in a container.

        Parameters
        ----------
        member: Any
            The value to check for membership.
        container: Any
            The container to check for membership.
        msg: str | None
            Optional message to display on failure.

        """

    @abc.abstractmethod
    def assertTrue(self, expr: Any, msg: str | None = None) -> None:  # noqa: N802  # pylint: disable=invalid-name
        """Assert that an expression is true.

        Parameters
        ----------
        expr: Any
            The expression to evaluate.
        msg: str | None
            Optional message to display on failure.

        """

    @abc.abstractmethod
    def assertIsInstance(self, obj: Any, cls: Any, msg: str | None = None) -> None:  # noqa: N802  # pylint: disable=invalid-name
        """Assert that an object is an instance of a class.

        Parameters
        ----------
        obj: Any
            The object to check.
        cls: Any
            The class to check against.
        msg: str | None
            Optional message to display on failure.

        """


class WrappedEstimatorBaseTestMixIn(TestMixIn, abc.ABC):
    """Unit tests for CloneEnsembleRegressor."""

    @staticmethod
    @abc.abstractmethod
    def get_wrapped_estimator_type() -> type:
        """Construct the wrapped estimator to be tested.

        Returns
        -------
        type
            The class of the wrapped estimator to be tested.

        """

    @staticmethod
    @abc.abstractmethod
    def get_test_parameters() -> dict[str, Any]:
        """Return a dictionary of parameters to be used for testing.

        Returns
        -------
        dict[str, Any]
            A dictionary of parameters to be used for testing.

        """

    def test_param_forwarding(self) -> None:
        """Parameters are forwarded to the wrapped estimator.

        Raises
        ------
        TypeError
            If the base estimator is not an instance of MockClassifier.

        """
        base = MockEstimator(alpha=1)
        estimator_class = self.get_wrapped_estimator_type()
        ensemble = estimator_class(
            estimator=base,
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
        estimator_class = self.get_wrapped_estimator_type()
        ensemble = estimator_class(
            estimator=base,
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


class WrappedRegressorBaseTestMixIn(WrappedEstimatorBaseTestMixIn, abc.ABC):
    """Unit tests for wrapped regressors."""

    def test_fit_sample_forwarding(self) -> None:
        """Each clone receives the full feature matrix and target vector.

        Raises
        ------
        TypeError
            If the base estimator is not an instance of MockClassifier.

        """
        estimator_class = self.get_wrapped_estimator_type()
        estimator_params = self.get_test_parameters()
        features = np.array([[i, i, i, i] for i in range(10)])
        y = np.arange(10)
        for parameters in ParameterGrid(estimator_params):
            base = MockEstimator()
            ensemble = estimator_class(estimator=base, **parameters)
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
        estimator_class = self.get_wrapped_estimator_type()
        estimator_params = self.get_test_parameters()
        features = [[i, i, i, i] for i in range(10)]
        y = [float(i) for i in range(10)]
        for parameters in ParameterGrid(estimator_params):
            base = MockEstimator()
            ensemble = estimator_class(estimator=base, **parameters)
            ensemble.fit(features, y)

            self.assertEqual(len(ensemble.estimators_), 3)
            for est in ensemble.estimators_:
                if not isinstance(est, MockEstimator):
                    raise TypeError("Expected an instance of MockEstimator")
                self.assertEqual(est.fit_args["X"], features)
                self.assertEqual(est.fit_args["y"], y)

    def test_linear_regression_dense_and_sparse(self) -> None:
        """Regressor works with both dense arrays and CSR sparse matrices."""
        estimator_class = self.get_wrapped_estimator_type()
        estimator_params = self.get_test_parameters()
        features = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [1, 2], [2, 1]])
        y = np.array([0.0, 1.0, 1.0, 0.0, 2.0, 1.0])

        # Dense array
        for parameters in ParameterGrid(estimator_params):
            reg = estimator_class(estimator=LinearRegression(), **parameters)
            reg.fit(features, y)
            preds_dense = reg.predict(features)
            self.assertIsInstance(preds_dense, np.ndarray)
            self.assertEqual(preds_dense.shape, (features.shape[0],))

            # Sparse matrix
            x_sparse = csr_matrix(features)
            reg_sparse = estimator_class(
                estimator=LinearRegression(),
                **parameters,
            )
            reg_sparse.fit(x_sparse, y)
            preds_sparse = reg_sparse.predict(x_sparse)
            self.assertIsInstance(preds_sparse, np.ndarray)
            self.assertEqual(preds_sparse.shape, (x_sparse.shape[0],))

            self.assertTrue(np.array_equal(preds_dense, preds_sparse))


class WrappedClassifierBaseTestMixIn(WrappedEstimatorBaseTestMixIn, abc.ABC):
    """Unit tests for wrapped classifiers."""

    def test_fit_sample_forwarding(self) -> None:
        """Each classifier clone receives the full training set.

        Raises
        ------
        TypeError
            If any of the clones is not an instance of MockClassifier.

        """
        estimator_class = self.get_wrapped_estimator_type()
        estimator_params = self.get_test_parameters()
        features = np.array([[i, i, i, i] for i in range(10)])
        y = np.arange(10) % 2
        for parameters in ParameterGrid(estimator_params):
            base = MockClassifier()
            ensemble = estimator_class(estimator=base, **parameters)
            ensemble.fit(features, y)

            self.assertEqual(len(ensemble.estimators_), 3)
            for est in ensemble.estimators_:
                if not isinstance(est, MockClassifier):
                    raise TypeError("Expected an instance of MockClassifier")
                self.assertTrue(np.array_equal(est.fit_args["X"], features))
                self.assertTrue(np.array_equal(est.fit_args["y"], y))

    def test_predict(self) -> None:
        """Hard voting returns the most frequent class per sample."""
        estimator_class = self.get_wrapped_estimator_type()
        estimator_params = self.get_test_parameters()
        features = np.array([[i, i, i, i] for i in range(6)])
        y = np.array([0, 1, 0, 1, 0, 1])
        for parameters in ParameterGrid(estimator_params):
            base = MockClassifier()
            ensemble = estimator_class(estimator=base, **parameters)
            ensemble.fit(features, y)
            preds = ensemble.predict(features)
            self.assertTrue(np.array_equal(preds, np.array([0, 1, 0, 1, 0, 1])))

    def test_predict_proba(self) -> None:
        """predict_proba returns the mean predicted probabilities of the clones."""
        estimator_class = self.get_wrapped_estimator_type()
        estimator_params = self.get_test_parameters()
        features = np.array([[i, i, i, i] for i in range(6)])
        y = np.array([0, 1, 0, 1, 0, 1])
        for parameters in ParameterGrid(estimator_params):
            base = MockClassifier()
            ensemble = estimator_class(estimator=base, **parameters)
            ensemble.fit(features, y)
            proba = ensemble.predict_proba(features)
            expected_proba = np.tile([0.7, 0.3], (len(features), 1))
            self.assertTrue(np.allclose(proba, expected_proba))

    @unittest.skip("Skip for now")
    def test_predict_soft_voting(self) -> None:
        """Soft voting uses the class with highest mean predicted probability."""
        estimator_class = self.get_wrapped_estimator_type()
        estimator_params = self.get_test_parameters()
        features = np.array([[i, i, i, i] for i in range(6)])
        y = np.array([0, 1, 0, 1, 0, 1])
        for parameters in ParameterGrid(estimator_params):
            base = MockClassifier()
            ensemble = estimator_class(estimator=base, voting="soft", **parameters)
            ensemble.fit(features, y)
            preds = ensemble.predict(features)
            self.assertTrue(np.array_equal(preds, np.zeros(len(features))))
            proba = ensemble.predict_proba(features)
            expected_proba = np.tile([0.7, 0.3], (len(features), 1))
            self.assertTrue(np.allclose(proba, expected_proba))

    def test_logistic_regression_dense_and_sparse(self) -> None:
        """Classifier works with both dense arrays and CSR sparse matrices."""
        estimator_class = self.get_wrapped_estimator_type()
        estimator_params = self.get_test_parameters()
        features = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [1, 2], [2, 1]])
        y = np.array([0, 1, 1, 0, 1, 0])

        for parameters in ParameterGrid(estimator_params):
            # Dense array
            clf = estimator_class(
                estimator=LogisticRegression(solver="liblinear"),
                **parameters,
            )
            clf.fit(features, y)
            preds_dense = clf.predict(features)
            self.assertEqual(preds_dense.shape, (features.shape[0],))

            # Sparse matrix
            x_sparse = csr_matrix(features)
            clf_sparse = estimator_class(
                estimator=LogisticRegression(solver="liblinear"),
                **parameters,
            )
            clf_sparse.fit(x_sparse, y)
            preds_sparse = clf_sparse.predict(x_sparse)
            self.assertEqual(preds_sparse.shape, (x_sparse.shape[0],))

            self.assertTrue(np.array_equal(preds_dense, preds_sparse))
