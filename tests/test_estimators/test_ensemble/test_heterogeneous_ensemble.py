"""Unit tests for CloneEnsembleClassifier and CloneEnsembleRegressor."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import ParameterGrid

from molpipeline.estimators.ensemble.homogeneous_ensemble import (
    HomogeneousEnsembleClassifier,
    HomogeneousEnsembleRegressor,
)
from molpipeline.experimental.model_selection.splitter import (
    BootstrapSplit,
    DataRepetitionSplit,
)
from tests.utils.mock_estimators import (
    MockClassifier,
    MockEstimator,
)

sampler_list = [DataRepetitionSplit(3), BootstrapSplit(3, random_state=20160316)]


class TestHomogeneousEnsembleRegressor(unittest.TestCase):
    """Unit tests for wrapped regressors."""

    def setUp(self) -> None:
        """Set up the parameters for the unit tests."""
        self.test_params = {
            "sampler": sampler_list,
        }

    def test_param_forwarding(self) -> None:
        """Parameters are forwarded to the wrapped estimator.

        Raises
        ------
        TypeError
            If the base estimator is not an instance of MockClassifier.

        """
        base = MockEstimator(alpha=1)
        ensemble = HomogeneousEnsembleRegressor(
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
        ensemble = HomogeneousEnsembleRegressor(
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

    def test_fit_sample_forwarding(self) -> None:
        """Each clone receives the full feature matrix and target vector.

        Raises
        ------
        TypeError
            If the base estimator is not an instance of MockClassifier.

        """
        features = np.array([[i, i, i, i] for i in range(10)])
        y = np.arange(10)
        for parameters in ParameterGrid(self.test_params):
            sampler = parameters["sampler"]
            ensemble = HomogeneousEnsembleRegressor(
                estimator=MockEstimator(),
                **parameters,
            )
            ensemble.fit(features, y)
            self.assertEqual(len(ensemble.estimators_), sampler.get_n_splits(features))
            splits = list(sampler.split(features, y))
            for est, split in zip(ensemble.estimators_, splits, strict=True):
                if not isinstance(est, MockEstimator):
                    raise TypeError("Expected an instance of MockEstimator")
                self.assertTrue(np.array_equal(est.fit_args["X"], features[split[0]]))
                self.assertTrue(np.array_equal(est.fit_args["y"], y[split[0]]))

    def test_fit_sample_forwarding_with_lists(self) -> None:
        """List inputs are handled and forwarded to every clone.

        Raises
        ------
        TypeError
            If the base estimator is not an instance of MockClassifier.

        """
        features = [[i, i, i, i] for i in range(10)]
        y = [float(i) for i in range(10)]
        for parameters in ParameterGrid(self.test_params):
            sampler = parameters["sampler"]
            ensemble = HomogeneousEnsembleRegressor(
                estimator=MockEstimator(),
                **parameters,
            )
            ensemble.fit(features, y)

            self.assertEqual(len(ensemble.estimators_), sampler.get_n_splits(features))
            splits = list(sampler.split(features, y))
            for est, split in zip(ensemble.estimators_, splits, strict=True):
                if not isinstance(est, MockEstimator):
                    raise TypeError("Expected an instance of MockEstimator")
                self.assertTrue(
                    np.allclose(est.fit_args["X"], np.asarray(features)[split[0]]),
                )
                self.assertTrue(np.allclose(est.fit_args["y"], np.asarray(y)[split[0]]))

    def test_linear_regression_dense_and_sparse(self) -> None:
        """Regressor works with both dense arrays and CSR sparse matrices."""
        features = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [1, 2], [2, 1]])
        y = np.array([0.0, 1.0, 1.0, 0.0, 2.0, 1.0])

        # Dense array
        for parameters in ParameterGrid(self.test_params):
            reg = HomogeneousEnsembleRegressor(
                estimator=LinearRegression(),
                **parameters,
            )
            reg.fit(features, y)
            preds_dense = reg.predict(features)
            self.assertIsInstance(preds_dense, np.ndarray)
            self.assertEqual(preds_dense.shape, (features.shape[0],))

            # Sparse matrix
            x_sparse = csr_matrix(features)
            reg_sparse = HomogeneousEnsembleRegressor(
                estimator=LinearRegression(),
                **parameters,
            )
            reg_sparse.fit(x_sparse, y)
            preds_sparse = reg_sparse.predict(x_sparse)
            self.assertIsInstance(preds_sparse, np.ndarray)
            self.assertEqual(preds_sparse.shape, (x_sparse.shape[0],))

            self.assertTrue(np.allclose(preds_dense, preds_sparse))


class TestHomogeneousEnsembleClassifier(unittest.TestCase):
    """Unit tests for wrapped regressors."""

    def setUp(self) -> None:
        """Set up the test parameters."""
        self.test_params = {
            "sampler": sampler_list,
            "voting": ["hard", "soft"],
        }

    def test_param_forwarding(self) -> None:
        """Parameters are forwarded to the wrapped estimator.

        Raises
        ------
        TypeError
            If the base estimator is not an instance of MockClassifier.

        """
        base = MockEstimator(alpha=1)
        ensemble = HomogeneousEnsembleClassifier(
            estimator=base,
            estimator__beta=2,
        )
        ensemble.set_params(estimator__gamma=3)
        base_est = ensemble.estimator
        if not isinstance(base_est, MockClassifier):
            raise TypeError("Expected an instance of MockClassifier")
        self.assertEqual(base_est.alpha, 1)
        self.assertEqual(base_est.beta, 2)
        self.assertEqual(base_est.gamma, 3)

    def test_get_params(self) -> None:
        """get_params exposes nested estimator parameters."""
        base = MockEstimator(alpha=1)
        ensemble = HomogeneousEnsembleClassifier(
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

    def test_fit_sample_forwarding(self) -> None:
        """Each classifier clone receives the full training set.

        Raises
        ------
        TypeError
            If any of the clones is not an instance of MockClassifier.

        """
        features = np.array([[i, i, i, i] for i in range(10)])
        y = np.arange(10) % 2
        for parameters in ParameterGrid(self.test_params):
            sampler = parameters["sampler"]
            ensemble = HomogeneousEnsembleClassifier(
                estimator=MockClassifier(),
                **parameters,
            )
            ensemble.fit(features, y)

            self.assertEqual(len(ensemble.estimators_), sampler.get_n_splits(features))
            splits = list(sampler.split(features, y))
            for est, split in zip(ensemble.estimators_, splits, strict=True):
                if not isinstance(est, MockClassifier):
                    raise TypeError("Expected an instance of MockClassifier")
                self.assertTrue(np.array_equal(est.fit_args["X"], features[split[0]]))
                self.assertTrue(np.array_equal(est.fit_args["y"], y[split[0]]))

    def test_predict(self) -> None:
        """Hard voting returns the most frequent class per sample."""
        y = np.array([0, 1, 0, 1, 0, 1])
        features = np.array([[i, i, i, i] for i in y])
        test_params = dict(self.test_params)
        test_params.pop("voting")
        for parameters in ParameterGrid(test_params):
            base = MockClassifier()
            ensemble = HomogeneousEnsembleClassifier(estimator=base, **parameters)
            ensemble.fit(features, y)
            preds = ensemble.predict(features)
            self.assertTrue(np.allclose(preds, y))

    def test_predict_proba(self) -> None:
        """predict_proba returns the mean predicted probabilities of the clones."""
        y = np.array([0, 1, 0, 1, 0, 1])
        features = np.array([[i, i, i, i] for i in y])
        for parameters in ParameterGrid(self.test_params):
            base = MockClassifier()
            ensemble = HomogeneousEnsembleClassifier(estimator=base, **parameters)
            ensemble.fit(features, y)
            proba = ensemble.predict_proba(features)
            expected_proba = np.abs(np.array([[yi - 0.7, yi - 0.3] for yi in y]))
            self.assertTrue(np.allclose(proba, expected_proba))

    def test_predict_soft_voting(self) -> None:
        """Soft voting uses the class with the highest mean predicted probability."""
        y = np.array([0, 1, 0, 1, 0, 1])
        features = np.array([[i, i, i, i] for i in y])
        test_params = dict(self.test_params)
        test_params.pop("voting")
        for parameters in ParameterGrid(test_params):
            ensemble = HomogeneousEnsembleClassifier(
                estimator=MockClassifier(),
                voting="soft",
                **parameters,
            )
            ensemble.fit(features, y)
            preds = ensemble.predict(features)
            self.assertTrue(np.allclose(preds, y))
            proba = ensemble.predict_proba(features)
            expected_proba = np.abs(np.array([[yi - 0.7, yi - 0.3] for yi in y]))
            self.assertTrue(np.allclose(proba, expected_proba))

    def test_logistic_regression_dense_and_sparse(self) -> None:
        """Classifier works with both dense arrays and CSR sparse matrices."""
        features, y = make_classification(random_state=20260316, shift=0)
        bin_features = np.array(np.array(features) > 0, dtype=np.int64)

        for parameters in ParameterGrid(self.test_params):
            # Dense array
            clf = HomogeneousEnsembleClassifier(
                estimator=LogisticRegression(solver="liblinear"),
                **parameters,
            )
            clf.fit(bin_features, y)
            preds_dense = clf.predict(bin_features)
            self.assertEqual(preds_dense.shape, (bin_features.shape[0],))

            # Sparse matrix
            x_sparse = csr_matrix(bin_features)
            clf_sparse = HomogeneousEnsembleClassifier(
                estimator=LogisticRegression(solver="liblinear"),
                **parameters,
            )
            clf_sparse.fit(x_sparse, y)
            preds_sparse = clf_sparse.predict(x_sparse)
            self.assertEqual(preds_sparse.shape, (x_sparse.shape[0],))

            self.assertTrue(np.allclose(preds_dense, preds_sparse))


if __name__ == "__main__":
    unittest.main()
