"""Unit tests for CloneEnsembleClassifier and CloneEnsembleRegressor."""

import unittest

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.utils._tags import get_tags  # noqa: PLC2701

from molpipeline.experimental.estimators.ensemble.homogeneous_ensemble import (
    HomogeneousEnsembleClassifier,
    HomogeneousEnsembleRegressor,
)
from molpipeline.experimental.model_selection.splitter import (
    BootstrapSplit,
    DataRepetitionSplit,
)
from molpipeline.utils.json_operations import recursive_from_json, recursive_to_json
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

    def test_regressor_has_regressor_estimator_type(self) -> None:
        """HomogeneousEnsembleRegressor is tagged as a regressor by sklearn."""
        reg = HomogeneousEnsembleRegressor(estimator=MockEstimator())
        tags = get_tags(reg)
        self.assertEqual(tags.estimator_type, "regressor")
        self.assertIsNotNone(tags.regressor_tags)

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
            sampler=BootstrapSplit(3, random_state=11),
            estimator__beta=2,
        )
        ensemble.set_params(estimator__gamma=3)
        params = ensemble.get_params(deep=True)
        self.assertIn("estimator__alpha", params)
        self.assertIn("estimator__beta", params)
        self.assertIn("estimator__gamma", params)
        self.assertIn("sampler__n_splits", params)
        self.assertIn("sampler__random_state", params)
        self.assertEqual(params["estimator__alpha"], 1)
        self.assertEqual(params["estimator__beta"], 2)
        self.assertEqual(params["estimator__gamma"], 3)
        self.assertEqual(params["sampler__n_splits"], 3)
        self.assertEqual(params["sampler__random_state"], 11)

    def test_set_params_sampler_updates_regressor_sampler(self) -> None:
        """set_params updates sampler parameters and preserves sampler type.

        Raises
        ------
        TypeError
            If the sampler is not a BootstrapSplit instance.

        """
        ensemble = HomogeneousEnsembleRegressor(
            estimator=MockEstimator(),
            sampler=BootstrapSplit(2, random_state=5),
        )

        ensemble.set_params(sampler__n_splits=7)

        sampler = ensemble.sampler
        self.assertIsInstance(sampler, BootstrapSplit)
        if not isinstance(sampler, BootstrapSplit):
            raise TypeError("Expected BootstrapSplit sampler")
        self.assertEqual(sampler.n_splits, 7)
        self.assertEqual(sampler.random_state, 5)

    def test_clone_regressor(self) -> None:
        """Cloning keeps hyperparameters while creating fresh nested objects.

        Raises
        ------
        TypeError
            If the type is not preserved by clone.

        """
        ensemble = HomogeneousEnsembleRegressor(
            estimator=MockEstimator(alpha=1, beta=2, gamma=3),
            sampler=BootstrapSplit(3, random_state=13),
            n_jobs=2,
            estimator__beta=2,
        )

        ensemble_clone = clone(ensemble)
        if not isinstance(ensemble_clone, HomogeneousEnsembleRegressor):
            raise TypeError("Expected an instance of HomogeneousEnsembleRegressor")
        params = ensemble_clone.get_params(deep=True)

        self.assertIsNot(ensemble_clone, ensemble)
        self.assertIsNot(ensemble_clone.estimator, ensemble.estimator)
        self.assertIsNot(ensemble_clone.sampler, ensemble.sampler)
        self.assertEqual(params["estimator__alpha"], 1)
        self.assertEqual(params["estimator__beta"], 2)
        self.assertEqual(params["estimator__gamma"], 3)
        self.assertEqual(params["sampler__n_splits"], 3)
        self.assertEqual(params["sampler__random_state"], 13)
        self.assertEqual(ensemble_clone.n_jobs, 2)

    def test_serialization_roundtrip(self) -> None:
        """Test the serialization with recursive_to_json and recursive_from_json."""
        ensemble = HomogeneousEnsembleRegressor(
            estimator=MockEstimator(alpha=1, beta=2, gamma=3),
            sampler=BootstrapSplit(3, random_state=13),
            n_jobs=2,
            estimator__beta=2,
        )
        reconstructed_ensemble = recursive_from_json(recursive_to_json(ensemble))
        self.assertEqual(joblib.hash(ensemble), joblib.hash(reconstructed_ensemble))

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

    def test_regressor_grid_search(self) -> None:
        """GridSearchCV works for the regressor with r2 scoring."""
        features, y = make_regression(
            n_samples=50,
            n_features=5,
            random_state=42,
        )
        reg = HomogeneousEnsembleRegressor(
            estimator=LinearRegression(),
            sampler=BootstrapSplit(3, random_state=42),
        )
        grid = GridSearchCV(
            reg,
            param_grid={"sampler__n_splits": [2, 3]},
            scoring="r2",
            cv=3,
        )
        grid.fit(features, y)
        self.assertFalse(np.isnan(grid.best_score_))

    def test_classifier_grid_search_multimetric(self) -> None:
        """Multimetric GridSearchCV works (roc_auc requires predict_proba + tags)."""
        features, y = make_classification(
            n_samples=50,
            n_features=5,
            random_state=42,
        )
        clf = HomogeneousEnsembleClassifier(
            estimator=LogisticRegression(solver="liblinear"),
            sampler=BootstrapSplit(3, random_state=42),
        )
        scoring = {
            "ba": "balanced_accuracy",
            "roc_auc": "roc_auc",
        }
        grid = GridSearchCV(
            clf,
            param_grid={"voting": ["hard", "soft"]},
            scoring=scoring,
            refit="ba",
            cv=3,
        )
        grid.fit(features, y)
        self.assertIn("mean_test_ba", grid.cv_results_)
        self.assertIn("mean_test_roc_auc", grid.cv_results_)
        self.assertFalse(
            np.any(np.isnan(grid.cv_results_["mean_test_ba"])),
        )
        self.assertFalse(
            np.any(np.isnan(grid.cv_results_["mean_test_roc_auc"])),
        )


class TestHomogeneousEnsembleClassifier(unittest.TestCase):
    """Unit tests for wrapped regressors."""

    def setUp(self) -> None:
        """Set up the test parameters."""
        self.test_params = {
            "sampler": sampler_list,
            "voting": ["hard", "soft"],
        }

    def test_classifier_has_classifier_estimator_type(self) -> None:
        """HomogeneousEnsembleClassifier is tagged as a classifier by sklearn."""
        clf = HomogeneousEnsembleClassifier(estimator=MockClassifier())
        tags = get_tags(clf)
        self.assertEqual(tags.estimator_type, "classifier")
        self.assertIsNotNone(tags.classifier_tags)

    def test_param_forwarding(self) -> None:
        """Parameters are forwarded to the wrapped estimator.

        Raises
        ------
        TypeError
            If the base estimator is not an instance of MockClassifier.

        """
        base = MockClassifier(alpha=1)
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
            sampler=BootstrapSplit(4, random_state=19),
            estimator__beta=2,
        )
        ensemble.set_params(estimator__gamma=3, voting="soft")
        params = ensemble.get_params(deep=True)
        self.assertIn("estimator__alpha", params)
        self.assertIn("estimator__beta", params)
        self.assertIn("estimator__gamma", params)
        self.assertIn("sampler__n_splits", params)
        self.assertIn("sampler__random_state", params)
        self.assertIn("voting", params)
        self.assertEqual(params["estimator__alpha"], 1)
        self.assertEqual(params["estimator__beta"], 2)
        self.assertEqual(params["estimator__gamma"], 3)
        self.assertEqual(params["sampler__n_splits"], 4)
        self.assertEqual(params["sampler__random_state"], 19)
        self.assertEqual(params["voting"], "soft")

    def test_set_params_sampler_updates_classifier_sampler(self) -> None:
        """set_params updates classifier sampler parameters and voting.

        Raises
        ------
        TypeError
            If the sampler is not a BootstrapSplit instance.

        """
        ensemble = HomogeneousEnsembleClassifier(
            estimator=MockClassifier(),
            sampler=BootstrapSplit(3, random_state=21),
            voting="hard",
        )

        ensemble.set_params(sampler__n_splits=6, voting="soft")

        sampler = ensemble.sampler
        self.assertIsInstance(sampler, BootstrapSplit)
        if not isinstance(sampler, BootstrapSplit):
            raise TypeError("Expected BootstrapSplit sampler")
        self.assertEqual(sampler.n_splits, 6)
        self.assertEqual(sampler.random_state, 21)
        self.assertEqual(ensemble.voting, "soft")

    def test_clone_classifier(self) -> None:
        """Cloning keeps classifier params and voting with fresh nested objects.

        Raises
        ------
        TypeError
            If the type is not preserved by clone.

        """
        ensemble = HomogeneousEnsembleClassifier(
            estimator=MockClassifier(alpha=4, gamma=8),
            sampler=BootstrapSplit(5, random_state=17),
            voting="soft",
            n_jobs=3,
            estimator__beta=6,
        )
        ensemble_clone = clone(ensemble)
        if not isinstance(ensemble_clone, HomogeneousEnsembleClassifier):
            raise TypeError("Expected an instance of HomogeneousEnsembleClassifier")
        params = ensemble_clone.get_params(deep=True)

        self.assertIsNot(ensemble_clone, ensemble)
        self.assertIsNot(ensemble_clone.estimator, ensemble.estimator)
        self.assertIsNot(ensemble_clone.sampler, ensemble.sampler)
        self.assertEqual(params["estimator__alpha"], 4)
        self.assertEqual(params["estimator__beta"], 6)
        self.assertEqual(params["estimator__gamma"], 8)
        self.assertEqual(params["sampler__n_splits"], 5)
        self.assertEqual(params["sampler__random_state"], 17)
        self.assertEqual(params["voting"], "soft")
        self.assertEqual(ensemble_clone.n_jobs, 3)

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

    def test_classes_set_after_fit(self) -> None:
        """Fitting sets the classes_ attribute from the target labels."""
        features = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
        y = np.array([0, 1, 0, 1])
        clf = HomogeneousEnsembleClassifier(
            estimator=MockClassifier(),
            sampler=DataRepetitionSplit(2),
        )
        clf.fit(features, y)
        self.assertTrue(hasattr(clf, "classes_"))
        self.assertTrue(np.array_equal(clf.classes_, np.array([0, 1])))

    def test_classifier_grid_search_with_roc_auc(self) -> None:
        """GridSearchCV with roc_auc scoring works (requires predict_proba + tags)."""
        features, y = make_classification(
            n_samples=50,
            n_features=5,
            random_state=42,
        )
        clf = HomogeneousEnsembleClassifier(
            estimator=LogisticRegression(solver="liblinear"),
            sampler=BootstrapSplit(3, random_state=42),
        )
        grid = GridSearchCV(
            clf,
            param_grid={"voting": ["hard", "soft"]},
            scoring="roc_auc",
            cv=3,
        )
        grid.fit(features, y)
        self.assertFalse(np.isnan(grid.best_score_))

    def test_classifier_grid_search_with_balanced_accuracy(self) -> None:
        """GridSearchCV with balanced_accuracy scoring works for the classifier."""
        features, y = make_classification(
            n_samples=50,
            n_features=5,
            random_state=42,
        )
        clf = HomogeneousEnsembleClassifier(
            estimator=LogisticRegression(solver="liblinear"),
            sampler=BootstrapSplit(3, random_state=42),
        )
        grid = GridSearchCV(
            clf,
            param_grid={"voting": ["hard", "soft"]},
            scoring="balanced_accuracy",
            cv=3,
        )
        grid.fit(features, y)
        self.assertFalse(np.isnan(grid.best_score_))


if __name__ == "__main__":
    unittest.main()
