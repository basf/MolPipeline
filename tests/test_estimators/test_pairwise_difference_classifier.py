"""Tests for pairwise difference learner utility functions."""

import unittest

import numpy as np
import numpy.typing as npt
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

from molpipeline.estimators.pairwise_difference_learner import (
    PairwiseDifferenceClassifier,
    PairwiseDifferenceRegressor,
    dual_vector_combinations,
    single_vector_combinations,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Two 2-row, 2-feature matrices used for dual combinations.
_V1: npt.NDArray[np.int_] = np.array([[1, 2], [3, 4]])
_V2: npt.NDArray[np.int_] = np.array([[5, 6], [7, 8]])

# A 3-row, 2-feature matrix used for single combinations.
_V: npt.NDArray[np.int_] = np.array([[1, 2], [3, 4], [5, 6]])


class TestDualVectorCombinations(unittest.TestCase):
    """Tests for dual_vector_combinations."""

    def test_combine_mode_values(self) -> None:
        """Concatenated rows must equal the Cartesian product of the two matrices."""
        result = dual_vector_combinations(_V1, _V2, mode="combine")
        expected = np.array(
            [
                [1, 2, 5, 6],
                [1, 2, 7, 8],
                [3, 4, 5, 6],
                [3, 4, 7, 8],
            ],
        )
        self.assertTrue(np.array_equal(result, expected))

    def test_combine_mode_shape(self) -> None:
        """Output shape must be (n1*n2, 2*n_features)."""
        result = dual_vector_combinations(_V1, _V2, mode="combine")
        self.assertEqual(result.shape, (4, 4))

    def test_diff_mode_values(self) -> None:
        """Row differences must equal a1 - a2 for every (a1, a2) in product(V1, V2)."""
        result = dual_vector_combinations(_V1, _V2, mode="diff")
        expected = np.array(
            [
                [1 - 5, 2 - 6],
                [1 - 7, 2 - 8],
                [3 - 5, 4 - 6],
                [3 - 7, 4 - 8],
            ],
        )
        self.assertTrue(np.array_equal(result, expected))

    def test_diff_mode_shape(self) -> None:
        """Output shape must be (n1*n2, n_features)."""
        result = dual_vector_combinations(_V1, _V2, mode="diff")
        self.assertEqual(result.shape, (4, 2))

    def test_combine_and_diff_mode_values(self) -> None:
        """Rows must equal [a1, a2, a1-a2] for every (a1, a2) in product(V1, V2)."""
        result = dual_vector_combinations(_V1, _V2, mode="combine_and_diff")
        expected = np.array(
            [
                [1, 2, 5, 6, 1 - 5, 2 - 6],
                [1, 2, 7, 8, 1 - 7, 2 - 8],
                [3, 4, 5, 6, 3 - 5, 4 - 6],
                [3, 4, 7, 8, 3 - 7, 4 - 8],
            ],
        )
        self.assertTrue(np.array_equal(result, expected))

    def test_combine_and_diff_mode_shape(self) -> None:
        """Output shape must be (n1*n2, 3*n_features)."""
        result = dual_vector_combinations(_V1, _V2, mode="combine_and_diff")
        self.assertEqual(result.shape, (4, 6))

    def test_invalid_mode_raises(self) -> None:
        """An unsupported mode string must raise ValueError."""
        with self.assertRaises(ValueError):
            dual_vector_combinations(_V1, _V2, mode="invalid")  # type: ignore[arg-type]

    def test_default_mode_is_combine(self) -> None:
        """Calling without explicit mode must behave identically to mode='combine'."""
        result_default = dual_vector_combinations(_V1, _V2)
        result_explicit = dual_vector_combinations(_V1, _V2, mode="combine")
        self.assertTrue(np.array_equal(result_default, result_explicit))


class TestSingleVectorCombinations(unittest.TestCase):
    """Tests for single_vector_combinations."""

    def test_combine_mode_values(self) -> None:
        """Concatenated rows must equal every unique unordered pair (no self-pairs)."""
        result = single_vector_combinations(_V, mode="combine")
        expected = np.array(
            [
                [1, 2, 3, 4],
                [1, 2, 5, 6],
                [3, 4, 5, 6],
            ],
        )
        self.assertTrue(np.array_equal(result, expected))

    def test_combine_mode_shape(self) -> None:
        """Output row count must be C(n, 2) = n*(n-1)/2."""
        n = _V.shape[0]
        expected_rows = n * (n - 1) // 2
        result = single_vector_combinations(_V, mode="combine")
        self.assertEqual(result.shape, (expected_rows, 4))

    def test_diff_mode_values(self) -> None:
        """Row differences must equal a1 - a2 for every unique pair."""
        result = single_vector_combinations(_V, mode="diff")
        expected = np.array(
            [
                [1 - 3, 2 - 4],
                [1 - 5, 2 - 6],
                [3 - 5, 4 - 6],
            ],
        )
        self.assertTrue(np.array_equal(result, expected))

    def test_diff_mode_shape(self) -> None:
        """Output shape must be (C(n,2), n_features)."""
        n = _V.shape[0]
        expected_rows = n * (n - 1) // 2
        result = single_vector_combinations(_V, mode="diff")
        self.assertEqual(result.shape, (expected_rows, 2))

    def test_combine_and_diff_mode_values(self) -> None:
        """Rows must equal [a1, a2, a1-a2] for every unique pair."""
        result = single_vector_combinations(_V, mode="combine_and_diff")
        expected = np.array(
            [
                [1, 2, 3, 4, 1 - 3, 2 - 4],
                [1, 2, 5, 6, 1 - 5, 2 - 6],
                [3, 4, 5, 6, 3 - 5, 4 - 6],
            ],
        )
        self.assertTrue(np.array_equal(result, expected))

    def test_combine_and_diff_mode_shape(self) -> None:
        """Output shape must be (C(n,2), 3*n_features)."""
        n = _V.shape[0]
        expected_rows = n * (n - 1) // 2
        result = single_vector_combinations(_V, mode="combine_and_diff")
        self.assertEqual(result.shape, (expected_rows, 6))

    def test_no_self_pairs(self) -> None:
        """Result must contain fewer rows than dual_vector_combinations(V, V)."""
        single_result = single_vector_combinations(_V, mode="combine")
        dual_result = dual_vector_combinations(_V, _V, mode="combine")
        # dual includes self-pairs so it must have strictly more rows
        self.assertGreater(dual_result.shape[0], single_result.shape[0])

    def test_invalid_mode_raises(self) -> None:
        """An unsupported mode string must raise ValueError."""
        with self.assertRaises(ValueError):
            single_vector_combinations(_V, mode="invalid")  # type: ignore[arg-type]

    def test_default_mode_is_combine(self) -> None:
        """Calling without explicit mode must behave identically to mode='combine'."""
        result_default = single_vector_combinations(_V)
        result_explicit = single_vector_combinations(_V, mode="combine")
        self.assertTrue(np.array_equal(result_default, result_explicit))


class TestPairwiseDifferenceRegressor(unittest.TestCase):
    """Tests for PairwiseDifferenceRegressor using make_regression."""

    def setUp(self) -> None:
        """Create a small regression dataset shared across all tests."""
        self.X: npt.NDArray[np.float64]
        self.y: npt.NDArray[np.float64]
        self.X, self.y = make_regression(
            n_samples=20,
            n_features=5,
            noise=0.1,
            random_state=0,
        )

    def test_fit_predict_shape(self) -> None:
        """Predictions must have the same length as the input."""
        model = PairwiseDifferenceRegressor(estimator=LinearRegression())
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        self.assertEqual(y_pred.shape[0], len(self.X))

    def test_predict_return_std_shapes(self) -> None:
        """return_std=True must return two arrays of length n_samples."""
        model = PairwiseDifferenceRegressor(estimator=LinearRegression())
        model.fit(self.X, self.y)
        mean_pred, std_pred = model.predict(self.X, return_std=True)
        self.assertEqual(mean_pred.shape[0], len(self.X))
        self.assertEqual(std_pred.shape[0], len(self.X))

    def test_std_non_negative(self) -> None:
        """All returned standard deviations must be >= 0."""
        model = PairwiseDifferenceRegressor(estimator=LinearRegression())
        model.fit(self.X, self.y)
        _, std_pred = model.predict(self.X, return_std=True)
        self.assertTrue(bool(np.all(std_pred >= 0)))

    def test_all_modes(self) -> None:
        """Every combination mode must produce predictions of the correct shape."""
        for mode in ("combine", "diff", "combine_and_diff"):
            with self.subTest(mode=mode):
                model = PairwiseDifferenceRegressor(
                    estimator=LinearRegression(),
                    mode=mode,
                )
                model.fit(self.X, self.y)
                y_pred = model.predict(self.X)
                self.assertEqual(y_pred.shape[0], len(self.X))


class TestPairwiseDifferenceClassifier(unittest.TestCase):
    """Tests for PairwiseDifferenceClassifier.

    Uses make_classification (binary and multiclass) and
    make_multilabel_classification to exercise the classifier.
    """

    def setUp(self) -> None:
        """Create small classification datasets shared across all tests."""
        self.X_bin: npt.NDArray[np.float64]
        self.y_bin: npt.NDArray[np.int_]
        self.X_bin, self.y_bin = make_classification(
            n_samples=20,
            n_features=5,
            n_classes=2,
            n_informative=3,
            n_redundant=2,
            random_state=0,
        )

        self.X_multi: npt.NDArray[np.float64]
        self.y_multi: npt.NDArray[np.int_]
        self.X_multi, self.y_multi = make_classification(
            n_samples=30,
            n_features=5,
            n_classes=3,
            n_informative=3,
            n_redundant=2,
            random_state=0,
        )

        self.X_ml: npt.NDArray[np.float64]
        self.Y_ml: npt.NDArray[np.int_]
        self.X_ml, self.Y_ml = make_multilabel_classification(
            n_samples=20,
            n_features=5,
            n_classes=3,
            random_state=0,
        )

    # ------------------------------------------------------------------
    # Binary classification
    # ------------------------------------------------------------------

    def test_fit_predict_binary_shape(self) -> None:
        """Predicted labels must have length n_samples for binary classification."""
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        model.fit(self.X_bin, self.y_bin)
        y_pred = model.predict(self.X_bin)
        self.assertEqual(len(y_pred), len(self.X_bin))

    def test_fit_predict_binary_labels_subset(self) -> None:
        """Predicted labels must be a subset of the training classes."""
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        model.fit(self.X_bin, self.y_bin)
        y_pred = model.predict(self.X_bin)
        self.assertTrue(set(y_pred).issubset(set(self.y_bin)))

    def test_predict_proba_binary_shape(self) -> None:
        """predict_proba must return shape (n_samples, n_classes) for binary case."""
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        model.fit(self.X_bin, self.y_bin)
        proba = model.predict_proba(self.X_bin)
        self.assertEqual(proba.shape, (len(self.X_bin), 2))

    # ------------------------------------------------------------------
    # Multiclass classification (make_classification, n_classes=3)
    # ------------------------------------------------------------------

    def test_fit_predict_multiclass_shape(self) -> None:
        """Predicted labels must have length n_samples for multiclass classification."""
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        model.fit(self.X_multi, self.y_multi)
        y_pred = model.predict(self.X_multi)
        self.assertEqual(len(y_pred), len(self.X_multi))

    def test_predict_proba_multiclass_shape(self) -> None:
        """predict_proba must return shape (n_samples, n_classes) for multiclass."""
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        model.fit(self.X_multi, self.y_multi)
        proba = model.predict_proba(self.X_multi)
        self.assertEqual(proba.shape, (len(self.X_multi), 3))

    def test_all_modes_multiclass(self) -> None:
        """Every combination mode must produce predictions of the correct length."""
        for mode in ("combine", "diff", "combine_and_diff"):
            with self.subTest(mode=mode):
                model = PairwiseDifferenceClassifier(
                    estimator=LogisticRegression(),
                    mode=mode,
                )
                model.fit(self.X_multi, self.y_multi)
                y_pred = model.predict(self.X_multi)
                self.assertEqual(len(y_pred), len(self.X_multi))

    # ------------------------------------------------------------------
    # Multilabel data (make_multilabel_classification)
    # ------------------------------------------------------------------

    def test_multilabel_first_column_as_target(self) -> None:
        """Single column of multilabel targets must fit and predict correctly."""
        y_single = self.Y_ml[:, 0]
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        model.fit(self.X_ml, y_single)
        y_pred = model.predict(self.X_ml)
        self.assertEqual(len(y_pred), len(self.X_ml))
