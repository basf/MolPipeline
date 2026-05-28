"""Tests for pairwise difference learner utility functions."""

import unittest

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
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
    dual_vector_combinations_dense,
    dual_vector_combinations_sparse,
    single_vector_combinations,
    single_vector_combinations_dense,
    single_vector_combinations_sparse,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Two 2-row, 2-feature matrices used for dual combinations.
_V1: npt.NDArray[np.int_] = np.array([[1, 2], [3, 4]])
_V2: npt.NDArray[np.int_] = np.array([[5, 6], [7, 8]])

# A 3-row, 2-feature matrix used for single combinations.
_V: npt.NDArray[np.int_] = np.array([[1, 2], [3, 4], [5, 6]])

# ---------------------------------------------------------------------------
# Sparse fixtures (mirrors of the dense fixtures)
# ---------------------------------------------------------------------------

_V1_SPARSE: sp.csr_matrix = sp.csr_matrix(_V1)
_V2_SPARSE: sp.csr_matrix = sp.csr_matrix(_V2)
_V_SPARSE: sp.csr_matrix = sp.csr_matrix(_V)

# ---------------------------------------------------------------------------
# Expected output constants – dual combinations (Cartesian product V1 x V2)
# ---------------------------------------------------------------------------

_DUAL_COMBINE_EXPECTED: npt.NDArray[np.int_] = np.array(
    [
        [1, 2, 5, 6],
        [1, 2, 7, 8],
        [3, 4, 5, 6],
        [3, 4, 7, 8],
    ]
)

_DUAL_DIFF_EXPECTED: npt.NDArray[np.int_] = np.array(
    [
        [1 - 5, 2 - 6],
        [1 - 7, 2 - 8],
        [3 - 5, 4 - 6],
        [3 - 7, 4 - 8],
    ]
)

_DUAL_COMBINE_AND_DIFF_EXPECTED: npt.NDArray[np.int_] = np.array(
    [
        [1, 2, 5, 6, 1 - 5, 2 - 6],
        [1, 2, 7, 8, 1 - 7, 2 - 8],
        [3, 4, 5, 6, 3 - 5, 4 - 6],
        [3, 4, 7, 8, 3 - 7, 4 - 8],
    ]
)

# ---------------------------------------------------------------------------
# Expected output constants – single combinations (unique pairs of V)
# ---------------------------------------------------------------------------

_SINGLE_COMBINE_EXPECTED: npt.NDArray[np.int_] = np.array(
    [
        [1, 2, 3, 4],
        [1, 2, 5, 6],
        [3, 4, 5, 6],
    ]
)

_SINGLE_DIFF_EXPECTED: npt.NDArray[np.int_] = np.array(
    [
        [1 - 3, 2 - 4],
        [1 - 5, 2 - 6],
        [3 - 5, 4 - 6],
    ]
)

_SINGLE_COMBINE_AND_DIFF_EXPECTED: npt.NDArray[np.int_] = np.array(
    [
        [1, 2, 3, 4, 1 - 3, 2 - 4],
        [1, 2, 5, 6, 1 - 5, 2 - 6],
        [3, 4, 5, 6, 3 - 5, 4 - 6],
    ]
)


# ===========================================================================
# Dense helper: dual_vector_combinations_dense
# ===========================================================================


class TestDualVectorCombinationsDense(unittest.TestCase):
    """Tests for dual_vector_combinations_dense."""

    def test_combine_mode(self) -> None:
        """combine: result is ndarray with shape (n1*n2, 2*f) and correct values."""
        result = dual_vector_combinations_dense(_V1, _V2, mode="combine")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 4))
        self.assertTrue(np.array_equal(result, _DUAL_COMBINE_EXPECTED))

    def test_diff_mode(self) -> None:
        """diff: result is ndarray with shape (n1*n2, f) and correct values."""
        result = dual_vector_combinations_dense(_V1, _V2, mode="diff")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 2))
        self.assertTrue(np.array_equal(result, _DUAL_DIFF_EXPECTED))

    def test_combine_and_diff_mode(self) -> None:
        """combine_and_diff: result is ndarray with shape (n1*n2, 3*f) and correct values."""
        result = dual_vector_combinations_dense(_V1, _V2, mode="combine_and_diff")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 6))
        self.assertTrue(np.array_equal(result, _DUAL_COMBINE_AND_DIFF_EXPECTED))

    def test_invalid_mode_raises(self) -> None:
        """An unsupported mode string must raise ValueError."""
        with self.assertRaises(ValueError):
            dual_vector_combinations_dense(_V1, _V2, mode="invalid")  # type: ignore

    def test_default_mode_is_combine(self) -> None:
        """Calling without explicit mode must behave identically to mode='combine'."""
        self.assertTrue(
            np.array_equal(
                dual_vector_combinations_dense(_V1, _V2),
                dual_vector_combinations_dense(_V1, _V2, mode="combine"),
            )
        )


# ===========================================================================
# Sparse helper: dual_vector_combinations_sparse
# ===========================================================================


class TestDualVectorCombinationsSparse(unittest.TestCase):
    """Tests for dual_vector_combinations_sparse."""

    def test_combine_mode(self) -> None:
        """combine: sparse result with shape (n1*n2, 2*f), values match dense."""
        result = dual_vector_combinations_sparse(_V1_SPARSE, _V2_SPARSE, mode="combine")
        self.assertTrue(sp.issparse(result))
        self.assertEqual(result.shape, (4, 4))
        self.assertTrue(np.array_equal(result.toarray(), _DUAL_COMBINE_EXPECTED))

    def test_diff_mode(self) -> None:
        """diff: sparse result with shape (n1*n2, f), values match dense."""
        result = dual_vector_combinations_sparse(_V1_SPARSE, _V2_SPARSE, mode="diff")
        self.assertTrue(sp.issparse(result))
        self.assertEqual(result.shape, (4, 2))
        self.assertTrue(np.array_equal(result.toarray(), _DUAL_DIFF_EXPECTED))

    def test_combine_and_diff_mode(self) -> None:
        """combine_and_diff: sparse result with shape (n1*n2, 3*f), values match dense."""
        result = dual_vector_combinations_sparse(
            _V1_SPARSE, _V2_SPARSE, mode="combine_and_diff"
        )
        self.assertTrue(sp.issparse(result))
        self.assertEqual(result.shape, (4, 6))
        self.assertTrue(np.array_equal(result.toarray(), _DUAL_COMBINE_AND_DIFF_EXPECTED))

    def test_invalid_mode_raises(self) -> None:
        """An unsupported mode string must raise ValueError."""
        with self.assertRaises(ValueError):
            dual_vector_combinations_sparse(
                _V1_SPARSE, _V2_SPARSE, mode="invalid"  # type: ignore[arg-type]
            )

    def test_default_mode_is_combine(self) -> None:
        """Calling without explicit mode must behave identically to mode='combine'."""
        result_default = dual_vector_combinations_sparse(_V1_SPARSE, _V2_SPARSE)
        result_explicit = dual_vector_combinations_sparse(
            _V1_SPARSE, _V2_SPARSE, mode="combine"
        )
        self.assertTrue(
            np.array_equal(result_default.toarray(), result_explicit.toarray())
        )


# ===========================================================================
# Dispatcher: dual_vector_combinations
# ===========================================================================


class TestDualVectorCombinations(unittest.TestCase):
    """Tests for the type-dispatching dual_vector_combinations."""

    def test_combine_mode(self) -> None:
        """Test combine mode for dense and sparse."""
        dense = dual_vector_combinations(_V1, _V2, mode="combine")
        self.assertIsInstance(dense, np.ndarray)
        self.assertEqual(dense.shape, (4, 4))
        self.assertTrue(np.array_equal(dense, _DUAL_COMBINE_EXPECTED))

        sparse = dual_vector_combinations(_V1_SPARSE, _V2_SPARSE, mode="combine")
        self.assertTrue(sp.issparse(sparse))
        self.assertEqual(sparse.shape, (4, 4))
        self.assertTrue(np.array_equal(sparse.toarray(), _DUAL_COMBINE_EXPECTED))

    def test_diff_mode(self) -> None:
        """Test diff mode for dense and sparse."""
        dense = dual_vector_combinations(_V1, _V2, mode="diff")
        self.assertIsInstance(dense, np.ndarray)
        self.assertEqual(dense.shape, (4, 2))
        self.assertTrue(np.array_equal(dense, _DUAL_DIFF_EXPECTED))

        sparse = dual_vector_combinations(_V1_SPARSE, _V2_SPARSE, mode="diff")
        self.assertTrue(sp.issparse(sparse))
        self.assertEqual(sparse.shape, (4, 2))
        self.assertTrue(np.array_equal(sparse.toarray(), _DUAL_DIFF_EXPECTED))

    def test_combine_and_diff_mode(self) -> None:
        """Test combine_and_diff mode for dense and sparse."""
        dense = dual_vector_combinations(_V1, _V2, mode="combine_and_diff")
        self.assertIsInstance(dense, np.ndarray)
        self.assertEqual(dense.shape, (4, 6))
        self.assertTrue(np.array_equal(dense, _DUAL_COMBINE_AND_DIFF_EXPECTED))

        sparse = dual_vector_combinations(_V1_SPARSE, _V2_SPARSE, mode="combine_and_diff")
        self.assertTrue(sp.issparse(sparse))
        self.assertEqual(sparse.shape, (4, 6))
        self.assertTrue(np.array_equal(sparse.toarray(), _DUAL_COMBINE_AND_DIFF_EXPECTED))

    def test_mixed_sparse_dispatch_returns_sparse(self) -> None:
        """If either input is sparse, result must be sparse."""
        result = dual_vector_combinations(_V1_SPARSE, _V2)
        self.assertTrue(sp.issparse(result))

    def test_invalid_mode_raises(self) -> None:
        """An unsupported mode string must raise ValueError."""
        with self.assertRaises(ValueError):
            dual_vector_combinations(_V1, _V2, mode="invalid")  # type: ignore[arg-type]

    def test_default_mode_is_combine(self) -> None:
        """Calling without explicit mode must behave identically to mode='combine'."""
        self.assertTrue(
            np.array_equal(
                dual_vector_combinations(_V1, _V2),
                dual_vector_combinations(_V1, _V2, mode="combine"),
            )
        )


# ===========================================================================
# Dense helper: single_vector_combinations_dense
# ===========================================================================


class TestSingleVectorCombinationsDense(unittest.TestCase):
    """Tests for single_vector_combinations_dense."""

    def test_combine_mode(self) -> None:
        """combine: result is ndarray with shape (C(n,2), 2*f) and correct values."""
        result = single_vector_combinations_dense(_V, mode="combine")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 4))
        self.assertTrue(np.array_equal(result, _SINGLE_COMBINE_EXPECTED))

    def test_diff_mode(self) -> None:
        """diff: result is ndarray with shape (C(n,2), f) and correct values."""
        result = single_vector_combinations_dense(_V, mode="diff")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(np.array_equal(result, _SINGLE_DIFF_EXPECTED))

    def test_combine_and_diff_mode(self) -> None:
        """combine_and_diff: result is ndarray with shape (C(n,2), 3*f) and correct values."""
        n = _V.shape[0]
        result = single_vector_combinations_dense(_V, mode="combine_and_diff")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 6))
        self.assertTrue(np.array_equal(result, _SINGLE_COMBINE_AND_DIFF_EXPECTED))

    def test_invalid_mode_raises(self) -> None:
        """An unsupported mode string must raise ValueError."""
        with self.assertRaises(ValueError):
            single_vector_combinations_dense(_V, mode="invalid")  # type: ignore

    def test_default_mode_is_combine(self) -> None:
        """Calling without explicit mode must behave identically to mode='combine'."""
        self.assertTrue(
            np.array_equal(
                single_vector_combinations_dense(_V),
                single_vector_combinations_dense(_V, mode="combine"),
            )
        )


# ===========================================================================
# Sparse helper: single_vector_combinations_sparse
# ===========================================================================


class TestSingleVectorCombinationsSparse(unittest.TestCase):
    """Tests for single_vector_combinations_sparse."""

    def test_combine_mode(self) -> None:
        """combine: sparse result with shape (C(n,2), 2*f), values match dense."""
        n = _V_SPARSE.shape[0]
        result = single_vector_combinations_sparse(_V_SPARSE, mode="combine")
        self.assertTrue(sp.issparse(result))
        self.assertEqual(result.shape, (n * (n - 1) // 2, 4))
        self.assertTrue(np.array_equal(result.toarray(), _SINGLE_COMBINE_EXPECTED))

    def test_diff_mode(self) -> None:
        """diff: sparse result with shape (C(n,2), f), values match dense."""
        n = _V_SPARSE.shape[0]
        result = single_vector_combinations_sparse(_V_SPARSE, mode="diff")
        self.assertTrue(sp.issparse(result))
        self.assertEqual(result.shape, (n * (n - 1) // 2, 2))
        self.assertTrue(np.array_equal(result.toarray(), _SINGLE_DIFF_EXPECTED))

    def test_combine_and_diff_mode(self) -> None:
        """combine_and_diff: sparse result with shape (C(n,2), 3*f), values match dense."""
        n = _V_SPARSE.shape[0]
        result = single_vector_combinations_sparse(_V_SPARSE, mode="combine_and_diff")
        self.assertTrue(sp.issparse(result))
        self.assertEqual(result.shape, (n * (n - 1) // 2, 6))
        self.assertTrue(
            np.array_equal(result.toarray(), _SINGLE_COMBINE_AND_DIFF_EXPECTED)
        )

    def test_no_self_pairs(self) -> None:
        """Result must contain fewer rows than dual_vector_combinations_sparse(V, V)."""
        single_result = single_vector_combinations_sparse(_V_SPARSE, mode="combine")
        dual_result = dual_vector_combinations_sparse(_V_SPARSE, _V_SPARSE, mode="combine")
        self.assertGreater(dual_result.shape[0], single_result.shape[0])

    def test_invalid_mode_raises(self) -> None:
        """An unsupported mode string must raise ValueError."""
        with self.assertRaises(ValueError):
            single_vector_combinations_sparse(
                _V_SPARSE, mode="invalid"  # type: ignore[arg-type]
            )

    def test_default_mode_is_combine(self) -> None:
        """Calling without explicit mode must behave identically to mode='combine'."""
        result_default = single_vector_combinations_sparse(_V_SPARSE)
        result_explicit = single_vector_combinations_sparse(_V_SPARSE, mode="combine")
        self.assertTrue(
            np.array_equal(result_default.toarray(), result_explicit.toarray())
        )

# ===========================================================================
# Dispatcher: single_vector_combinations
# ===========================================================================


class TestSingleVectorCombinations(unittest.TestCase):
    """Tests for the type-dispatching single_vector_combinations."""

    def test_combine_mode(self) -> None:
        """combine: dense→ndarray, sparse→sparse; shape (C(n,2), 2*f) and correct values."""
        n = _V.shape[0]
        expected_rows = n * (n - 1) // 2

        dense = single_vector_combinations(_V, mode="combine")
        self.assertIsInstance(dense, np.ndarray)
        self.assertEqual(dense.shape, (expected_rows, 4))
        self.assertTrue(np.array_equal(dense, _SINGLE_COMBINE_EXPECTED))

        sparse = single_vector_combinations(_V_SPARSE, mode="combine")
        self.assertTrue(sp.issparse(sparse))
        self.assertEqual(sparse.shape, (expected_rows, 4))
        self.assertTrue(np.array_equal(sparse.toarray(), _SINGLE_COMBINE_EXPECTED))

    def test_diff_mode(self) -> None:
        """diff: dense→ndarray, sparse→sparse; shape (C(n,2), f) and correct values."""
        n = _V.shape[0]
        expected_rows = n * (n - 1) // 2

        dense = single_vector_combinations(_V, mode="diff")
        self.assertIsInstance(dense, np.ndarray)
        self.assertEqual(dense.shape, (expected_rows, 2))
        self.assertTrue(np.array_equal(dense, _SINGLE_DIFF_EXPECTED))

        sparse = single_vector_combinations(_V_SPARSE, mode="diff")
        self.assertTrue(sp.issparse(sparse))
        self.assertEqual(sparse.shape, (expected_rows, 2))
        self.assertTrue(np.array_equal(sparse.toarray(), _SINGLE_DIFF_EXPECTED))

    def test_combine_and_diff_mode(self) -> None:
        """combine_and_diff: dense→ndarray, sparse→sparse; shape (C(n,2), 3*f) and correct values."""
        n = _V.shape[0]
        expected_rows = n * (n - 1) // 2

        dense = single_vector_combinations(_V, mode="combine_and_diff")
        self.assertIsInstance(dense, np.ndarray)
        self.assertEqual(dense.shape, (expected_rows, 6))
        self.assertTrue(np.array_equal(dense, _SINGLE_COMBINE_AND_DIFF_EXPECTED))

        sparse = single_vector_combinations(_V_SPARSE, mode="combine_and_diff")
        self.assertTrue(sp.issparse(sparse))
        self.assertEqual(sparse.shape, (expected_rows, 6))
        self.assertTrue(
            np.array_equal(sparse.toarray(), _SINGLE_COMBINE_AND_DIFF_EXPECTED)
        )

    def test_no_self_pairs(self) -> None:
        """Result must contain fewer rows than dual_vector_combinations(V, V) for both backends."""
        self.assertGreater(
            dual_vector_combinations(_V, _V, mode="combine").shape[0],
            single_vector_combinations(_V, mode="combine").shape[0],
        )
        self.assertGreater(
            dual_vector_combinations(_V_SPARSE, _V_SPARSE, mode="combine").shape[0],
            single_vector_combinations(_V_SPARSE, mode="combine").shape[0],
        )

    def test_invalid_mode_raises(self) -> None:
        """An unsupported mode string must raise ValueError."""
        with self.assertRaises(ValueError):
            single_vector_combinations(_V, mode="invalid")  # type: ignore[arg-type]

    def test_default_mode_is_combine(self) -> None:
        """Calling without explicit mode must behave identically to mode='combine'."""
        self.assertTrue(
            np.array_equal(
                single_vector_combinations(_V),
                single_vector_combinations(_V, mode="combine"),
            )
        )


# ===========================================================================
# PairwiseDifferenceRegressor
# ===========================================================================


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

    def test_predict(self) -> None:
        """predict must return an ndarray of length n_samples."""
        model = PairwiseDifferenceRegressor(estimator=LinearRegression())
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_pred.shape[0], len(self.X))

    def test_predict_return_std(self) -> None:
        """return_std=True must return (mean, std) arrays of length n_samples, std >= 0."""
        model = PairwiseDifferenceRegressor(estimator=LinearRegression())
        model.fit(self.X, self.y)
        mean_pred, std_pred = model.predict(self.X, return_std=True)
        self.assertEqual(mean_pred.shape[0], len(self.X))
        self.assertEqual(std_pred.shape[0], len(self.X))
        self.assertTrue(bool(np.all(std_pred >= 0)))

    def test_all_modes(self) -> None:
        """Every combination mode must produce predictions of the correct shape."""
        for mode in ("combine", "diff", "combine_and_diff"):
            with self.subTest(mode=mode):
                model = PairwiseDifferenceRegressor(
                    estimator=LinearRegression(), mode=mode
                )
                model.fit(self.X, self.y)
                y_pred = model.predict(self.X)
                self.assertIsInstance(y_pred, np.ndarray)
                self.assertEqual(y_pred.shape[0], len(self.X))

    def test_fit_returns_self(self) -> None:
        """fit must return the estimator itself (sklearn convention)."""
        model = PairwiseDifferenceRegressor(estimator=LinearRegression())
        self.assertIs(model.fit(self.X, self.y), model)


# ===========================================================================
# PairwiseDifferenceClassifier
# ===========================================================================


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

    def test_binary_predict(self) -> None:
        """predict must return labels of correct length that are a subset of training classes."""
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        model.fit(self.X_bin, self.y_bin)
        y_pred = model.predict(self.X_bin)
        self.assertEqual(len(y_pred), len(self.X_bin))
        self.assertTrue(set(y_pred).issubset(set(self.y_bin)))

    def test_binary_predict_proba(self) -> None:
        """predict_proba must return shape (n, 2) with rows summing to 1."""
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        model.fit(self.X_bin, self.y_bin)
        proba = model.predict_proba(self.X_bin)
        self.assertEqual(proba.shape, (len(self.X_bin), 2))
        np.testing.assert_allclose(
            proba.sum(axis=1), np.ones(len(self.X_bin)), atol=1e-6
        )

    def test_fit_returns_self_binary(self) -> None:
        """fit must return the estimator itself (sklearn convention)."""
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        self.assertIs(model.fit(self.X_bin, self.y_bin), model)

    # ------------------------------------------------------------------
    # Multiclass classification (make_classification, n_classes=3)
    # ------------------------------------------------------------------

    def test_multiclass_predict(self) -> None:
        """predict must return labels of correct length that are a subset of training classes."""
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        model.fit(self.X_multi, self.y_multi)
        y_pred = model.predict(self.X_multi)
        self.assertEqual(len(y_pred), len(self.X_multi))
        self.assertTrue(set(y_pred).issubset(set(self.y_multi)))

    def test_multiclass_predict_proba(self) -> None:
        """predict_proba must return shape (n, 3) with rows summing to 1."""
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        model.fit(self.X_multi, self.y_multi)
        proba = model.predict_proba(self.X_multi)
        self.assertEqual(proba.shape, (len(self.X_multi), 3))
        np.testing.assert_allclose(
            proba.sum(axis=1), np.ones(len(self.X_multi)), atol=1e-6
        )

    def test_all_modes_multiclass(self) -> None:
        """Every combination mode must produce predictions of the correct length."""
        for mode in ("combine", "diff", "combine_and_diff"):
            with self.subTest(mode=mode):
                model = PairwiseDifferenceClassifier(
                    estimator=LogisticRegression(), mode=mode
                )
                model.fit(self.X_multi, self.y_multi)
                y_pred = model.predict(self.X_multi)
                self.assertEqual(len(y_pred), len(self.X_multi))

    # ------------------------------------------------------------------
    # Multilabel data (make_multilabel_classification)
    # ------------------------------------------------------------------

    def test_multilabel_first_column_as_target(self) -> None:
        """Single column of multilabel targets must fit, predict correct length, and stay within training classes."""
        y_single = self.Y_ml[:, 0]
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        model.fit(self.X_ml, y_single)
        y_pred = model.predict(self.X_ml)
        self.assertEqual(len(y_pred), len(self.X_ml))
        self.assertTrue(set(y_pred).issubset(set(y_single)))

