"""Tests for pairwise-difference learner utility functions and estimators.

Covers dense/sparse combination helpers and the sklearn-compatible
PairwiseDifferenceRegressor and PairwiseDifferenceClassifier estimators.
"""

import unittest

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier

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

_SPARSE_CLASSES = [
    sp.csr_matrix,
    sp.coo_matrix,
    sp.bsr_matrix,
    sp.coo_matrix,
    sp.lil_matrix,
]

# ---------------------------------------------------------------------------
# Expected output constants - dual combinations (all row-pairs from V1 x V2)
# ---------------------------------------------------------------------------

_DUAL_COMBINE_EXPECTED: npt.NDArray[np.int_] = np.array(
    [
        [1, 2, 5, 6],
        [1, 2, 7, 8],
        [3, 4, 5, 6],
        [3, 4, 7, 8],
    ],
)

_DUAL_DIFF_EXPECTED: npt.NDArray[np.int_] = np.array(
    [
        [1 - 5, 2 - 6],
        [1 - 7, 2 - 8],
        [3 - 5, 4 - 6],
        [3 - 7, 4 - 8],
    ],
)

_DUAL_COMBINE_AND_DIFF_EXPECTED: npt.NDArray[np.int_] = np.array(
    [
        [1, 2, 5, 6, 1 - 5, 2 - 6],
        [1, 2, 7, 8, 1 - 7, 2 - 8],
        [3, 4, 5, 6, 3 - 5, 4 - 6],
        [3, 4, 7, 8, 3 - 7, 4 - 8],
    ],
)

# ---------------------------------------------------------------------------
# Expected output constants - single combinations (unique pairs of V)
# ---------------------------------------------------------------------------

_SINGLE_COMBINE_EXPECTED: npt.NDArray[np.int_] = np.array(
    [
        [1, 2, 3, 4],
        [1, 2, 5, 6],
        [3, 4, 5, 6],
    ],
)

_SINGLE_DIFF_EXPECTED: npt.NDArray[np.int_] = np.array(
    [
        [1 - 3, 2 - 4],
        [1 - 5, 2 - 6],
        [3 - 5, 4 - 6],
    ],
)

_SINGLE_COMBINE_AND_DIFF_EXPECTED: npt.NDArray[np.int_] = np.array(
    [
        [1, 2, 3, 4, 1 - 3, 2 - 4],
        [1, 2, 5, 6, 1 - 5, 2 - 6],
        [3, 4, 5, 6, 3 - 5, 4 - 6],
    ],
)


# ===========================================================================
# Dense helper: dual_vector_combinations_dense
# ===========================================================================


class TestDualVectorCombinationsDense(unittest.TestCase):
    """Tests for the dense cross-pairing helper dual_vector_combinations_dense."""

    def test_combine_mode(self) -> None:
        """Test combine returns ndarray with correct shape and values."""
        result = dual_vector_combinations_dense(_V1, _V2, mode="combine")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 4))
        self.assertTrue(np.array_equal(result, _DUAL_COMBINE_EXPECTED))

    def test_diff_mode(self) -> None:
        """Test diff returns ndarray with correct shape and values."""
        result = dual_vector_combinations_dense(_V1, _V2, mode="diff")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 2))
        self.assertTrue(np.array_equal(result, _DUAL_DIFF_EXPECTED))

    def test_combine_and_diff_mode(self) -> None:
        """Test combine_and_diff returns ndarray with correct shape and values."""
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
            ),
        )


# ===========================================================================
# Sparse helper: dual_vector_combinations_sparse
# ===========================================================================


class TestDualVectorCombinationsSparse(unittest.TestCase):
    """Tests for the sparse cross-pairing helper dual_vector_combinations_sparse."""

    def test_combine_mode(self) -> None:
        """Test combine returns sparse result with correct shape matching dense."""
        result = dual_vector_combinations_sparse(_V1_SPARSE, _V2_SPARSE, mode="combine")
        self.assertTrue(sp.issparse(result))
        self.assertEqual(result.shape, (4, 4))
        self.assertTrue(np.array_equal(result.toarray(), _DUAL_COMBINE_EXPECTED))

    def test_diff_mode(self) -> None:
        """Diff returns a sparse result with correct shape and values matching dense."""
        result = dual_vector_combinations_sparse(_V1_SPARSE, _V2_SPARSE, mode="diff")
        self.assertTrue(sp.issparse(result))
        self.assertEqual(result.shape, (4, 2))
        self.assertTrue(np.array_equal(result.toarray(), _DUAL_DIFF_EXPECTED))

    def test_combine_and_diff_mode(self) -> None:
        """combine_and_diff returns sparse result with correct shape matching dense."""
        result = dual_vector_combinations_sparse(
            _V1_SPARSE,
            _V2_SPARSE,
            mode="combine_and_diff",
        )
        self.assertTrue(sp.issparse(result))
        self.assertEqual(result.shape, (4, 6))
        self.assertTrue(
            np.array_equal(result.toarray(), _DUAL_COMBINE_AND_DIFF_EXPECTED),
        )

    def test_invalid_mode_raises(self) -> None:
        """An unsupported mode string must raise ValueError."""
        with self.assertRaises(ValueError):
            dual_vector_combinations_sparse(
                _V1_SPARSE,
                _V2_SPARSE,
                mode="invalid",  # type: ignore[arg-type]
            )

    def test_default_mode_is_combine(self) -> None:
        """Calling without explicit mode must behave identically to mode='combine'."""
        result_default = dual_vector_combinations_sparse(_V1_SPARSE, _V2_SPARSE)
        result_explicit = dual_vector_combinations_sparse(
            _V1_SPARSE,
            _V2_SPARSE,
            mode="combine",
        )
        self.assertTrue(
            np.array_equal(result_default.toarray(), result_explicit.toarray()),
        )


class TestDualVectorCombinations(unittest.TestCase):
    """Unit tests for the type-dispatching dual_vector_combinations function."""

    def test_combine_mode(self) -> None:
        """Verify combine mode dispatches correctly for dense and sparse inputs.

        Dense input returns an ndarray; sparse input returns a sparse matrix.
        Values match the expected output for both backends.
        """
        dense = dual_vector_combinations(_V1, _V2, mode="combine")
        self.assertIsInstance(dense, np.ndarray)
        self.assertEqual(dense.shape, (4, 4))
        self.assertTrue(np.array_equal(dense, _DUAL_COMBINE_EXPECTED))

        sparse = dual_vector_combinations(_V1_SPARSE, _V2_SPARSE, mode="combine")
        self.assertTrue(sp.issparse(sparse))
        self.assertEqual(sparse.shape, (4, 4))
        self.assertTrue(np.array_equal(sparse.toarray(), _DUAL_COMBINE_EXPECTED))

    def test_diff_mode(self) -> None:
        """Verify diff mode dispatches correctly for dense and sparse inputs.

        Dense input returns an ndarray; sparse input returns a sparse matrix.
        Values match the expected output for both backends.
        """
        dense = dual_vector_combinations(_V1, _V2, mode="diff")
        self.assertIsInstance(dense, np.ndarray)
        self.assertEqual(dense.shape, (4, 2))
        self.assertTrue(np.array_equal(dense, _DUAL_DIFF_EXPECTED))

        sparse = dual_vector_combinations(_V1_SPARSE, _V2_SPARSE, mode="diff")
        self.assertTrue(sp.issparse(sparse))
        self.assertEqual(sparse.shape, (4, 2))
        self.assertTrue(np.array_equal(sparse.toarray(), _DUAL_DIFF_EXPECTED))

    def test_combine_and_diff_mode(self) -> None:
        """Verify that combine_and_diff dispatches to the correct backend.

        Dense input returns an ndarray; sparse input returns a sparse matrix.
        Values match the expected output for both backends.
        """
        dense = dual_vector_combinations(_V1, _V2, mode="combine_and_diff")
        self.assertIsInstance(dense, np.ndarray)
        self.assertEqual(dense.shape, (4, 6))
        self.assertTrue(np.array_equal(dense, _DUAL_COMBINE_AND_DIFF_EXPECTED))

        sparse = dual_vector_combinations(
            _V1_SPARSE,
            _V2_SPARSE,
            mode="combine_and_diff",
        )
        self.assertTrue(sp.issparse(sparse))
        self.assertEqual(sparse.shape, (4, 6))
        self.assertTrue(
            np.array_equal(sparse.toarray(), _DUAL_COMBINE_AND_DIFF_EXPECTED),
        )

    def test_mixed_sparse_dispatch_returns_sparse(self) -> None:
        """If either input is sparse, result must be dense."""
        result = dual_vector_combinations(_V1_SPARSE, _V2)
        self.assertIsInstance(result, np.ndarray)

    def test_invalid_mode_raises(self) -> None:
        """An unsupported mode string must raise ValueError."""
        with self.assertRaises(ValueError):
            dual_vector_combinations(_V1, _V2, mode="invalid")  # type: ignore

    def test_default_mode_is_combine(self) -> None:
        """Calling without explicit mode must behave identically to mode='combine'."""
        self.assertTrue(
            np.array_equal(
                dual_vector_combinations(_V1, _V2),
                dual_vector_combinations(_V1, _V2, mode="combine"),
            ),
        )


# ===========================================================================
# Dense helper: single_vector_combinations_dense
# ===========================================================================


class TestSingleVectorCombinationsDense(unittest.TestCase):
    """Unit tests for the dense unique-pairs helper single_vector_combinations_dense."""

    def test_combine_mode(self) -> None:
        """Test combine returns ndarray with correct shape and values."""
        result = single_vector_combinations_dense(_V, mode="combine")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 4))
        self.assertTrue(np.array_equal(result, _SINGLE_COMBINE_EXPECTED))

    def test_diff_mode(self) -> None:
        """Test diff returns ndarray with correct shape and values."""
        result = single_vector_combinations_dense(_V, mode="diff")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(np.array_equal(result, _SINGLE_DIFF_EXPECTED))

    def test_combine_and_diff_mode(self) -> None:
        """combine_and_diff returns an ndarray with correct shape and values."""
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
            ),
        )


# ===========================================================================
# Sparse helper: single_vector_combinations_sparse
# ===========================================================================


class TestSingleVectorCombinationsSparse(unittest.TestCase):
    """Tests for the sparse unique-pairs helper single_vector_combinations_sparse."""

    def test_combine_mode(self) -> None:
        """Test combine returns sparse result with correct shape matching dense."""
        n = _V_SPARSE.shape[0]
        result = single_vector_combinations_sparse(_V_SPARSE, mode="combine")
        self.assertTrue(sp.issparse(result))
        self.assertEqual(result.shape, (n * (n - 1) // 2, 4))
        self.assertTrue(np.array_equal(result.toarray(), _SINGLE_COMBINE_EXPECTED))

    def test_diff_mode(self) -> None:
        """Test diff returns sparse result with correct shape matching dense."""
        n = _V_SPARSE.shape[0]
        result = single_vector_combinations_sparse(_V_SPARSE, mode="diff")
        self.assertTrue(sp.issparse(result))
        self.assertEqual(result.shape, (n * (n - 1) // 2, 2))
        self.assertTrue(np.array_equal(result.toarray(), _SINGLE_DIFF_EXPECTED))

    def test_combine_and_diff_mode(self) -> None:
        """combine_and_diff returns sparse result with correct shape matching dense."""
        n = _V_SPARSE.shape[0]
        result = single_vector_combinations_sparse(_V_SPARSE, mode="combine_and_diff")
        self.assertTrue(sp.issparse(result))
        self.assertEqual(result.shape, (n * (n - 1) // 2, 6))
        self.assertTrue(
            np.array_equal(result.toarray(), _SINGLE_COMBINE_AND_DIFF_EXPECTED),
        )

    def test_no_self_pairs(self) -> None:
        """Result must contain fewer rows than dual_vector_combinations_sparse(V, V)."""
        single_result = single_vector_combinations_sparse(_V_SPARSE, mode="combine")
        dual_result = dual_vector_combinations_sparse(
            _V_SPARSE,
            _V_SPARSE,
            mode="combine",
        )
        self.assertGreater(dual_result.shape[0], single_result.shape[0])

    def test_invalid_mode_raises(self) -> None:
        """An unsupported mode string must raise ValueError."""
        with self.assertRaises(ValueError):
            single_vector_combinations_sparse(
                _V_SPARSE,
                mode="invalid",  # type: ignore[arg-type]
            )

    def test_default_mode_is_combine(self) -> None:
        """Calling without explicit mode must behave identically to mode='combine'."""
        result_default = single_vector_combinations_sparse(_V_SPARSE)
        result_explicit = single_vector_combinations_sparse(_V_SPARSE, mode="combine")
        self.assertTrue(
            np.array_equal(result_default.toarray(), result_explicit.toarray()),
        )


class TestSingleVectorCombinations(unittest.TestCase):
    """Tests for the type-dispatching single_vector_combinations."""

    def test_combine_mode(self) -> None:
        """Verify combine mode dispatches correctly for dense and sparse inputs.

        Dense input returns an ndarray; sparse input returns a sparse matrix.
        Values match the expected output for both backends.
        """
        n = _V.shape[0]
        expected_rows = n * (n - 1) // 2

        dense = single_vector_combinations(_V, mode="combine")
        self.assertIsInstance(dense, np.ndarray)
        self.assertEqual(dense.shape, (expected_rows, 4))
        self.assertTrue(np.array_equal(dense, _SINGLE_COMBINE_EXPECTED))

        for sparse_cls in _SPARSE_CLASSES:
            with self.subTest(sparse_cls=sparse_cls.__name__):
                v = sparse_cls(_V)
                sparse = single_vector_combinations(v, mode="combine")
                self.assertTrue(sp.issparse(sparse))
                self.assertEqual(sparse.shape, (expected_rows, 4))
                self.assertTrue(
                    np.array_equal(sparse.toarray(), _SINGLE_COMBINE_EXPECTED),
                )

    def test_diff_mode(self) -> None:
        """Verify diff mode dispatches correctly for dense and sparse inputs.

        Dense input returns an ndarray; sparse input returns a sparse matrix.
        Values match the expected output for both backends.
        """
        n = _V.shape[0]
        expected_rows = n * (n - 1) // 2

        dense = single_vector_combinations(_V, mode="diff")
        self.assertIsInstance(dense, np.ndarray)
        self.assertEqual(dense.shape, (expected_rows, 2))
        self.assertTrue(np.array_equal(dense, _SINGLE_DIFF_EXPECTED))

        for sparse_cls in _SPARSE_CLASSES:
            with self.subTest(sparse_cls=sparse_cls.__name__):
                v = sparse_cls(_V)
                sparse = single_vector_combinations(v, mode="diff")
                self.assertTrue(sp.issparse(sparse))
                self.assertEqual(sparse.shape, (expected_rows, 2))
                self.assertTrue(np.array_equal(sparse.toarray(), _SINGLE_DIFF_EXPECTED))

    def test_combine_and_diff_mode(self) -> None:
        """Verify that combine_and_diff dispatches to the correct backend.

        Dense input returns an ndarray; sparse input returns a sparse matrix.
        Values match the expected output for both backends.
        """
        n = _V.shape[0]
        expected_rows = n * (n - 1) // 2

        dense = single_vector_combinations(_V, mode="combine_and_diff")
        self.assertIsInstance(dense, np.ndarray)
        self.assertEqual(dense.shape, (expected_rows, 6))
        self.assertTrue(np.array_equal(dense, _SINGLE_COMBINE_AND_DIFF_EXPECTED))

        for sparse_cls in _SPARSE_CLASSES:
            with self.subTest(sparse_cls=sparse_cls.__name__):
                v = sparse_cls(_V)
                sparse = single_vector_combinations(v, mode="combine_and_diff")
                self.assertTrue(sp.issparse(sparse))
                self.assertEqual(sparse.shape, (expected_rows, 6))
                self.assertTrue(
                    np.array_equal(sparse.toarray(), _SINGLE_COMBINE_AND_DIFF_EXPECTED),
                )

    def test_no_self_pairs(self) -> None:
        """Result must contain fewer rows than dual_vector_combinations(V, V).

        Applies for both backends.
        """
        self.assertGreater(
            dual_vector_combinations(_V, _V, mode="combine").shape[0],
            single_vector_combinations(_V, mode="combine").shape[0],
        )
        for sparse_cls in _SPARSE_CLASSES:
            with self.subTest(sparse_cls=sparse_cls.__name__):
                v = sparse_cls(_V)
                self.assertGreater(
                    dual_vector_combinations(v, v, mode="combine").shape[0],
                    single_vector_combinations(v, mode="combine").shape[0],
                )

    def test_invalid_mode_raises(self) -> None:
        """An unsupported mode string must raise ValueError."""
        with self.assertRaises(ValueError):
            single_vector_combinations(_V, mode="invalid")  # type: ignore

    def test_default_mode_is_combine(self) -> None:
        """Calling without explicit mode must behave identically to mode='combine'."""
        self.assertTrue(
            np.array_equal(
                single_vector_combinations(_V),
                single_vector_combinations(_V, mode="combine"),
            ),
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
        """Predict must return an ndarray of length n_samples."""
        model = PairwiseDifferenceRegressor(estimator=LinearRegression())
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_pred.shape[0], len(self.X))

    def test_predict_return_std(self) -> None:
        """return_std=True returns (mean, std) of length n_samples with std >= 0."""
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
                    estimator=LinearRegression(),
                    mode=mode,
                )
                model.fit(self.X, self.y)
                y_pred = model.predict(self.X)
                self.assertIsInstance(y_pred, np.ndarray)
                self.assertEqual(y_pred.shape[0], len(self.X))

    def test_fit_returns_self(self) -> None:
        """Fit must return the estimator itself (sklearn convention)."""
        model = PairwiseDifferenceRegressor(estimator=LinearRegression())
        self.assertIs(model.fit(self.X, self.y), model)

    def test_numeric_correctness_linear(self) -> None:  # noqa: PLR6301
        """On a noiseless linear dataset the regressor should recover y closely."""
        rng = np.random.default_rng(42)
        x = rng.random((15, 3))
        y = x @ np.array([1.0, -2.0, 0.5])  # exact linear relationship, no noise
        model = PairwiseDifferenceRegressor(estimator=LinearRegression())
        model.fit(x, y)
        y_pred = model.predict(x)
        np.testing.assert_allclose(y_pred, y, atol=1e-6)

    def test_fit_twice_same_predictions(self) -> None:
        """Calling fit twice must produce identical results (no state leakage)."""
        model = PairwiseDifferenceRegressor(estimator=LinearRegression())
        model.fit(self.X, self.y)
        y_pred_first = model.predict(self.X)
        model.fit(self.X, self.y)
        y_pred_second = model.predict(self.X)
        np.testing.assert_array_equal(y_pred_first, y_pred_second)

    def test_clone_and_fit(self) -> None:
        """clone() followed by fit must work correctly (sklearn compat)."""
        original = PairwiseDifferenceRegressor(estimator=LinearRegression())
        cloned = clone(original)
        cloned.fit(self.X, self.y)
        y_pred = cloned.predict(self.X)
        self.assertEqual(y_pred.shape[0], len(self.X))


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
        """Predicted labels have correct length and are a subset of training classes."""
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
            proba.sum(axis=1),
            np.ones(len(self.X_bin)),
            atol=1e-6,
        )

    def test_fit_returns_self_binary(self) -> None:
        """Fit must return the estimator itself (sklearn convention)."""
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        self.assertIs(model.fit(self.X_bin, self.y_bin), model)

    # ------------------------------------------------------------------
    # Multiclass classification (make_classification, n_classes=3)
    # ------------------------------------------------------------------

    def test_multiclass_predict(self) -> None:
        """Predicted labels have correct length and are a subset of training classes."""
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
            proba.sum(axis=1),
            np.ones(len(self.X_multi)),
            atol=1e-6,
        )

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
        """Multilabel single-column target.

        Fitted and predicted within training classes.
        """
        y_single = self.Y_ml[:, 0]
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        model.fit(self.X_ml, y_single)
        y_pred = model.predict(self.X_ml)
        self.assertEqual(len(y_pred), len(self.X_ml))
        self.assertTrue(set(y_pred).issubset(set(y_single)))

    # ------------------------------------------------------------------
    # Sklearn compatibility
    # ------------------------------------------------------------------

    def test_fit_twice_same_predictions(self) -> None:
        """Calling fit twice must produce identical predictions (no state leakage).

        This guards against the estimators_ list growing across repeated fits.
        """
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        model.fit(self.X_bin, self.y_bin)
        y_pred_first = model.predict(self.X_bin)
        model.fit(self.X_bin, self.y_bin)
        y_pred_second = model.predict(self.X_bin)
        np.testing.assert_array_equal(y_pred_first, y_pred_second)

    def test_clone_and_fit(self) -> None:
        """clone() followed by fit must work correctly."""
        original = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        cloned = clone(original)
        cloned.fit(self.X_bin, self.y_bin)
        y_pred = cloned.predict(self.X_bin)
        self.assertEqual(len(y_pred), len(self.X_bin))

    # ------------------------------------------------------------------
    # Fallback branch: estimator without predict_proba
    # ------------------------------------------------------------------

    def test_predict_proba_fallback_no_predict_proba(self) -> None:
        """Classifier must work when the underlying estimator has no predict_proba.

        Uses RidgeClassifier (no predict_proba) to exercise the fallback branch
        and confirms that predict_proba rows still sum to 1.
        """
        model = PairwiseDifferenceClassifier(estimator=RidgeClassifier())
        model.fit(self.X_bin, self.y_bin)
        proba = model.predict_proba(self.X_bin)
        self.assertEqual(proba.shape, (len(self.X_bin), 2))
        np.testing.assert_allclose(
            proba.sum(axis=1),
            np.ones(len(self.X_bin)),
            atol=1e-6,
        )

    def test_multiclass_fallback_no_predict_proba(self) -> None:
        """Fallback branch must also work for multiclass data (no predict_proba).

        Uses RidgeClassifier (no predict_proba) with a 3-class dataset and
        confirms output shape is (n_samples, 3) with rows summing to 1.
        """
        model = PairwiseDifferenceClassifier(estimator=RidgeClassifier())
        model.fit(self.X_multi, self.y_multi)
        proba = model.predict_proba(self.X_multi)
        self.assertEqual(proba.shape, (len(self.X_multi), 3))
        np.testing.assert_allclose(
            proba.sum(axis=1),
            np.ones(len(self.X_multi)),
            atol=1e-6,
        )

    def test_multiclass_fallback_predict(self) -> None:
        """Predict must return correct-length output with valid class labels.

        Uses RidgeClassifier (no predict_proba) with a 3-class dataset.
        """
        model = PairwiseDifferenceClassifier(estimator=RidgeClassifier())
        model.fit(self.X_multi, self.y_multi)
        y_pred = model.predict(self.X_multi)
        self.assertEqual(len(y_pred), len(self.X_multi))
        self.assertTrue(set(y_pred).issubset(set(self.y_multi)))

    def test_predict_proba_normalised_multiclass(self) -> None:
        """predict_proba rows must always sum to 1, even for multiclass.

        Uses an imbalanced multiclass scenario where the independent per-class
        scores could sum to more than 1 without normalisation.
        """
        model = PairwiseDifferenceClassifier(estimator=LogisticRegression())
        model.fit(self.X_multi, self.y_multi)
        proba = model.predict_proba(self.X_multi)
        np.testing.assert_allclose(
            proba.sum(axis=1),
            np.ones(len(self.X_multi)),
            atol=1e-6,
        )
