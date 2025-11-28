"""Unit tests for tanimoto based functions."""

import unittest

import numpy as np
import numpy.typing as npt
from scipy import sparse

from molpipeline.kernel.tanimoto_functions import (
    self_tanimoto_distance,
    self_tanimoto_similarity,
    tanimoto_distance_sparse,
    tanimoto_similarity_sparse,
)


class ABCTanimotoTestCase(unittest.TestCase):
    """Abstract base class for Tanimoto tests."""

    def setUp(self) -> None:
        """Set up common test data."""
        self.matrix_a_np: npt.NDArray[np.int_] = np.array(
            [[1, 0, 1, 0], [1, 1, 0, 0]],
            dtype=int,
        )
        self.matrix_b_np: npt.NDArray[np.int_] = np.array(
            [[1, 0, 1, 1], [0, 1, 0, 1]],
            dtype=int,
        )
        self.matrix_a_sparse: sparse.csr_matrix = sparse.csr_matrix(self.matrix_a_np)
        self.matrix_b_sparse: sparse.csr_matrix = sparse.csr_matrix(self.matrix_b_np)


class TestTanimotoSimilaritySparse(ABCTanimotoTestCase):
    """Tests for `tanimoto_similarity_sparse` with mixed dense/sparse inputs."""

    def test_tanimoto_similarity_numpy_inputs(self) -> None:
        """Validate correct similarity for dense NumPy inputs."""
        sim = tanimoto_similarity_sparse(self.matrix_a_np, self.matrix_b_np)
        self.assertEqual(sim.dtype, float, msg=f"Unexpected dtype: {sim.dtype}")
        expected = np.array([[2 / 3, 0.0], [0.25, 1 / 3]])
        self.assertTrue(np.allclose(sim, expected))

    def test_tanimoto_similarity_sparse_inputs(self) -> None:
        """Validate correct similarity for sparse CSR inputs."""
        sim = tanimoto_similarity_sparse(self.matrix_a_sparse, self.matrix_b_sparse)
        self.assertEqual(sim.dtype, float, msg=f"Unexpected dtype: {sim.dtype}")
        expected = np.array([[2 / 3, 0.0], [0.25, 1 / 3]])
        self.assertTrue(np.allclose(sim, expected))

    def test_tanimoto_similarity_same_object_path(self) -> None:
        """Ensure identity branch (matrix compared to itself) reuses norms."""
        sim = tanimoto_similarity_sparse(self.matrix_a_sparse, self.matrix_a_sparse)
        self.assertEqual(sim.dtype, float, msg=f"Unexpected dtype: {sim.dtype}")
        expected = np.array([[1.0, 1 / 3], [1 / 3, 1.0]])
        self.assertTrue(np.allclose(sim, expected))

    def test_zero_vector_handling_indirect(self) -> None:
        """Test behavior with zero rows (similarity should remain finite)."""
        mat = np.array([[0, 0, 0], [1, 0, 0]], dtype=int)
        sim = tanimoto_similarity_sparse(mat, mat)
        expected = np.array([[0.0, 0.0], [0.0, 1.0]])
        self.assertTrue(np.allclose(sim, expected))


class TestTanimotoDistanceSparse(ABCTanimotoTestCase):
    """Tests for `tanimoto_distance_sparse` correctness and range."""

    def test_tanimoto_distance_numpy(self) -> None:
        """Distance should equal (1 - similarity)."""
        sim = tanimoto_similarity_sparse(self.matrix_a_np, self.matrix_b_np)
        dist = tanimoto_distance_sparse(self.matrix_a_np, self.matrix_b_np)
        self.assertTrue(np.allclose(dist, 1 - sim))

    def test_tanimoto_distance_sparse(self) -> None:
        """Distance should equal (1 - similarity)."""
        sim = tanimoto_similarity_sparse(self.matrix_a_sparse, self.matrix_b_sparse)
        dist = tanimoto_distance_sparse(self.matrix_a_sparse, self.matrix_b_sparse)
        self.assertTrue(np.allclose(dist, 1 - sim))


class TestSelfTanimotoSimilarity(ABCTanimotoTestCase):
    """Tests for `self_tanimoto_similarity` including input validation."""

    def test_self_tanimoto_similarity_numpy(self) -> None:
        """Validate similarity for dense NumPy self-comparison."""
        sim = self_tanimoto_similarity(self.matrix_a_np)
        expected = np.array([[1.0, 1 / 3], [1 / 3, 1.0]])
        self.assertTrue(np.allclose(sim, expected))

    def test_self_tanimoto_similarity_sparse(self) -> None:
        """Validate similarity for sparse self-comparison."""
        sim = self_tanimoto_similarity(self.matrix_a_sparse)
        expected = np.array([[1.0, 1 / 3], [1 / 3, 1.0]])
        self.assertTrue(np.allclose(sim, expected))

    def test_zero_vector_handling(self) -> None:
        """Ensure zero vectors produce zeros."""
        mat = np.array([[0, 0, 0], [1, 0, 0]], dtype=int)
        sim = self_tanimoto_similarity(mat)
        expected = np.array([[0.0, 0.0], [0.0, 1.0]])
        self.assertTrue(np.allclose(sim, expected))

    def test_type_error_in_self_similarity(self) -> None:
        """Non-sparse, non-ndarray input should raise TypeError."""
        with self.assertRaises(TypeError):
            self_tanimoto_similarity([[1, 0], [0, 1]])  # type: ignore[arg-type]


class TestSelfTanimotoDistance(unittest.TestCase):
    """Tests for `self_tanimoto_distance` ensuring complementarity."""

    def setUp(self) -> None:
        """Create sparse matrix for distance self-comparison."""
        self.matrix_a_sparse: sparse.csr_matrix = sparse.csr_matrix(
            np.array([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=int),
        )

    def test_self_tanimoto_distance(self) -> None:
        """Distance should be (1 - self similarity)."""
        sim = self_tanimoto_similarity(self.matrix_a_sparse)
        dist = self_tanimoto_distance(self.matrix_a_sparse)
        self.assertTrue(np.allclose(dist, 1 - sim))
