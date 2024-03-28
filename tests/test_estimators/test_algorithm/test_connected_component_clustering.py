"""Test connected component clustering algorithm."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from molpipeline.estimators.algorithm.connected_component_clustering import (
    calc_chunk_size_from_memory_requirement,
    connected_components_iterative_algorithm,
)


class TestConnectedComponentClusteringAlgorithm(unittest.TestCase):
    """Test connected component clustering algorithm."""

    def test_connected_components_iterative(self) -> None:
        """Test the iterative connected component clustering algorithm."""

        # test data: all possible fingerprints of len 3 but duplicated
        fingerprint_matrix = csr_matrix(
            [
                [0, 0, 0],  # 0
                [1, 0, 0],  # 1
                [1, 1, 0],  # 2
                [1, 0, 1],  # 3
                [0, 1, 0],  # 4
                [0, 1, 1],  # 5
                [0, 0, 1],  # 6
                [1, 1, 1],  # 7
                [0, 0, 0],  # 8
                [1, 0, 0],  # 9
                [1, 1, 0],  # 10
                [1, 0, 1],  # 11
                [0, 1, 0],  # 12
                [0, 1, 1],  # 13
                [0, 0, 1],  # 14
                [1, 1, 1],  # 15
            ]
        )

        eps: float = 1e-10

        expected_clusterings: list[dict[str, Any]] = [
            {
                "threshold": 0.0,
                "expected_clustering": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "threshold": 0.0 + eps,
                "expected_clustering": [0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
            },
            {
                "threshold": 1 / 3 - eps,
                "expected_clustering": [0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
            },
            {
                "threshold": 1 / 3,
                "expected_clustering": [0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
            },
            {
                "threshold": 1 / 3 + eps,
                "expected_clustering": [0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
            },
            {
                "threshold": 0.5 - eps,
                "expected_clustering": [0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
            },
            {
                "threshold": 0.5,
                "expected_clustering": [0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
            },
            {
                "threshold": 0.5 + eps,
                "expected_clustering": [0, 1, 2, 2, 3, 2, 4, 2, 5, 1, 2, 2, 3, 2, 4, 2],
            },
            {
                "threshold": 2 / 3 - eps,
                "expected_clustering": [0, 1, 2, 2, 3, 2, 4, 2, 5, 1, 2, 2, 3, 2, 4, 2],
            },
            {
                "threshold": 2 / 3,
                "expected_clustering": [0, 1, 2, 2, 3, 2, 4, 2, 5, 1, 2, 2, 3, 2, 4, 2],
            },
            {
                "threshold": 2 / 3 + eps,
                "expected_clustering": [0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7],
            },
            {
                "threshold": 1.0 - eps,
                "expected_clustering": [0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7],
            },
            {
                "threshold": 1.0,
                "expected_clustering": [0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7],
            },
        ]

        for test_case_dict in expected_clusterings:
            threshold = test_case_dict["threshold"]
            expected_clustering = test_case_dict["expected_clustering"]

            self.assertEqual(fingerprint_matrix.shape[0], len(expected_clustering))

            nof_cc, cc_labels = connected_components_iterative_algorithm(
                fingerprint_matrix, threshold
            )

            exp_nof_clusters = np.unique(expected_clustering).shape[0]
            self.assertEqual(exp_nof_clusters, nof_cc)

            self.assertTrue(np.equal(cc_labels, expected_clustering).all())

    def test_connected_components_iterative_chunking(self) -> None:
        """Test the chunking of the iterative connected component clustering algorithm."""
        # test standard case: two samples with a similarity of >=0.5 group into the same cluster

        fingerprint_matrix = csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

        nof_cc, cc_labels = connected_components_iterative_algorithm(
            fingerprint_matrix, 0.5, 2
        )
        self.assertEqual(nof_cc, 2)
        self.assertTrue(np.equal(cc_labels, [0, 1, 0]).all())

        # test equals case: two samples with a similarity of 1 group into the same cluster
        nof_cc, cc_labels = connected_components_iterative_algorithm(
            fingerprint_matrix, 1.0, 2
        )
        self.assertEqual(nof_cc, 2)
        self.assertTrue(np.equal(cc_labels, [0, 1, 0]).all())

        # test chunk size 1
        nof_cc, cc_labels = connected_components_iterative_algorithm(
            fingerprint_matrix, 0.5, 1
        )
        self.assertEqual(nof_cc, 2)
        self.assertTrue(np.equal(cc_labels, [0, 1, 0]).all())

        # test chunk size 3
        nof_cc, cc_labels = connected_components_iterative_algorithm(
            fingerprint_matrix, 0.5, 3
        )
        self.assertEqual(nof_cc, 2)
        self.assertTrue(np.equal(cc_labels, [0, 1, 0]).all())

        # test chunk size 4
        nof_cc, cc_labels = connected_components_iterative_algorithm(
            fingerprint_matrix, 0.5, 4
        )
        self.assertEqual(nof_cc, 2)
        self.assertTrue(np.equal(cc_labels, [0, 1, 0]).all())

    def test_calc_chunk_size_from_memory_requirement_for_tanimoto_similarity_sparse(
        self,
    ) -> None:
        """Test the calculation of the chunk size from the memory requirement."""
        matrix = csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        self.assertEqual(matrix.todense().nbytes, 72)

        # test memory_cutoff less than memory of row
        self.assertEqual(
            calc_chunk_size_from_memory_requirement(
                matrix.shape[0],
                matrix.shape[1],
                matrix.dtype.itemsize,
                23 / (1 << 30),  # 23 byte in GB. A row is 24 byte.
            ),
            1,
        )
        self.assertEqual(
            calc_chunk_size_from_memory_requirement(
                matrix.shape[0],
                matrix.shape[1],
                matrix.dtype.itemsize,
                24 / (1 << 30),  # 24 byte in GB. Exactly a row.
            ),
            1,
        )
        self.assertEqual(
            calc_chunk_size_from_memory_requirement(
                matrix.shape[0],
                matrix.shape[1],
                matrix.dtype.itemsize,
                25 / (1 << 30),  # 25 byte in GB. Slightly more than one row.
            ),
            1,
        )
        self.assertEqual(
            calc_chunk_size_from_memory_requirement(
                matrix.shape[0],
                matrix.shape[1],
                matrix.dtype.itemsize,
                47 / (1 << 30),  # 47 byte in GB. Almost two rows.
            ),
            1,
        )
        self.assertEqual(
            calc_chunk_size_from_memory_requirement(
                matrix.shape[0],
                matrix.shape[1],
                matrix.dtype.itemsize,
                48 / (1 << 30),  # 48 byte in GB. Exactly two rows.
            ),
            2,
        )
        self.assertEqual(
            calc_chunk_size_from_memory_requirement(
                matrix.shape[0],
                matrix.shape[1],
                matrix.dtype.itemsize,
                49 / (1 << 30),  # 49 byte in GB. Slightly more than two rows.
            ),
            2,
        )
        self.assertEqual(
            calc_chunk_size_from_memory_requirement(
                matrix.shape[0],
                matrix.shape[1],
                matrix.dtype.itemsize,
                71 / (1 << 30),  # 71 byte in GB. Slightly less than two rows.
            ),
            2,
        )
        self.assertEqual(
            calc_chunk_size_from_memory_requirement(
                matrix.shape[0],
                matrix.shape[1],
                matrix.dtype.itemsize,
                72 / (1 << 30),  # 72 byte in GB. Exact size of matrix.
            ),
            3,
        )
        self.assertEqual(
            calc_chunk_size_from_memory_requirement(
                matrix.shape[0],
                matrix.shape[1],
                matrix.dtype.itemsize,
                73 / (1 << 30),  # 73 byte in GB. Slightly more than full matrix.
            ),
            3,
        )

        # test memory cutoff larger than matrix: chunk size contains all rows
        self.assertEqual(
            calc_chunk_size_from_memory_requirement(
                matrix.shape[0], matrix.shape[1], matrix.dtype.itemsize, 0.5
            ),  # 500mb
            3,
        )
