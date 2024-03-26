"""Test connected component clustering estimator."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from molpipeline.estimators import ConnectedComponentClustering


class TestConnectedComponentClusteringEstimator(unittest.TestCase):
    """Test connected component clustering estimator."""

    def test_connected_component_clustering_estimator(self) -> None:
        """Test connected component clustering estimator."""
        ccc = ConnectedComponentClustering(distance_threshold=0.5, max_memory_usage=0.1)

        # test no chunking needed
        self.assertEqual(
            ccc.fit_predict(csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]])).tolist(),
            [0, 1, 0],
        )
        self.assertEqual(ccc.n_clusters_, 2)

        # test chunking needed
        matrix = csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        nof_bytes_per_row = np.dtype("float64").itemsize * matrix.shape[1] / (1 << 30)
        ccc = ConnectedComponentClustering(
            distance_threshold=0.5,
            max_memory_usage=nof_bytes_per_row,
        )
        self.assertEqual(
            ccc.fit_predict(matrix).tolist(),
            [0, 1, 0],
        )
        self.assertEqual(ccc.n_clusters_, 2)
