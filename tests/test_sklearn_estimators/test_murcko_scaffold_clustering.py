"""Test Murcko scaffold clustering estimator."""
from __future__ import annotations

import unittest

import numpy as np

from molpipeline.sklearn_estimators.murcko_scaffold_clustering import (
    MurckoScaffoldClustering,
)

SCAFFOLD_SMILES: list[str] = [
    "Cc1ccccc1",
    "Cc1cc(Oc2nccc(CCC)c2)ccc1",
    "c1ccccc1",
]

LINEAR_SMILES: list[str] = ["CC", "CCC", "CCCN"]


class TestMurckoScaffoldClusteringEstimator(unittest.TestCase):
    """Test Murcko scaffold clustering estimator."""

    def test_murcko_scaffold_clustering_ignore(self) -> None:
        """Test Murcko scaffold clustering estimator."""

        estimator_ignore_linear: MurckoScaffoldClustering = MurckoScaffoldClustering(
            use_smiles=True, n_jobs=1, linear_molecules_strategy="ignore"
        )

        # test basic scaffold-based clustering works as intended
        scafffold_cluster_labels = estimator_ignore_linear.fit_predict(SCAFFOLD_SMILES)
        expected_scaffold_labels = [1.0, 0.0, 1.0]

        self.assertEqual(estimator_ignore_linear.n_clusters_, 2)
        self.assertListEqual(
            scafffold_cluster_labels.tolist(), expected_scaffold_labels
        )

        # test linear molecule handling. We expect the linear molecules to be clustered in the same cluster
        input_smiles = SCAFFOLD_SMILES + LINEAR_SMILES
        cluster_labels = estimator_ignore_linear.fit_predict(input_smiles)
        nan_mask = np.isnan(cluster_labels)
        expected_nan_mask = [False, False, False, True, True, True]

        self.assertEqual(estimator_ignore_linear.n_clusters_, 2)
        self.assertListEqual(nan_mask.tolist(), expected_nan_mask)
        self.assertListEqual(cluster_labels[~nan_mask].tolist(), [1.0, 0.0, 1.0])

    def test_murcko_scaffold_clustering_own_cluster(self) -> None:
        """Test Murcko scaffold clustering estimator."""
        # create new estimator with "own_cluster" strategy
        estimator_cluster_linear: MurckoScaffoldClustering = MurckoScaffoldClustering(
            use_smiles=True, n_jobs=1, linear_molecules_strategy="own_cluster"
        )

        # test linear molecule handling. We expect the linear molecules to be clustered in the same cluster
        input_smiles = SCAFFOLD_SMILES + LINEAR_SMILES
        cluster_labels = estimator_cluster_linear.fit_predict(input_smiles)
        expected_cluster_labels = [1.0, 0.0, 1.0, 2.0, 2.0, 2.0]
        self.assertEqual(estimator_cluster_linear.n_clusters_, 3)
        self.assertListEqual(cluster_labels.tolist(), expected_cluster_labels)
