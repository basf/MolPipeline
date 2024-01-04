"""Test Murcko scaffold clustering estimator."""
from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import adjusted_rand_score

from molpipeline.sklearn_estimators.murcko_scaffold_clustering import (
    MurckoScaffoldClustering,
)


class TestMurckoScaffoldClusteringEstimator(unittest.TestCase):
    """Test Murcko scaffold clustering estimator."""

    def test_murcko_scaffold_clustering_estimator(self) -> None:
        """Test Murcko scaffold clustering estimator."""

        scaffold_smiles: list[str] = [
            "Cc1ccccc1",
            "Cc1cc(Oc2nccc(CCC)c2)ccc1",
            "c1ccccc1",
        ]
        linear_smiles: list[str] = ["CC", "CCC", "CCCN"]

        estimator_ignore_linear: MurckoScaffoldClustering = MurckoScaffoldClustering(
            n_jobs=1, linear_molecules_strategy="ignore"
        )

        # test basic scaffold-based clustering works as intended
        input_smiles = scaffold_smiles
        self.assertTrue(
            np.allclose(
                adjusted_rand_score(
                    estimator_ignore_linear.fit_predict(input_smiles), [0, 1, 0]
                ),
                1.0,
                atol=1e-10,
            )
        )
        self.assertEqual(estimator_ignore_linear.n_clusters_, 2)

        # test linear molecule handling. We expect the linear molecules to be clustered in the same cluster
        input_smiles = scaffold_smiles + linear_smiles
        cluster_labels = estimator_ignore_linear.fit_predict(input_smiles)
        self.assertTrue(
            np.equal(
                np.isnan(cluster_labels), [False, False, False, True, True, True]
            ).all()
        )
        self.assertTrue(
            np.allclose(
                adjusted_rand_score(
                    cluster_labels[~np.isnan(cluster_labels)], [0, 1, 0]
                ),
                1.0,
                atol=1e-10,
            )
        )
        self.assertEqual(estimator_ignore_linear.n_clusters_, 2)

        # create new estimator with "own_cluster" strategy
        estimator_cluster_linear: MurckoScaffoldClustering = MurckoScaffoldClustering(
            n_jobs=1, linear_molecules_strategy="own_cluster"
        )

        # test linear molecule handling. We expect the linear molecules to be clustered in the same cluster
        input_smiles = scaffold_smiles + linear_smiles
        self.assertTrue(
            np.allclose(
                adjusted_rand_score(
                    estimator_cluster_linear.fit_predict(input_smiles),
                    [0, 1, 0, 2, 2, 2],
                ),
                1.0,
                atol=1e-10,
            )
        )
        self.assertEqual(estimator_cluster_linear.n_clusters_, 3)
