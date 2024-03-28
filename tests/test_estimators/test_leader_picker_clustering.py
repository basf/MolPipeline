"""Test leader picker clustering estimator."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np
from rdkit import DataStructs

from molpipeline import Pipeline
from molpipeline.any2mol import AutoToMol
from molpipeline.estimators import LeaderPickerClustering
from molpipeline.mol2any import MolToMorganFP


class TestLeaderPickerEstimator(unittest.TestCase):
    """Test LeaderPicker clustering estimator."""

    def test_leader_picker_clustering_estimator(self) -> None:
        """Test LeaderPicker clustering estimator."""

        fingerprint_matrix = [
            DataStructs.CreateFromBitString(x)
            for x in [
                "000",  # 0
                "100",  # 1
                "110",  # 2
                "101",  # 3
                "010",  # 4
                "011",  # 5
                "001",  # 6
                "111",  # 7
                "000",
                "100",
                "110",
                "101",
                "010",
                "011",
                "001",
                "111",
            ]
        ]

        eps: float = 1e-10

        expected_clusterings: list[dict[str, Any]] = [
            {
                "threshold": 0.0,
                "expected_clustering": [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
            },
            {
                "threshold": 0.0 + eps,
                "expected_clustering": [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
            },
            {
                "threshold": 1 / 3 - eps,
                "expected_clustering": [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
            },
            {
                "threshold": 1 / 3,
                "expected_clustering": [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
            },
        ]

        for test_case_dict in expected_clusterings:
            threshold = test_case_dict["threshold"]
            expected_clustering = test_case_dict["expected_clustering"]

            self.assertEqual(len(fingerprint_matrix), len(expected_clustering))

            estimator = LeaderPickerClustering(distance_threshold=threshold)
            actual_labels = estimator.fit_predict(fingerprint_matrix)

            exp_nof_clusters = np.unique(expected_clustering).shape[0]
            self.assertEqual(exp_nof_clusters, estimator.n_clusters_)

            self.assertTrue(np.equal(actual_labels, expected_clustering).all())

    def test_leader_picker_pipeline(self) -> None:
        """Test leader picker clustering in pipeline."""

        test_smiles = ["C", "N", "c1ccccc1", "c1ccc(O)cc1", "CCCCCCO", "CCCCCCC"]

        distances = [0.05, 0.95]
        expected_labels = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 2, 3, 3]]
        expected_centroids = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 4]]

        for dist, exp_labels, exp_centroids in zip(
            distances, expected_labels, expected_centroids
        ):

            leader_picker = LeaderPickerClustering(distance_threshold=dist)
            pipeline = Pipeline(
                [
                    ("auto2mol", AutoToMol()),
                    (
                        "morgan2",
                        MolToMorganFP(
                            return_as="explicit_bit_vect", n_bits=1024, radius=2
                        ),
                    ),
                    ("leader_picker", leader_picker),
                ],
            )

            actual_labels = pipeline.fit_predict(test_smiles)

            self.assertTrue(np.equal(actual_labels, exp_labels).all())
            self.assertIsNotNone(leader_picker.centroids_)
            self.assertTrue(np.equal(leader_picker.centroids_, exp_centroids).all())  # type: ignore
