"""Test cases for the nearest neighbors estimators."""

from unittest import TestCase

import numpy as np
from sklearn.base import clone

from molpipeline import ErrorFilter, FilterReinserter, Pipeline, PostPredictionWrapper
from molpipeline.any2mol import SmilesToMol
from molpipeline.estimators import NamedNearestNeighbors, TanimotoToTraining
from molpipeline.mol2any import MolToMorganFP
from molpipeline.utils.kernel import tanimoto_distance_sparse

TEST_SMILES = [
    "c1ccccc1",
    "c1cc(-C(=O)O)ccc1",
    "CCCCCCN",
    "CCCCCCO",
]

TWO_NN = [
    ["c1ccccc1", "c1cc(-C(=O)O)ccc1"],
    ["c1cc(-C(=O)O)ccc1", "c1ccccc1"],
    ["CCCCCCN", "CCCCCCO"],
    ["CCCCCCO", "CCCCCCN"],
]

TWO_NN_SIMILARITIES = np.array(
    [
        [1.0, 3 / 14],
        [1.0, 3 / 14],
        [1.0, 4 / 9],
        [1.0, 4 / 9],
    ]
)


class TestNamedNearestNeighbors(TestCase):
    """Test the NamedNearestNeighbors class if correct names are returned."""

    def test_fit_predict_one(self) -> None:
        """Test the fit_predict method with one prediced NN."""
        model = Pipeline(
            [
                ("mol", SmilesToMol()),
                ("fingerprint", MolToMorganFP(return_as="dense")),
                ("lookup", NamedNearestNeighbors(n_neighbors=1, metric="jaccard")),
            ]
        )
        result = model.fit_predict(TEST_SMILES, TEST_SMILES).tolist()
        self.assertListEqual(result, [[nn_list[0]] for nn_list in TWO_NN])

    def test_fit_predict_two(self) -> None:
        """Test the fit_predict method with two prediced NN."""
        model = Pipeline(
            [
                ("mol", SmilesToMol()),
                ("fingerprint", MolToMorganFP(return_as="dense")),
                ("lookup", NamedNearestNeighbors(n_neighbors=2, metric="jaccard")),
            ]
        )
        result = model.fit_predict(TEST_SMILES, TEST_SMILES).tolist()
        self.assertListEqual(result, TWO_NN)

    def test_fit_and_predict_with_distance(self) -> None:
        """Test the fit_predict method with distance."""
        model = Pipeline(
            [
                ("mol", SmilesToMol()),
                ("fingerprint", MolToMorganFP(return_as="dense")),
                ("lookup", NamedNearestNeighbors(n_neighbors=2, metric="jaccard")),
            ]
        )
        model.fit(TEST_SMILES, TEST_SMILES)
        result = model.predict(TEST_SMILES, **{"return_distance": True})
        neighbors = result[:, :, 0]
        distances = result[:, :, 1].astype(np.float64)
        self.assertListEqual(neighbors.tolist(), TWO_NN)
        self.assertTrue(np.allclose(1 - distances, TWO_NN_SIMILARITIES))

    def test_fit_predict_with_n_neigbours(self) -> None:
        """Test the fit_predict method with parameter n_neighbors."""
        model = Pipeline(
            [
                ("mol", SmilesToMol()),
                ("fingerprint", MolToMorganFP(return_as="dense")),
                ("lookup", NamedNearestNeighbors(n_neighbors=1, metric="jaccard")),
            ]
        )
        result = model.fit_predict(
            TEST_SMILES, TEST_SMILES, lookup__n_neighbors=2
        ).tolist()
        self.assertListEqual(result, TWO_NN)

    def test_fit_and_predict_with_n_neigbours(self) -> None:
        """Test the fit_predict method with parameter n_neighbors."""
        model = Pipeline(
            [
                ("mol", SmilesToMol()),
                ("fingerprint", MolToMorganFP(return_as="dense")),
                ("lookup", NamedNearestNeighbors(n_neighbors=1, metric="jaccard")),
            ]
        )
        model.fit(TEST_SMILES, TEST_SMILES)
        result = model.predict(TEST_SMILES, **{"n_neighbors": 2}).tolist()
        self.assertListEqual(result, TWO_NN)

    def test_fit_and_predict_with_fit_predict(self) -> None:
        """Test if the fit_predict method gives same results as calling fit and predict separately."""
        model1 = Pipeline(
            [
                ("mol", SmilesToMol()),
                ("fingerprint", MolToMorganFP(return_as="dense")),
                ("lookup", NamedNearestNeighbors(n_neighbors=2, metric="jaccard")),
            ]
        )
        model2 = clone(model1)
        result1 = model1.fit_predict(TEST_SMILES, TEST_SMILES).tolist()
        model2.fit(TEST_SMILES, TEST_SMILES)
        result2 = model2.predict(TEST_SMILES).tolist()
        self.assertListEqual(result1, TWO_NN)
        self.assertListEqual(result2, TWO_NN)
        self.assertListEqual(result1, result2)

    def test_fit_predict_custom_metric(self) -> None:
        """Test if the custom metric can be used."""
        model = Pipeline(
            [
                ("mol", SmilesToMol()),
                ("fingerprint", MolToMorganFP(return_as="sparse")),
                (
                    "lookup",
                    NamedNearestNeighbors(
                        n_neighbors=2, metric=tanimoto_distance_sparse
                    ),
                ),
            ]
        )
        result = model.fit_predict(TEST_SMILES, TEST_SMILES).tolist()
        self.assertListEqual(result, TWO_NN)

    def test_fit_predict_precomputed(self) -> None:
        """Test the fit_predict method with precomputed similarities."""
        model = Pipeline(
            [
                ("mol", SmilesToMol()),
                ("fingerprint", MolToMorganFP()),
                ("tanimoto", TanimotoToTraining(distance=True)),
                ("lookup", NamedNearestNeighbors(n_neighbors=2, metric="precomputed")),
            ]
        )
        result = model.fit_predict(TEST_SMILES, TEST_SMILES).tolist()
        self.assertListEqual(result, TWO_NN)

    def test_fit_predict_invalid(self) -> None:
        """Test the fit_predict method with invalid smiles."""

        with_invald_smiles = ["CC1CC"] + TEST_SMILES

        error_filter = ErrorFilter(filter_everything=True)
        model = Pipeline(
            [
                ("mol", SmilesToMol()),
                ("error_filter", error_filter),
                ("fingerprint", MolToMorganFP(return_as="dense")),
                ("lookup", NamedNearestNeighbors(n_neighbors=2, metric="jaccard")),
                (
                    "error_replacer",
                    PostPredictionWrapper(
                        FilterReinserter.from_error_filter(
                            error_filter, fill_value="invalid"
                        )
                    ),
                ),
            ]
        )
        result = model.fit_predict(with_invald_smiles, with_invald_smiles).tolist()
        self.assertListEqual(result, [["invalid", "invalid"]] + TWO_NN)

        result_only_valid = model.predict(TEST_SMILES).tolist()
        self.assertListEqual(result_only_valid, TWO_NN)

    def test_fit_and_predict_invalid_with_distance(self) -> None:
        """Test the fit_predict method with invalid smiles and distance."""
        with_invald_smiles = ["CC1CC"] + TEST_SMILES

        error_filter = ErrorFilter(filter_everything=True)
        model = Pipeline(
            [
                ("mol", SmilesToMol()),
                ("error_filter", error_filter),
                ("fingerprint", MolToMorganFP(return_as="dense")),
                ("lookup", NamedNearestNeighbors(n_neighbors=2, metric="jaccard")),
                (
                    "error_replacer",
                    PostPredictionWrapper(
                        FilterReinserter.from_error_filter(
                            error_filter, fill_value="invalid"
                        )
                    ),
                ),
            ]
        )
        model.fit(with_invald_smiles, with_invald_smiles)
        result = model.predict(with_invald_smiles, **{"return_distance": True})
        neighbors = result[:, :, 0]
        distances = result[:, :, 1]
        self.assertListEqual(neighbors.tolist(), [["invalid", "invalid"]] + TWO_NN)
        self.assertTrue(
            1 - np.allclose(distances[1:, :].astype(np.float64), TWO_NN_SIMILARITIES)
        )
