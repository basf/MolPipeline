"""Test cases for the nearest neighbors estimators."""

from unittest import TestCase

import numpy as np
from sklearn.base import clone

from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.error_handling import ErrorFilter, ErrorReplacer
from molpipeline.pipeline_elements.mol2any import MolToFoldedMorganFingerprint
from molpipeline.pipeline_elements.post_prediction import PostPredictionWrapper
from molpipeline.sklearn_estimators.algorithm.nearest_neighbor import (
    NamedNearestNeighbors,
)
from molpipeline.sklearn_estimators.similarity_transformation import TanimotoToTraining
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
                ("mol", SmilesToMolPipelineElement()),
                ("fingerprint", MolToFoldedMorganFingerprint(output_datatype="dense")),
                ("lookup", NamedNearestNeighbors(n_neighbors=1, metric="jaccard")),
            ]
        )
        result = model.fit_predict(TEST_SMILES, TEST_SMILES).tolist()
        self.assertListEqual(result, [[nn_list[0]] for nn_list in TWO_NN])

    def test_fit_predict_two(self) -> None:
        """Test the fit_predict method with two prediced NN."""
        model = Pipeline(
            [
                ("mol", SmilesToMolPipelineElement()),
                ("fingerprint", MolToFoldedMorganFingerprint(output_datatype="dense")),
                ("lookup", NamedNearestNeighbors(n_neighbors=2, metric="jaccard")),
            ]
        )
        result = model.fit_predict(TEST_SMILES, TEST_SMILES).tolist()
        self.assertListEqual(result, TWO_NN)

    def test_fit_and_predict_with_distance(self) -> None:
        """Test the fit_predict method with distance."""
        model = Pipeline(
            [
                ("mol", SmilesToMolPipelineElement()),
                ("fingerprint", MolToFoldedMorganFingerprint(output_datatype="dense")),
                ("lookup", NamedNearestNeighbors(n_neighbors=2, metric="jaccard")),
            ]
        )
        model.fit(TEST_SMILES, TEST_SMILES)
        result = model.predict(TEST_SMILES, **{"return_distance": True})
        neighbors = result[:, :, 0]
        distances = result[:, :, 1].astype(np.float_)
        self.assertListEqual(neighbors.tolist(), TWO_NN)
        self.assertTrue(np.allclose(1 - distances, TWO_NN_SIMILARITIES))

    def test_fit_predict_with_n_neigbours(self) -> None:
        """Test the fit_predict method with parameter n_neighbors."""
        model = Pipeline(
            [
                ("mol", SmilesToMolPipelineElement()),
                ("fingerprint", MolToFoldedMorganFingerprint(output_datatype="dense")),
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
                ("mol", SmilesToMolPipelineElement()),
                ("fingerprint", MolToFoldedMorganFingerprint(output_datatype="dense")),
                ("lookup", NamedNearestNeighbors(n_neighbors=1, metric="jaccard")),
            ]
        )
        model.fit(TEST_SMILES, TEST_SMILES)
        result = model.predict(TEST_SMILES, **{"n_neighbors": 2}).tolist()
        self.assertListEqual(result, TWO_NN)

    def test_fit_and_predict_with_fit_predict(self) -> None:
        """Test if the fit_predict method gives same results as callaing fit and predict separately."""
        model1 = Pipeline(
            [
                ("mol", SmilesToMolPipelineElement()),
                ("fingerprint", MolToFoldedMorganFingerprint(output_datatype="dense")),
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
                ("mol", SmilesToMolPipelineElement()),
                ("fingerprint", MolToFoldedMorganFingerprint(output_datatype="sparse")),
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
                ("mol", SmilesToMolPipelineElement()),
                ("fingerprint", MolToFoldedMorganFingerprint()),
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
                ("mol", SmilesToMolPipelineElement()),
                ("error_filter", error_filter),
                ("fingerprint", MolToFoldedMorganFingerprint(output_datatype="dense")),
                ("lookup", NamedNearestNeighbors(n_neighbors=2, metric="jaccard")),
                (
                    "error_replacer",
                    PostPredictionWrapper(
                        ErrorReplacer.from_error_filter(
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
                ("mol", SmilesToMolPipelineElement()),
                ("error_filter", error_filter),
                ("fingerprint", MolToFoldedMorganFingerprint(output_datatype="dense")),
                ("lookup", NamedNearestNeighbors(n_neighbors=2, metric="jaccard")),
                (
                    "error_replacer",
                    PostPredictionWrapper(
                        ErrorReplacer.from_error_filter(
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
            1 - np.allclose(distances[1:, :].astype(np.float_), TWO_NN_SIMILARITIES)
        )
