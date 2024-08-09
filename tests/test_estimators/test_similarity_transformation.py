"""This module tests the precomputed kernel estimators."""

from __future__ import annotations

import unittest

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier

from molpipeline import ErrorFilter, FilterReinserter, Pipeline, PostPredictionWrapper
from molpipeline.any2mol import SmilesToMol
from molpipeline.estimators import TanimotoToTraining
from molpipeline.mol2any import MolToMorganFP
from molpipeline.utils.kernel import tanimoto_similarity_sparse

COMPOUND_LIST = [
    "CCC",
    "CCOCC",
    "CCNCC",
    "CCO",
    "CCSC",
    "c1ccccc1",
    "c1c[nH]cc1",
]

IS_AROMATIC = [
    False,
    False,
    False,
    False,
    False,
    True,
    True,
]


def _generate_morgan_fingerprints(compound_list: list[str]) -> sparse.csr_matrix:
    """Generate the Morgan fingerprints.

    Parameters
    ----------
    compound_list: list[str]
        List of SMILES strings.

    Returns
    -------
    sparse.csr_matrix
        Morgan fingerprints.
    """
    morgan_pipeline = Pipeline(
        [
            ("smi2mol", SmilesToMol()),
            ("mol2fp", MolToMorganFP()),
        ]
    )
    fingerprint = morgan_pipeline.fit_transform(compound_list)
    return fingerprint


def _calculate_rdkit_self_similarity(
    compound_list: list[str],
) -> npt.NDArray[np.float64]:
    """Calculate the self similarity using RDKit.

    Parameters
    ----------
    compound_list: list[str]
        List of SMILES strings.

    Returns
    -------
    npt.NDArray[np.float64]
        Self similarity.
    """
    fp_list = []
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    for smi in compound_list:
        mol = Chem.MolFromSmiles(smi)
        fp_list.append(morgan_generator.GetFingerprint(mol))
    sim = []
    for fp1 in fp_list:
        sim.append(BulkTanimotoSimilarity(fp1, fp_list))
    return np.array(sim)


class TestTanimotoSimilarityToTraining(unittest.TestCase):
    """Test for the TanimotoSimilarityToTraining."""

    def test_fit_transform(self) -> None:
        """Test if similarity calculation works for fit_transform."""
        # Reference: Only the fingerprint calculation
        fingerprint = _generate_morgan_fingerprints(COMPOUND_LIST)
        self_similarity = tanimoto_similarity_sparse(fingerprint, fingerprint)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("mol2fp", MolToMorganFP()),
                ("precompute_tanimoto", TanimotoToTraining()),
            ]
        )

        # Test fit_transform
        pipeline_sim = full_pipeline.fit_transform(COMPOUND_LIST)
        self.assertTrue(np.allclose(pipeline_sim, self_similarity))

        # Test transform
        reversed_pipeline_sim = full_pipeline.transform(COMPOUND_LIST[::-1])
        reversed_self_similarity = tanimoto_similarity_sparse(
            fingerprint[::-1], fingerprint
        )
        self.assertTrue(np.allclose(reversed_pipeline_sim, reversed_self_similarity))

    def test_fit_and_transform(self) -> None:
        """Test if the similarity calculation works for fit and transform separately."""
        # Reference: Only the fingerprint calculation
        fingerprint = _generate_morgan_fingerprints(COMPOUND_LIST)
        self_similarity = tanimoto_similarity_sparse(fingerprint, fingerprint)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("mol2fp", MolToMorganFP()),
                ("precompute_tanimoto", TanimotoToTraining()),
            ]
        )
        full_pipeline.fit(COMPOUND_LIST)
        pipeline_sim = full_pipeline.transform(COMPOUND_LIST)
        self.assertTrue(np.allclose(pipeline_sim, self_similarity))

    def test_fit_transform_rdkit(self) -> None:
        """Test if the similarity calculation matches the RDKit implementation for fit_transform."""
        # Reference: RDKit implementation
        self_similarity = _calculate_rdkit_self_similarity(COMPOUND_LIST)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("mol2fp", MolToMorganFP()),
                ("precompute_tanimoto", TanimotoToTraining()),
            ]
        )
        pipeline_sim = full_pipeline.fit_transform(COMPOUND_LIST)
        self.assertTrue(np.allclose(pipeline_sim, self_similarity))

    def test_fit_and_transform_rdkit(self) -> None:
        """Test if the similarity calculation matches the RDKit implementation for fit and transform separately."""
        self_similarity = _calculate_rdkit_self_similarity(COMPOUND_LIST)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("mol2fp", MolToMorganFP()),
                ("precompute_tanimoto", TanimotoToTraining()),
            ]
        )
        full_pipeline.fit(COMPOUND_LIST)
        pipeline_sim = full_pipeline.transform(COMPOUND_LIST)
        self.assertTrue(np.allclose(pipeline_sim, self_similarity))

    def test_nearest_neighbor_pipeline(self) -> None:
        """Test if the nearest neighbor pipeline works."""
        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("mol2fp", MolToMorganFP()),
                ("precompute_tanimoto", TanimotoToTraining()),
                ("nearest_neighbor", KNeighborsClassifier(n_neighbors=1)),
            ]
        )
        full_pipeline.fit(COMPOUND_LIST, IS_AROMATIC)
        prediction = full_pipeline.predict(COMPOUND_LIST)
        self.assertTrue(np.allclose(prediction, IS_AROMATIC))

    def test_error_handling(self) -> None:
        """Test if the error handling works."""
        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("mol2fp", MolToMorganFP()),
                ("precompute_tanimoto", TanimotoToTraining()),
                ("nearest_neighbor", KNeighborsClassifier(n_neighbors=1)),
            ]
        )
        full_pipeline.fit(COMPOUND_LIST, IS_AROMATIC)
        prediction = full_pipeline.predict(COMPOUND_LIST)
        self.assertTrue(np.allclose(prediction, IS_AROMATIC))

        error_filter = ErrorFilter(filter_everything=True)
        error_replacer = FilterReinserter.from_error_filter(
            error_filter, fill_value=np.nan
        )
        # Test if the error handling works
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("error_filter", error_filter),
                ("mol2fp", MolToMorganFP()),
                ("precompute_tanimoto", TanimotoToTraining()),
                ("nearest_neighbor", KNeighborsClassifier(n_neighbors=1)),
                ("error_replacer", PostPredictionWrapper(error_replacer)),
            ]
        )
        full_pipeline.fit(COMPOUND_LIST + ["C#C#C"], IS_AROMATIC + [False])
        prediction = full_pipeline.predict(COMPOUND_LIST + ["C#C#C"]).tolist()
        self.assertListEqual(prediction[:-1], IS_AROMATIC)
        self.assertTrue(np.isnan(prediction[-1]))

        single_prediction = full_pipeline.predict(["C#C#C"])
        self.assertTrue(np.isnan(single_prediction))

        none_prediction = full_pipeline.predict([])
        self.assertListEqual(none_prediction, [])

    def test_fit_transform_distance(self) -> None:
        """Test if distance calculation works for fit_transform."""
        # Reference: Only the fingerprint calculation
        fingerprint = _generate_morgan_fingerprints(COMPOUND_LIST)
        self_distance = 1 - tanimoto_similarity_sparse(fingerprint, fingerprint)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("mol2fp", MolToMorganFP()),
                ("precompute_tanimoto", TanimotoToTraining(distance=True)),
            ]
        )

        # Test fit_transform
        pipeline_distance = full_pipeline.fit_transform(COMPOUND_LIST)
        self.assertTrue(np.allclose(pipeline_distance, self_distance))

        # Test transform
        reversed_pipeline_distance = full_pipeline.transform(COMPOUND_LIST[::-1])
        reversed_self_distance = 1 - tanimoto_similarity_sparse(
            fingerprint[::-1], fingerprint
        )
        self.assertTrue(np.allclose(reversed_pipeline_distance, reversed_self_distance))

    def test_fit_and_transform_distance(self) -> None:
        """Test if the distance calculation works for fit and transform separately."""
        # Reference: Only the fingerprint calculation
        fingerprint = _generate_morgan_fingerprints(COMPOUND_LIST)
        self_distance = 1 - tanimoto_similarity_sparse(fingerprint, fingerprint)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("mol2fp", MolToMorganFP()),
                ("precompute_tanimoto", TanimotoToTraining(distance=True)),
            ]
        )
        full_pipeline.fit(COMPOUND_LIST)
        pipeline_distance = full_pipeline.transform(COMPOUND_LIST)
        self.assertTrue(np.allclose(pipeline_distance, self_distance))

    def test_fit_transform_rdkit_distance(self) -> None:
        """Test if the distance calculation matches the RDKit implementation for fit_transform."""
        # Reference: RDKit implementation
        self_distance = 1.0 - _calculate_rdkit_self_similarity(COMPOUND_LIST)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("mol2fp", MolToMorganFP()),
                ("precompute_tanimoto", TanimotoToTraining(distance=True)),
            ]
        )
        pipeline_distance = full_pipeline.fit_transform(COMPOUND_LIST)
        self.assertTrue(np.allclose(pipeline_distance, self_distance))

    def test_fit_and_transform_rdkit_distance(self) -> None:
        """Test if the distance calculation matches the RDKit implementation for fit and transform separately."""
        # Reference: RDKit implementation
        self_distance = 1.0 - _calculate_rdkit_self_similarity(COMPOUND_LIST)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("mol2fp", MolToMorganFP()),
                ("precompute_tanimoto", TanimotoToTraining(distance=True)),
            ]
        )
        full_pipeline.fit(COMPOUND_LIST)
        pipeline_distance = full_pipeline.transform(COMPOUND_LIST)
        self.assertTrue(np.allclose(pipeline_distance, self_distance))
