"""This module tests the precomputed kernel estimators."""

import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from sklearn.neighbors import KNeighborsClassifier

from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.error_handling import ErrorFilter, ErrorReplacer
from molpipeline.pipeline_elements.mol2any import MolToFoldedMorganFingerprint
from molpipeline.pipeline_elements.post_prediction import PostPredictionWrapper
from molpipeline.sklearn_estimators.similarity_transformation import (
    TanimotoSimilarityToTraining,
)
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


class TestTanimotoSimilarityToTraining(unittest.TestCase):
    """Test for the TanimotoSimilarityToTraining."""

    def test_fit_transform(self) -> None:
        """Test if similarity calculation works for fit_transform."""
        # Reference: Only the fingerprint calculation
        morgan_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMolPipelineElement()),
                ("mol2fp", MolToFoldedMorganFingerprint()),
            ]
        )
        fingerprint = morgan_pipeline.fit_transform(COMPOUND_LIST)
        self_similarity = tanimoto_similarity_sparse(fingerprint, fingerprint)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMolPipelineElement()),
                ("mol2fp", MolToFoldedMorganFingerprint()),
                ("precompute_tanimoto", TanimotoSimilarityToTraining()),
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
        morgan_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMolPipelineElement()),
                ("mol2fp", MolToFoldedMorganFingerprint()),
            ]
        )
        fingerprint = morgan_pipeline.fit_transform(COMPOUND_LIST)
        self_similarity = tanimoto_similarity_sparse(fingerprint, fingerprint)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMolPipelineElement()),
                ("mol2fp", MolToFoldedMorganFingerprint()),
                ("precompute_tanimoto", TanimotoSimilarityToTraining()),
            ]
        )
        full_pipeline.fit(COMPOUND_LIST)
        pipeline_sim = full_pipeline.transform(COMPOUND_LIST)
        self.assertTrue(np.allclose(pipeline_sim, self_similarity))

    def test_fit_transform_rdkit(self) -> None:
        """Test if the similarity calculation matches the RDKit implementation for fit_transform."""
        # Reference: RDKit implementation
        fp_list = []
        for smi in COMPOUND_LIST:
            mol = Chem.MolFromSmiles(smi)
            fp_list.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2))
        sim = []
        for fp1 in fp_list:
            sim.append(BulkTanimotoSimilarity(fp1, fp_list))
        self_similarity = np.array(sim)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMolPipelineElement()),
                ("mol2fp", MolToFoldedMorganFingerprint()),
                ("precompute_tanimoto", TanimotoSimilarityToTraining()),
            ]
        )
        pipeline_sim = full_pipeline.fit_transform(COMPOUND_LIST)
        self.assertTrue(np.allclose(pipeline_sim, self_similarity))

    def test_fit_and_transform_rdkit(self) -> None:
        """Test if the similarity calculation matches the RDKit implementation for fit and transform separately."""
        # Reference: RDKit implementation
        fp_list = []
        for smi in COMPOUND_LIST:
            mol = Chem.MolFromSmiles(smi)
            fp_list.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2))
        sim = []
        for fp1 in fp_list:
            sim.append(BulkTanimotoSimilarity(fp1, fp_list))
        self_similarity = np.array(sim)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMolPipelineElement()),
                ("mol2fp", MolToFoldedMorganFingerprint()),
                ("precompute_tanimoto", TanimotoSimilarityToTraining()),
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
                ("smi2mol", SmilesToMolPipelineElement()),
                ("mol2fp", MolToFoldedMorganFingerprint()),
                ("precompute_tanimoto", TanimotoSimilarityToTraining()),
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
                ("smi2mol", SmilesToMolPipelineElement()),
                ("mol2fp", MolToFoldedMorganFingerprint()),
                ("precompute_tanimoto", TanimotoSimilarityToTraining()),
                ("nearest_neighbor", KNeighborsClassifier(n_neighbors=1)),
            ]
        )
        full_pipeline.fit(COMPOUND_LIST, IS_AROMATIC)
        prediction = full_pipeline.predict(COMPOUND_LIST)
        self.assertTrue(np.allclose(prediction, IS_AROMATIC))

        error_filter = ErrorFilter(filter_everything=True)
        error_replacer = ErrorReplacer.from_error_filter(
            error_filter, fill_value=np.nan
        )
        # Test if the error handling works
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMolPipelineElement()),
                ("error_filter", error_filter),
                ("mol2fp", MolToFoldedMorganFingerprint()),
                ("precompute_tanimoto", TanimotoSimilarityToTraining()),
                ("nearest_neighbor", KNeighborsClassifier(n_neighbors=1)),
                ("error_replacer", PostPredictionWrapper(error_replacer)),
            ]
        )
        full_pipeline.fit(COMPOUND_LIST + ["C#C#C"], IS_AROMATIC + [False])
        prediction = full_pipeline.predict(COMPOUND_LIST + ["C#C#C"])
        self.assertTrue(np.allclose(prediction[:-1], IS_AROMATIC))
        self.assertTrue(np.isnan(prediction[-1]))

        single_prediction = full_pipeline.predict(["C#C#C"])
        self.assertTrue(np.isnan(single_prediction))

        none_prediction = full_pipeline.predict([])
        self.assertListEqual(none_prediction, [])

    def test_fit_transform_distance(self) -> None:
        """Test if distance calculation works for fit_transform."""
        # Reference: Only the fingerprint calculation
        morgan_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMolPipelineElement()),
                ("mol2fp", MolToFoldedMorganFingerprint()),
            ]
        )
        fingerprint = morgan_pipeline.fit_transform(COMPOUND_LIST)
        self_distance = 1 - tanimoto_similarity_sparse(fingerprint, fingerprint)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMolPipelineElement()),
                ("mol2fp", MolToFoldedMorganFingerprint()),
                ("precompute_tanimoto", TanimotoSimilarityToTraining(distance=True)),
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
        morgan_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMolPipelineElement()),
                ("mol2fp", MolToFoldedMorganFingerprint()),
            ]
        )
        fingerprint = morgan_pipeline.fit_transform(COMPOUND_LIST)
        self_distance = 1 - tanimoto_similarity_sparse(fingerprint, fingerprint)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMolPipelineElement()),
                ("mol2fp", MolToFoldedMorganFingerprint()),
                ("precompute_tanimoto", TanimotoSimilarityToTraining(distance=True)),
            ]
        )
        full_pipeline.fit(COMPOUND_LIST)
        pipeline_distance = full_pipeline.transform(COMPOUND_LIST)
        self.assertTrue(np.allclose(pipeline_distance, self_distance))

    def test_fit_transform_rdkit_distance(self) -> None:
        """Test if the distance calculation matches the RDKit implementation for fit_transform."""
        # Reference: RDKit implementation
        fp_list = []
        for smi in COMPOUND_LIST:
            mol = Chem.MolFromSmiles(smi)
            fp_list.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2))
        sim = []
        for fp1 in fp_list:
            sim.append(BulkTanimotoSimilarity(fp1, fp_list))
        self_distance = 1 - np.array(sim)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMolPipelineElement()),
                ("mol2fp", MolToFoldedMorganFingerprint()),
                ("precompute_tanimoto", TanimotoSimilarityToTraining(distance=True)),
            ]
        )
        pipeline_distance = full_pipeline.fit_transform(COMPOUND_LIST)
        self.assertTrue(np.allclose(pipeline_distance, self_distance))

    def test_fit_and_transform_rdkit_distance(self) -> None:
        """Test if the distance calculation matches the RDKit implementation for fit and transform separately."""
        # Reference: RDKit implementation
        fp_list = []
        for smi in COMPOUND_LIST:
            mol = Chem.MolFromSmiles(smi)
            fp_list.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2))
        sim = []
        for fp1 in fp_list:
            sim.append(BulkTanimotoSimilarity(fp1, fp_list))
        self_distance = 1 - np.array(sim)

        # Setup the full pipeline
        full_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMolPipelineElement()),
                ("mol2fp", MolToFoldedMorganFingerprint()),
                ("precompute_tanimoto", TanimotoSimilarityToTraining(distance=True)),
            ]
        )
        full_pipeline.fit(COMPOUND_LIST)
        pipeline_distance = full_pipeline.transform(COMPOUND_LIST)
        self.assertTrue(np.allclose(pipeline_distance, self_distance))
