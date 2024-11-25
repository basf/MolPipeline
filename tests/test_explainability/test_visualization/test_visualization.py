"""Test visualization methods for explanations."""

import unittest
from typing import ClassVar

import numpy as np
import scipy
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.explainability import (
    SHAPFeatureAndAtomExplanation,
    SHAPFeatureExplanation,
    SHAPTreeExplainer,
    structure_heatmap,
    structure_heatmap_shap,
)
from molpipeline.explainability.explainer import SHAPKernelExplainer
from molpipeline.mol2any import MolToMorganFP
from molpipeline.utils.subpipeline import get_featurization_subpipeline

TEST_SMILES = ["CC", "CCO", "COC", "c1ccccc1(N)", "CCC(-O)O", "CCCN"]
CONTAINS_OX = [0, 1, 1, 0, 1, 0]

_RANDOM_STATE = 67056


def _get_test_morgan_rf_pipeline() -> Pipeline:
    """Get a test pipeline with Morgan fingerprints and a random forest classifier.

    Returns
    -------
    Pipeline
        Pipeline with Morgan fingerprints and a random forest classifier.
    """
    pipeline = Pipeline(
        [
            ("smi2mol", SmilesToMol()),
            ("morgan", MolToMorganFP(radius=1, n_bits=1024)),
            (
                "model",
                RandomForestClassifier(n_estimators=2, random_state=_RANDOM_STATE),
            ),
        ]
    )
    return pipeline


class TestExplainabilityVisualization(unittest.TestCase):
    """Test the public interface of the visualization methods for explanations."""

    test_pipeline: ClassVar[Pipeline]
    test_tree_explainer: ClassVar[SHAPTreeExplainer]
    test_tree_explanations: ClassVar[list[SHAPFeatureAndAtomExplanation]]
    test_kernel_explainer: ClassVar[SHAPKernelExplainer]
    test_kernel_explanations: ClassVar[list[SHAPFeatureAndAtomExplanation]]

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the tests."""
        cls.test_pipeline = _get_test_morgan_rf_pipeline()
        cls.test_pipeline.fit(TEST_SMILES, CONTAINS_OX)

        # tree explainer
        cls.test_tree_explainer = SHAPTreeExplainer(cls.test_pipeline)
        cls.test_tree_explanations = cls.test_tree_explainer.explain(TEST_SMILES)

        # kernel explainer
        featurization_subpipeline = get_featurization_subpipeline(
            cls.test_pipeline, raise_not_found=True
        )
        data_transformed = featurization_subpipeline.transform(TEST_SMILES)  # type: ignore[union-attr]
        if scipy.sparse.issparse(data_transformed):
            # convert sparse matrix to dense array because SHAPKernelExplainer
            # does not support sparse matrix as `data` and then explain dense matrices.
            # We stick to dense matrices for simplicity.
            data_transformed = data_transformed.toarray()
        cls.test_kernel_explainer = SHAPKernelExplainer(
            cls.test_pipeline, data=data_transformed
        )
        cls.test_kernel_explanations = cls.test_kernel_explainer.explain(TEST_SMILES)

    def test_structure_heatmap_fingerprint_based_atom_coloring(self) -> None:
        """Test structure heatmap fingerprint-based atom coloring."""
        for explanation_list in [
            self.test_tree_explanations,
            self.test_kernel_explanations,
        ]:
            for explanation in explanation_list:
                self.assertTrue(explanation.is_valid())
                self.assertIsInstance(explanation.atom_weights, np.ndarray)
                image = structure_heatmap(
                    explanation.molecule,
                    explanation.atom_weights,  # type: ignore[arg-type]
                    width=128,
                    height=128,
                )  # type: ignore[union-attr]
                self.assertIsNotNone(image)
                self.assertEqual(image.format, "PNG")

    def test_structure_heatmap_shap_explanation(self) -> None:
        """Test structure heatmap SHAP explanation."""
        for explanation_list in [
            self.test_tree_explanations,
            self.test_kernel_explanations,
        ]:
            for explanation in explanation_list:
                self.assertTrue(explanation.is_valid())
                self.assertIsInstance(explanation.atom_weights, np.ndarray)
                image = structure_heatmap_shap(
                    explanation=explanation,
                    width=128,
                    height=128,
                )  # type: ignore[union-attr]
                self.assertIsNotNone(image)
                self.assertEqual(image.format, "PNG")

    def test_explicit_hydrogens(self) -> None:
        """Test that the visualization methods work with explicit hydrogens."""
        mol_implicit_hydrogens = Chem.MolFromSmiles("C")
        explanations1 = self.test_tree_explainer.explain(
            [Chem.MolToSmiles(mol_implicit_hydrogens)]
        )
        mol_added_hydrogens = Chem.AddHs(mol_implicit_hydrogens)
        explanations2 = self.test_tree_explainer.explain(
            [Chem.MolToSmiles(mol_added_hydrogens)]
        )
        mol_explicit_hydrogens = Chem.MolFromSmiles("[H]C([H])([H])[H]")
        explanations3 = self.test_tree_explainer.explain(
            [Chem.MolToSmiles(mol_explicit_hydrogens)]
        )

        # test explanations' atom weights
        self.assertEqual(len(explanations1), 1)
        self.assertEqual(len(explanations2), 1)
        self.assertEqual(len(explanations3), 1)
        self.assertIsInstance(explanations1[0].atom_weights, np.ndarray)
        self.assertIsInstance(explanations2[0].atom_weights, np.ndarray)
        self.assertIsInstance(explanations3[0].atom_weights, np.ndarray)
        self.assertEqual(len(explanations1[0].atom_weights), 1)  # type: ignore[arg-type]
        self.assertEqual(len(explanations2[0].atom_weights), 1)  # type: ignore[arg-type]
        self.assertEqual(len(explanations3[0].atom_weights), 1)  # type: ignore[arg-type]

        # test visualization
        all_explanations = explanations1 + explanations2 + explanations3
        for explanation in all_explanations:
            self.assertTrue(explanation.is_valid())
            image = structure_heatmap(
                explanation.molecule,
                explanation.atom_weights,  # type: ignore[arg-type]
                width=128,
                height=128,
            )  # type: ignore[union-attr]
            self.assertIsNotNone(image)
            self.assertEqual(image.format, "PNG")
