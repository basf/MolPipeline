"""Test visualization methods for explanations."""

import unittest
from typing import ClassVar

import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.experimental.explainability import (
    SHAPFeatureAndAtomExplanation,
    SHAPFeatureExplanation,
    SHAPKernelExplainer,
    SHAPTreeExplainer,
    structure_heatmap,
    structure_heatmap_shap,
)
from molpipeline.mol2any import MolToMorganFP
from tests.test_experimental.test_explainability.utils import (
    construct_kernel_shap_kwargs,
)

TEST_SMILES = ["CC", "CCO", "COC", "c1ccccc1(N)", "CCC(-O)O", "CCCN"]
CONTAINS_OX = [0, 1, 1, 0, 1, 0]  # classification labels
REGRESSION_LABELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # regression labels


_RANDOM_STATE = 67056


def _get_test_morgan_rf_pipeline(task: str = "classification") -> Pipeline:
    """Get a test pipeline with Morgan fingerprints and a random forest classifier.

    Parameters
    ----------
    task : str, optional (default="classification")
        Task of the pipeline. Either "classification" or "regression".

    Returns
    -------
    Pipeline
        Pipeline with Morgan fingerprints and a random forest classifier.
    """

    if task == "classification":
        model = RandomForestClassifier(n_estimators=2, random_state=_RANDOM_STATE)
    elif task == "regression":
        model = RandomForestRegressor(n_estimators=2, random_state=_RANDOM_STATE)
    else:
        raise ValueError(f"Invalid task: {task}")

    pipeline = Pipeline(
        [
            ("smi2mol", SmilesToMol()),
            ("morgan", MolToMorganFP(radius=1, n_bits=1024)),
            (
                "model",
                model,
            ),
        ]
    )
    return pipeline


class TestExplainabilityVisualization(unittest.TestCase):
    """Test the public interface of the visualization methods for explanations."""

    test_pipeline_clf: ClassVar[Pipeline]
    test_tree_explainer_clf: ClassVar[SHAPTreeExplainer]
    test_tree_explanations_clf: ClassVar[
        list[SHAPFeatureAndAtomExplanation | SHAPFeatureExplanation]
    ]
    test_kernel_explainer_clf: ClassVar[SHAPKernelExplainer]
    test_kernel_explanations_clf: ClassVar[
        list[SHAPFeatureAndAtomExplanation | SHAPFeatureExplanation]
    ]

    test_pipeline_reg: ClassVar[Pipeline]
    test_tree_explainer_reg: ClassVar[SHAPTreeExplainer]
    test_tree_explanations_reg: ClassVar[
        list[SHAPFeatureAndAtomExplanation | SHAPFeatureExplanation]
    ]
    test_kernel_explainer_reg: ClassVar[SHAPKernelExplainer]
    test_kernel_explanations_reg: ClassVar[
        list[SHAPFeatureAndAtomExplanation | SHAPFeatureExplanation]
    ]

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the tests."""
        # test pipeline for classification
        cls.test_pipeline_clf = _get_test_morgan_rf_pipeline(task="classification")
        cls.test_pipeline_clf.fit(TEST_SMILES, CONTAINS_OX)

        # test pipeline for regression
        cls.test_pipeline_reg = _get_test_morgan_rf_pipeline(task="regression")
        cls.test_pipeline_reg.fit(TEST_SMILES, REGRESSION_LABELS)

        # tree explainer for classification
        cls.test_tree_explainer_clf = SHAPTreeExplainer(cls.test_pipeline_clf)
        cls.test_tree_explanations_clf = cls.test_tree_explainer_clf.explain(
            TEST_SMILES,
        )

        # tree explainer for regression
        cls.test_tree_explainer_reg = SHAPTreeExplainer(cls.test_pipeline_reg)
        cls.test_tree_explanations_reg = cls.test_tree_explainer_reg.explain(
            TEST_SMILES
        )

        # kernel explainer for classification
        kernel_kwargs_clf = construct_kernel_shap_kwargs(
            cls.test_pipeline_clf, TEST_SMILES
        )
        cls.test_kernel_explainer_clf = SHAPKernelExplainer(
            cls.test_pipeline_clf, **kernel_kwargs_clf
        )
        cls.test_kernel_explanations_clf = cls.test_kernel_explainer_clf.explain(
            TEST_SMILES
        )

        # kernel explainer for regression
        kernel_kwargs_reg = construct_kernel_shap_kwargs(
            cls.test_pipeline_reg, TEST_SMILES
        )
        cls.test_kernel_explainer_reg = SHAPKernelExplainer(
            cls.test_pipeline_reg, **kernel_kwargs_reg
        )
        cls.test_kernel_explanations_reg = cls.test_kernel_explainer_reg.explain(
            TEST_SMILES
        )

    def test_structure_heatmap_fingerprint_based_atom_coloring(self) -> None:
        """Test structure heatmap fingerprint-based atom coloring."""
        for explanation_list in [
            self.test_tree_explanations_clf,
            self.test_kernel_explanations_clf,
            self.test_tree_explanations_reg,
            self.test_kernel_explanations_reg,
        ]:
            for explanation in explanation_list:
                self.assertTrue(explanation.is_valid())
                self.assertIsInstance(explanation.atom_weights, np.ndarray)  # type: ignore[union-attr]
                image = structure_heatmap(
                    explanation.molecule,
                    explanation.atom_weights,  # type: ignore
                    width=8,
                    height=8,
                )  # type: ignore[union-attr]
                self.assertIsNotNone(image)
                self.assertEqual(image.format, "PNG")

    def test_structure_heatmap_shap_explanation(self) -> None:
        """Test structure heatmap SHAP explanation."""
        for explanation_list in [
            self.test_tree_explanations_clf,
            self.test_kernel_explanations_clf,
            self.test_tree_explanations_reg,
            self.test_kernel_explanations_reg,
        ]:
            for explanation in explanation_list:
                self.assertTrue(explanation.is_valid())
                self.assertIsInstance(explanation, SHAPFeatureAndAtomExplanation)
                self.assertIsInstance(explanation.atom_weights, np.ndarray)  # type: ignore[union-attr]
                image = structure_heatmap_shap(
                    explanation=explanation,  # type: ignore[arg-type]
                    width=8,
                    height=8,
                )  # type: ignore[union-attr]
                self.assertIsNotNone(image)
                self.assertEqual(image.format, "PNG")

    def test_explicit_hydrogens(self) -> None:
        """Test that the visualization methods work with explicit hydrogens."""
        mol_implicit_hydrogens = Chem.MolFromSmiles("C")
        explanations1 = self.test_tree_explainer_clf.explain(
            [Chem.MolToSmiles(mol_implicit_hydrogens)]
        )
        mol_added_hydrogens = Chem.AddHs(mol_implicit_hydrogens)
        explanations2 = self.test_tree_explainer_clf.explain(
            [Chem.MolToSmiles(mol_added_hydrogens)]
        )
        mol_explicit_hydrogens = Chem.MolFromSmiles("[H]C([H])([H])[H]")
        explanations3 = self.test_tree_explainer_clf.explain(
            [Chem.MolToSmiles(mol_explicit_hydrogens)]
        )

        # test explanations' atom weights
        self.assertEqual(len(explanations1), 1)
        self.assertEqual(len(explanations2), 1)
        self.assertEqual(len(explanations3), 1)
        self.assertTrue(hasattr(explanations1[0], "atom_weights"))
        self.assertTrue(hasattr(explanations2[0], "atom_weights"))
        self.assertTrue(hasattr(explanations3[0], "atom_weights"))
        self.assertIsInstance(explanations1[0].atom_weights, np.ndarray)  # type: ignore[union-attr]
        self.assertIsInstance(explanations2[0].atom_weights, np.ndarray)  # type: ignore[union-attr]
        self.assertIsInstance(explanations3[0].atom_weights, np.ndarray)  # type: ignore[union-attr]
        self.assertEqual(len(explanations1[0].atom_weights), 1)  # type: ignore
        self.assertEqual(len(explanations2[0].atom_weights), 1)  # type: ignore
        self.assertEqual(len(explanations3[0].atom_weights), 1)  # type: ignore

        # test visualization
        all_explanations = explanations1 + explanations2 + explanations3
        for explanation in all_explanations:
            self.assertTrue(explanation.is_valid())
            image = structure_heatmap(
                explanation.molecule,
                explanation.atom_weights,  # type: ignore
                width=8,
                height=8,
            )  # type: ignore[union-attr]
            self.assertIsNotNone(image)
            self.assertEqual(image.format, "PNG")
