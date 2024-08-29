"""Test visualization methods for explanations."""

import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.ensemble import RandomForestClassifier

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.explainability import SHAPTreeExplainer, Explanation
from molpipeline.explainability.visualization.visualization import (
    structure_heatmap,
    show_png,
    make_sum_of_gaussians_grid,
)
from molpipeline.mol2any import MolToMorganFP

TEST_SMILES = ["CC", "CCO", "COC", "c1ccccc1(N)", "CCC(-O)O", "CCCN"]
CONTAINS_OX = [0, 1, 1, 0, 1, 0]

_RANDOM_STATE = 67056


def _get_test_explanations() -> list[Explanation]:
    """Get test explanations."""
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
    pipeline.fit(TEST_SMILES, CONTAINS_OX)

    explainer = SHAPTreeExplainer(pipeline)
    explanations = explainer.explain(TEST_SMILES)
    return explanations


class TestExplainabilityVisualization(unittest.TestCase):
    """Test the public interface of the visualization methods for explanations."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the tests."""
        cls.explanations = _get_test_explanations()

    def test_fingerprint_based_atom_coloring(self) -> None:
        """Test fingerprint-based atom coloring."""

        for explanation in self.explanations:
            self.assertTrue(explanation.is_valid())
            self.assertIsInstance(explanation.atom_weights, np.ndarray)
            drawer = structure_heatmap(
                explanation.molecule,
                explanation.atom_weights,  # type: ignore[arg-type]
                width=128,
                height=128,
            )  # type: ignore[union-attr]
            self.assertIsNotNone(drawer)
            figure_bytes = drawer.GetDrawingText()
            image = show_png(figure_bytes)
            self.assertEqual(image.format, "PNG")


class TestSumOfGaussiansGrid(unittest.TestCase):
    """Test visualization methods for explanations."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the tests."""
        cls.explanations = _get_test_explanations()

    def test_grid_with_shap_atom_weights(self) -> None:
        """Test grid with SHAP atom weights."""

        for explanation in self.explanations:
            self.assertTrue(explanation.is_valid())
            self.assertIsInstance(explanation.atom_weights, np.ndarray)

            mol_copy = Chem.Mol(explanation.molecule)
            mol_copy = Draw.PrepareMolForDrawing(mol_copy)
            value_grid = make_sum_of_gaussians_grid(
                mol_copy,
                atom_weights=explanation.atom_weights,
                atom_width=np.inf,
                grid_resolution=[64, 64],
                padding=[0.4, 0.4],
            )
            self.assertIsNotNone(value_grid)
            self.assertEqual(value_grid.values.size, 64 * 64)

            # test that the range of summed gaussian values is as expected for SHAP
            self.assertTrue(value_grid.values.min() >= -1)
            self.assertTrue(value_grid.values.max() <= 1)

    # def test_color_limits(self) -> None:
    #     """Test color limits."""
    #
    #     for explanation in self.explanations:
    #         self.assertTrue(explanation.is_valid())
    #         self.assertIsInstance(explanation.atom_weights, np.ndarray)
    #         drawer = structure_heatmap(
    #             explanation.molecule,
    #             explanation.atom_weights,  # type: ignore[arg-type]
    #             width=128,
    #             height=128,
    #             color_limits=(-1, 1),
    #         )
    #         self.assertIsNotNone(drawer)
    #         figure_bytes = drawer.GetDrawingText()
    #         image = show_png(figure_bytes)
    #         self.assertEqual(image.format, "PNG")
