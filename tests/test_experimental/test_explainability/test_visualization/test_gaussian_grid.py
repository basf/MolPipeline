"""Test gaussian grid visualization."""

import unittest
from typing import ClassVar

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from molpipeline import Pipeline
from molpipeline.experimental.explainability import (
    SHAPFeatureAndAtomExplanation,
    SHAPFeatureExplanation,
    SHAPTreeExplainer,
)
from molpipeline.experimental.explainability.visualization.visualization import (
    make_sum_of_gaussians_grid,
)
from tests.test_experimental.test_explainability.test_visualization.test_visualization import (
    _get_test_morgan_rf_pipeline,
)

TEST_SMILES = ["CC", "CCO", "COC", "c1ccccc1(N)", "CCC(-O)O", "CCCN"]
CONTAINS_OX = [0, 1, 1, 0, 1, 0]


class TestSumOfGaussiansGrid(unittest.TestCase):
    """Test sum of gaussian grid ."""

    # pylint: disable=duplicate-code
    test_pipeline: ClassVar[Pipeline]
    test_explainer: ClassVar[SHAPTreeExplainer]
    test_explanations: ClassVar[
        list[SHAPFeatureAndAtomExplanation | SHAPFeatureExplanation]
    ]

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the tests."""
        cls.test_pipeline = _get_test_morgan_rf_pipeline()
        cls.test_pipeline.fit(TEST_SMILES, CONTAINS_OX)
        cls.test_explainer = SHAPTreeExplainer(cls.test_pipeline)
        cls.test_explanations = cls.test_explainer.explain(TEST_SMILES)

    def test_grid_with_shap_atom_weights(self) -> None:
        """Test grid with SHAP atom weights."""
        for explanation in self.test_explanations:
            self.assertTrue(explanation.is_valid())
            self.assertIsInstance(explanation.atom_weights, np.ndarray)  # type: ignore[union-attr]

            mol_copy = Chem.Mol(explanation.molecule)
            mol_copy = Draw.PrepareMolForDrawing(mol_copy)
            value_grid = make_sum_of_gaussians_grid(
                mol_copy,
                atom_weights=explanation.atom_weights,  # type: ignore[union-attr]
                atom_width=np.inf,
                grid_resolution=[8, 8],
                padding=[0.4, 0.4],
            )
            self.assertIsNotNone(value_grid)
            self.assertEqual(value_grid.values.size, 8 * 8)

            # test that the range of summed gaussian values is as expected for SHAP
            self.assertTrue(value_grid.values.min() >= -1)
            self.assertTrue(value_grid.values.max() <= 1)
