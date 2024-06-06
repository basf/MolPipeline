"""Test visualization methods for explanations."""

import unittest

from sklearn.ensemble import RandomForestClassifier

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.explainability import SHAPTreeExplainer
from molpipeline.explainability.visualization import rdkit_gaussplot, show_png
from molpipeline.mol2any import MolToMorganFP

TEST_SMILES = ["CC", "CCO", "COC", "c1ccccc1(N)", "CCC(-O)O", "CCCN"]
CONTAINS_OX = [0, 1, 1, 0, 1, 0]

_RANDOM_STATE = 67056


class TestExplainabilityVisualization(unittest.TestCase):
    """Test visualization methods for explanations."""

    def test_test_fingerprint_based_atom_coloring(self) -> None:
        """Test fingerprint-based atom coloring."""

        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("morgan", MolToMorganFP(radius=1, n_bits=1024)),
                ("model", RandomForestClassifier(random_state=_RANDOM_STATE)),
            ]
        )
        pipeline.fit(TEST_SMILES, CONTAINS_OX)

        explainer = SHAPTreeExplainer(pipeline)
        explanations = explainer.explain(TEST_SMILES)

        for explanation in explanations:
            self.assertTrue(explanation.is_valid())
            drawer = rdkit_gaussplot(
                explanation.molecule, explanation.atom_weights.tolist()
            )

            self.assertIsNotNone(drawer)

            figure_bytes = drawer.GetDrawingText()

            image = show_png(figure_bytes)

            self.assertEqual(image.format, "PNG")
