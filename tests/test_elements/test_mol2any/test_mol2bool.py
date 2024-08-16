"""Test mol to bool conversion."""

import unittest

from molpipeline import Pipeline
from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.any2mol import AutoToMol
from molpipeline.mol2any import MolToBool


class TestMolToBool(unittest.TestCase):
    """Unittest for MolToBool."""

    def test_bool_conversion(self) -> None:
        """Test if the invalid instances are converted to bool."""
        mol2bool = MolToBool()
        result = mol2bool.transform(
            [
                1,
                2,
                InvalidInstance(element_id="test", message="test", element_name="Test"),
                4,
            ]
        )
        self.assertEqual(result, [True, True, False, True])

    def test_bool_conversion_pipeline(self) -> None:
        """Test if the invalid instances are converted to bool in pipeline."""
        pipeline = Pipeline(
            [
                ("auto_to_mol", AutoToMol()),
                ("mol2bool", MolToBool()),
            ]
        )
        result = pipeline.transform(["CC", "CCC", "no%valid~smiles"])
        self.assertEqual(result, [True, True, False])
