"""Test the custom filter element."""

import unittest

from molpipeline import Pipeline
from molpipeline.any2mol import AutoToMol
from molpipeline.experimental import CustomFilter
from molpipeline.mol2any import MolToBool


class TestCustomFilter(unittest.TestCase):
    """Test the custom filter element."""

    smiles_list = [
        "CC",
        "CCC",
        "CCCC",
        "CO",
    ]

    def test_transform(self) -> None:
        """Test the custom filter."""
        mol_list = AutoToMol().transform(self.smiles_list)
        res_filter = CustomFilter(lambda x: x.GetNumAtoms() == 2).transform(mol_list)
        res_bool = MolToBool().transform(res_filter)
        self.assertEqual(res_bool, [True, False, False, True])

    def test_pipeline(self) -> None:
        """Test the custom filter in pipeline."""
        pipeline = Pipeline(
            [
                ("auto_to_mol", AutoToMol()),
                ("custom_filter", CustomFilter(lambda x: x.GetNumAtoms() == 2)),
                ("mol_to_bool", MolToBool()),
            ]
        )
        self.assertEqual(
            pipeline.transform(self.smiles_list), [True, False, False, True]
        )
