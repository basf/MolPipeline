"""Test the custom filter element."""

import unittest

from rdkit import Chem

from molpipeline import Pipeline
from molpipeline.experimental import CustomFilter
from molpipeline.mol2any import MolToBool


class TestCustomFilter(unittest.TestCase):
    """Test the custom filter element."""

    def test_transform(self) -> None:
        """Test the custom filter."""
        mol_list = [
            Chem.MolFromSmiles("CC"),
            Chem.MolFromSmiles("CCC"),
            Chem.MolFromSmiles("CCCC"),
            Chem.MolFromSmiles("CO"),
        ]
        res_filter = CustomFilter(lambda x: x.GetNumAtoms() == 2).transform(mol_list)
        res_bool = MolToBool().transform(res_filter)
        self.assertEqual(res_bool, [True, False, False, True])

        pipeline = Pipeline(
            [
                ("custom_filter", CustomFilter(lambda x: x.GetNumAtoms() == 2)),
                ("mol_to_bool", MolToBool()),
            ]
        )
        self.assertEqual(pipeline.transform(mol_list), [True, False, False, True])
