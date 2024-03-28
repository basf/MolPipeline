"""Unittests for testing conversion of molecules to InChI and InChIKey."""

import unittest

from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import MolToInchi, MolToInchiKey
from molpipeline.pipeline import Pipeline

# pylint: disable=duplicate-code  # test case molecules are allowed to be duplicated
SMILES_ANTIMONY = "[SbH6+3]"
SMILES_BENZENE = "c1ccccc1"
SMILES_CHLOROBENZENE = "Clc1ccccc1"
SMILES_CL_BR = "NC(Cl)(Br)C(=O)O"
SMILES_METAL_AU = "OC[C@H]1OC(S[Au])[C@H](O)[C@@H](O)[C@@H]1O"


class TestMol2Inchi(unittest.TestCase):
    """Test case for testing conversion of molecules to InChI and InChIKey."""

    def test_to_inchi(self) -> None:
        """Test if smiles converted correctly to inchi string.

        Returns
        -------
        None
        """

        input_smiles = ["CN(C)CCOC(C1=CC=CC=C1)C1=CC=CC=C1"]
        expected_inchis = [
            "InChI=1S/C17H21NO/c1-18(2)13-14-19-17(15-9-5-3-6-10-15)16-11-7-4-8-12-16/h3-12,17H,13-14H2,1-2H3"
        ]
        pipeline = Pipeline(
            [
                ("Smiles2Mol", SmilesToMol()),
                ("Mol2Inchi", MolToInchi()),
            ]
        )
        actual_inchis = pipeline.fit_transform(input_smiles)
        self.assertEqual(expected_inchis, actual_inchis)

    def test_to_inchikey(self) -> None:
        """Test if smiles is converted correctly to inchikey string.

        Returns
        -------
        None
        """

        input_smiles = ["CN(C)CCOC(C1=CC=CC=C1)C1=CC=CC=C1"]
        expected_inchikeys = ["ZZVUWRFHKOJYTH-UHFFFAOYSA-N"]

        pipeline = Pipeline(
            [
                ("Smiles2Mol", SmilesToMol()),
                ("Mol2Inchi", MolToInchiKey()),
            ],
        )
        actual_inchikeys = pipeline.fit_transform(input_smiles)
        self.assertEqual(expected_inchikeys, actual_inchikeys)


if __name__ == "__main__":
    unittest.main()
