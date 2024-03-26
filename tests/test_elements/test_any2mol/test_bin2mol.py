"""Unittests for testing conversion of binary string to RDKit molecules."""

import unittest

from rdkit import Chem, rdBase

from molpipeline.any2mol import BinaryToMol

# pylint: disable=duplicate-code  # test case molecules are allowed to be duplicated
SMILES_ANTIMONY = "[SbH6+3]"
SMILES_BENZENE = "c1ccccc1"
SMILES_CHLOROBENZENE = "Clc1ccccc1"
SMILES_CL_BR = "NC(Cl)(Br)C(=O)O"
SMILES_METAL_AU = "OC[C@H]1OC(S[Au])[C@H](O)[C@@H](O)[C@@H]1O"

# RDKit mols
MOL_ANTIMONY = Chem.MolFromSmiles(SMILES_ANTIMONY)
MOL_BENZENE = Chem.MolFromSmiles(SMILES_BENZENE)
MOL_CHLOROBENZENE = Chem.MolFromSmiles(SMILES_CHLOROBENZENE)
MOL_CL_BR = Chem.MolFromSmiles(SMILES_CL_BR)
MOL_METAL_AU = Chem.MolFromSmiles(SMILES_METAL_AU)


class TestBin2Mol(unittest.TestCase):
    """Test case for testing conversion of binary string to molecules."""

    def test_bin2mol(self) -> None:
        """Test molecules can be read from binary string."""
        test_mols = [
            MOL_ANTIMONY,
            MOL_BENZENE,
            MOL_CHLOROBENZENE,
            MOL_CL_BR,
            MOL_METAL_AU,
        ]
        for mol in test_mols:
            bin2mol = BinaryToMol()
            mol = bin2mol.pretransform_single(mol.ToBinary())
            log_block = rdBase.BlockLogs()
            self.assertEqual(Chem.MolToInchi(mol), Chem.MolToInchi(mol))
            del log_block
