"""Unittests for testing conversion of molecules to binary string."""
import unittest

from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.mol2any.mol2bin import MolToBinaryPipelineElement
from rdkit import Chem, rdBase

# pylint: disable=duplicate-code  # test case molecules are allowed to be duplicated
SMILES_ANTIMONY = "[SbH6+3]"
SMILES_BENZENE = "c1ccccc1"
SMILES_CHLOROBENZENE = "Clc1ccccc1"
SMILES_CL_BR = "NC(Cl)(Br)C(=O)O"
SMILES_METAL_AU = "OC[C@H]1OC(S[Au])[C@H](O)[C@@H](O)[C@@H]1O"

MOL_ANTIMONY = Chem.MolFromSmiles(SMILES_ANTIMONY)
MOL_BENZENE = Chem.MolFromSmiles(SMILES_BENZENE)
MOL_CHLOROBENZENE = Chem.MolFromSmiles(SMILES_CHLOROBENZENE)
MOL_CL_BR = Chem.MolFromSmiles(SMILES_CL_BR)
MOL_METAL_AU = Chem.MolFromSmiles(SMILES_METAL_AU)


class TestMol2Binary(unittest.TestCase):
    """Test case for testing conversion of molecules to binary string representation."""

    def test_mol_to_binary(self) -> None:
        """Test if smiles converted correctly to binary string."""

        test_smiles = [
            SMILES_ANTIMONY,
            SMILES_BENZENE,
            SMILES_CHLOROBENZENE,
            SMILES_CL_BR,
            SMILES_METAL_AU,
        ]
        expected_mols = [
            MOL_ANTIMONY,
            MOL_BENZENE,
            MOL_CHLOROBENZENE,
            MOL_CL_BR,
            MOL_METAL_AU,
        ]

        pipeline = Pipeline(
            [
                ("Smiles2Mol", SmilesToMolPipelineElement()),
                ("Mol2Binary", MolToBinaryPipelineElement()),
            ]
        )
        log_block = rdBase.BlockLogs()
        binary_mols = pipeline.fit_transform(test_smiles)
        self.assertEqual(len(test_smiles), len(binary_mols))
        actual_mols = [Chem.Mol(mol) for mol in binary_mols]
        self.assertTrue(
            all(
                Chem.MolToInchi(smiles_mol) == Chem.MolToInchi(original_mol)
                for smiles_mol, original_mol in zip(actual_mols, expected_mols)
            )
        )
        del log_block

    def test_mol_to_binary_invalid_input(self) -> None:
        """Test how invalid input is handled."""

        pipeline = Pipeline(
            [
                ("Mol2Binary", MolToBinaryPipelineElement()),
            ]
        )

        # test empty molecule
        binary_mols = pipeline.fit_transform([Chem.MolFromSmiles("")])
        self.assertEqual(len(binary_mols), 1)
        self.assertEqual(Chem.MolToSmiles(Chem.Mol(binary_mols[0])), "")

        # test None as input
        self.assertRaises(AttributeError, pipeline.fit_transform, [None])
