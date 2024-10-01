"""Unittests for testing conversion of various inputs to RDKit molecules."""

import gzip
import unittest

from rdkit import Chem, rdBase

from molpipeline import Pipeline
from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.any2mol import AutoToMol, BinaryToMol, SDFToMol, SmilesToMol
from tests import TEST_DATA_DIR

# pylint: disable=duplicate-code  # test case molecules are allowed to be duplicated
SMILES_ANTIMONY = "[SbH6+3]"
SMILES_BENZENE = "c1ccccc1"
SMILES_CHLOROBENZENE = "Clc1ccccc1"
SMILES_CL_BR = "NC(Cl)(Br)C(=O)O"
SMILES_METAL_AU = "OC[C@H]1OC(S[Au])[C@H](O)[C@@H](O)[C@@H]1O"

INCHI_BENZENE = "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"
INCHI_CHLOROBENZENE = "InChI=1S/C6H5Cl/c7-6-4-2-1-3-5-6/h1-5H"

# SDF
with gzip.open(TEST_DATA_DIR / "P86_B_400.sdf.gz") as file:
    SDF_P86_B_400 = file.read()
_sdf_supplier = Chem.SDMolSupplier()
_sdf_supplier.SetData(SDF_P86_B_400)

# INCHI
INCHI_P86_LIGAND = "InChI=1S/C16H24I2N2O/c1-3-20(4-2)14-5-7-19(8-6-14)11-12-9-13(17)10-15(18)16(12)21/h9-10,14,21H,3-8,11H2,1-2H3"


# RDKit mols
MOL_ANTIMONY = Chem.MolFromSmiles(SMILES_ANTIMONY)
MOL_BENZENE = Chem.MolFromSmiles(SMILES_BENZENE)
MOL_CHLOROBENZENE = Chem.MolFromSmiles(SMILES_CHLOROBENZENE)
MOL_CL_BR = Chem.MolFromSmiles(SMILES_CL_BR)
MOL_METAL_AU = Chem.MolFromSmiles(SMILES_METAL_AU)
MOL_P86_LIGAND = next(_sdf_supplier)


class TestAuto2Mol(unittest.TestCase):
    """Test case for testing conversion of input to molecules."""

    def test_auto2mol_for_smiles(self) -> None:
        """Test molecules can be read from smiles automatically."""

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
                (
                    "Auto2Mol",
                    AutoToMol(
                        elements=(
                            SmilesToMol(),
                            BinaryToMol(),
                            SDFToMol(),
                        )
                    ),
                ),
            ]
        )
        log_block = rdBase.BlockLogs()
        actual_mols = pipeline.fit_transform(test_smiles)
        self.assertEqual(len(test_smiles), len(actual_mols))
        self.assertTrue(
            all(
                Chem.MolToInchi(smiles_mol) == Chem.MolToInchi(original_mol)
                for smiles_mol, original_mol in zip(actual_mols, expected_mols)
            )
        )
        del log_block

    def test_auto2mol_for_inchi(self) -> None:
        """Test molecules can be read from inchi automatically."""

        test_inchis = [INCHI_BENZENE, INCHI_CHLOROBENZENE]
        expected_mols = [MOL_BENZENE, MOL_CHLOROBENZENE]

        pipeline = Pipeline(
            [
                (
                    "Auto2Mol",
                    AutoToMol(),
                ),
            ]
        )
        log_block = rdBase.BlockLogs()
        actual_mols = pipeline.fit_transform(test_inchis)
        self.assertEqual(len(test_inchis), len(actual_mols))
        self.assertTrue(
            all(
                Chem.MolToInchi(smiles_mol) == Chem.MolToInchi(original_mol)
                for smiles_mol, original_mol in zip(actual_mols, expected_mols)
            )
        )
        del log_block

    def test_auto2mol_for_sdf(self) -> None:
        """Test molecules can be read from sdf automatically."""

        pipeline = Pipeline(
            [
                (
                    "Auto2Mol",
                    AutoToMol(
                        elements=(
                            SmilesToMol(),
                            BinaryToMol(),
                            SDFToMol(),
                        )
                    ),
                ),
            ]
        )
        log_block = rdBase.BlockLogs()
        actual_mols = pipeline.fit_transform([SDF_P86_B_400])
        self.assertEqual(len(actual_mols), 1)
        self.assertTrue(
            Chem.MolToInchi(actual_mols[0]).startswith(INCHI_P86_LIGAND),
        )
        del log_block

    def test_auto2mol_for_binary(self) -> None:
        """Test molecules can be read from binary automatically."""

        test_bin_mols = [
            MOL_ANTIMONY.ToBinary(),
            MOL_BENZENE.ToBinary(),
            MOL_CHLOROBENZENE.ToBinary(),
            MOL_CL_BR.ToBinary(),
            MOL_METAL_AU.ToBinary(),
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
                (
                    "Auto2Mol",
                    AutoToMol(
                        elements=(
                            SmilesToMol(),
                            BinaryToMol(),
                            SDFToMol(),
                        )
                    ),
                ),
            ]
        )
        log_block = rdBase.BlockLogs()
        actual_mols = pipeline.fit_transform(test_bin_mols)
        self.assertEqual(len(test_bin_mols), len(actual_mols))
        self.assertTrue(
            all(
                Chem.MolToInchi(smiles_mol) == Chem.MolToInchi(original_mol)
                for smiles_mol, original_mol in zip(actual_mols, expected_mols)
            )
        )
        del log_block

    def test_auto2mol_for_molecule(self) -> None:
        """Test molecules can be read from molecules automatically."""

        test_mols = [
            MOL_ANTIMONY,
            MOL_BENZENE,
            MOL_CHLOROBENZENE,
            MOL_CL_BR,
            MOL_METAL_AU,
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
                (
                    "Auto2Mol",
                    AutoToMol(
                        elements=(
                            SmilesToMol(),
                            BinaryToMol(),
                            SDFToMol(),
                        )
                    ),
                ),
            ]
        )
        log_block = rdBase.BlockLogs()
        actual_mols = pipeline.fit_transform(test_mols)
        self.assertEqual(len(test_mols), len(actual_mols))
        self.assertTrue(
            all(
                Chem.MolToInchi(smiles_mol) == Chem.MolToInchi(original_mol)
                for smiles_mol, original_mol in zip(actual_mols, expected_mols)
            )
        )
        del log_block

    def test_auto2mol_mixed_inputs(self) -> None:
        """Test molecules can be read from mixed inputs automatically."""

        test_inputs = [
            SDF_P86_B_400,
            MOL_ANTIMONY.ToBinary(),
            MOL_BENZENE,
            SMILES_CHLOROBENZENE,
            MOL_CL_BR.ToBinary(),
            SMILES_METAL_AU,
        ]
        test_mols = [
            MOL_P86_LIGAND,
            MOL_ANTIMONY,
            MOL_BENZENE,
            MOL_CHLOROBENZENE,
            MOL_CL_BR,
            MOL_METAL_AU,
        ]

        pipeline = Pipeline(
            [
                (
                    "Auto2Mol",
                    AutoToMol(
                        elements=(
                            SmilesToMol(),
                            BinaryToMol(),
                            SDFToMol(),
                        )
                    ),
                ),
            ]
        )
        log_block = rdBase.BlockLogs()
        actual_mols = pipeline.fit_transform(test_inputs)
        self.assertEqual(len(test_inputs), len(actual_mols))
        self.assertTrue(
            all(
                Chem.MolToInchi(smiles_mol) == Chem.MolToInchi(original_mol)
                for smiles_mol, original_mol in zip(actual_mols, test_mols)
            )
        )
        del log_block

    def test_auto2mol_invalid_input_nones(self) -> None:
        """Test molecules can be read from invalid input with Nones automatically."""

        test_inputs = [
            SDF_P86_B_400,
            None,
            MOL_BENZENE,
            None,
        ]

        pipeline = Pipeline(
            [
                (
                    "Auto2Mol",
                    AutoToMol(
                        elements=(
                            SmilesToMol(),
                            BinaryToMol(),
                            SDFToMol(),
                        )
                    ),
                ),
            ]
        )
        log_block = rdBase.BlockLogs()
        actual_mols = pipeline.fit_transform(test_inputs)
        self.assertEqual(len(test_inputs), len(actual_mols))

        self.assertEqual(
            Chem.MolToInchi(actual_mols[0]), Chem.MolToInchi(MOL_P86_LIGAND)
        )
        self.assertTrue(isinstance(actual_mols[1], InvalidInstance))
        self.assertEqual(Chem.MolToInchi(actual_mols[2]), Chem.MolToInchi(MOL_BENZENE))
        self.assertTrue(isinstance(actual_mols[3], InvalidInstance))
        del log_block

    def test_auto2mol_invalid_input_no_matching_reader(self) -> None:
        """Test molecules can be read from invalid input with no matching readers automatically."""

        test_inputs = [
            SDF_P86_B_400,
            MOL_BENZENE,
            MOL_BENZENE.ToBinary(),
            SMILES_CHLOROBENZENE,
        ]

        pipeline = Pipeline(
            [
                (
                    "Auto2Mol",
                    AutoToMol(elements=(SmilesToMol(),)),
                ),
            ]
        )
        log_block = rdBase.BlockLogs()
        actual_mols = pipeline.fit_transform(test_inputs)
        self.assertEqual(len(test_inputs), len(actual_mols))

        self.assertTrue(isinstance(actual_mols[0], InvalidInstance))
        self.assertEqual(Chem.MolToInchi(actual_mols[1]), Chem.MolToInchi(MOL_BENZENE))
        self.assertTrue(isinstance(actual_mols[2], InvalidInstance))
        self.assertEqual(
            Chem.MolToInchi(actual_mols[3]), Chem.MolToInchi(MOL_CHLOROBENZENE)
        )
        del log_block

    def test_auto2mol_invalid_input_empty_elements(self) -> None:
        """Test molecules can be read from invalid input with no reader elements automatically."""

        test_inputs = [
            SDF_P86_B_400,
            MOL_BENZENE,
            MOL_BENZENE.ToBinary(),
            SMILES_CHLOROBENZENE,
        ]

        pipeline = Pipeline(
            [
                (
                    "Auto2Mol",
                    AutoToMol(elements=()),
                ),
            ]
        )
        log_block = rdBase.BlockLogs()
        actual_mols = pipeline.fit_transform(test_inputs)
        self.assertEqual(len(test_inputs), len(actual_mols))

        self.assertTrue(isinstance(actual_mols[0], InvalidInstance))
        self.assertEqual(Chem.MolToInchi(actual_mols[1]), Chem.MolToInchi(MOL_BENZENE))
        self.assertTrue(isinstance(actual_mols[2], InvalidInstance))
        self.assertTrue(isinstance(actual_mols[3], InvalidInstance))
        del log_block
