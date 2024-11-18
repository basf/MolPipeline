"""Test smiles to mol pipeline element."""

import unittest
from typing import Any

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol


class TestSmiles2Mol(unittest.TestCase):
    """Test case for testing conversion of SMILES input to molecules."""

    def test_smiles2mol_explict_hydrogens(self) -> None:
        """Test smiles reading with and without explicit smiles."""
        smiles = "C[H]"

        # test: remove explicit Hs
        pipeline = Pipeline(
            [
                (
                    "Smiles2Mol",
                    SmilesToMol(remove_hydrogens=True),
                ),
            ]
        )
        mols = pipeline.fit_transform([smiles])
        self.assertEqual(len(mols), 1)
        self.assertIsNotNone(mols[0])
        self.assertEqual(mols[0].GetNumAtoms(), 1)

        # test: keep explicit Hs
        pipeline2 = Pipeline(
            [
                (
                    "Smiles2Mol",
                    SmilesToMol(remove_hydrogens=False),
                ),
            ]
        )
        mols2 = pipeline2.fit_transform([smiles])
        self.assertEqual(len(mols2), 1)
        self.assertIsNotNone(mols2[0])
        self.assertEqual(mols2[0].GetNumAtoms(), 2)

    def test_getter_setter(self) -> None:
        """Test getter and setter methods."""
        smiles2mol = SmilesToMol(remove_hydrogens=False)
        self.assertEqual(smiles2mol.get_params()["remove_hydrogens"], False)
        params: dict[str, Any] = {
            "remove_hydrogens": True,
        }
        smiles2mol.set_params(**params)
        self.assertEqual(smiles2mol.get_params()["remove_hydrogens"], True)
