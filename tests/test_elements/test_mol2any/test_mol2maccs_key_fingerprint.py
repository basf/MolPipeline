"""Tests for the MolToMACCSFP pipeline element."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import MolToMACCSFP

# pylint: disable=duplicate-code
# Similar to test_mol2morgan_fingerprint.py and test_mol2path_fingerprint.py

test_smiles = [
    "c1ccccc1",
    "c1ccccc1C",
    "NCCOCCCC(=O)O",
]


class TestMolToMACCSFP(unittest.TestCase):
    """Unittest for MolToMACCSFP, which calculates MACCS Key Fingerprints."""

    def test_can_be_constructed(self) -> None:
        """Test if the MolToMACCSFP pipeline element can be constructed.

        Returns
        -------
        None
        """
        mol_fp = MolToMACCSFP()
        mol_fp_copy = mol_fp.copy()
        self.assertTrue(mol_fp_copy is not mol_fp)
        for key, value in mol_fp.get_params().items():
            self.assertEqual(value, mol_fp_copy.get_params()[key])
        mol_fp_recreated = MolToMACCSFP(**mol_fp.get_params())
        for key, value in mol_fp.get_params().items():
            self.assertEqual(value, mol_fp_recreated.get_params()[key])

    def test_output_types(self) -> None:
        """Test equality of different output_types."""

        smi2mol = SmilesToMol()
        sparse_maccs = MolToMACCSFP(return_as="sparse")
        dense_maccs = MolToMACCSFP(return_as="dense")
        explicit_bit_vect_maccs = MolToMACCSFP(return_as="explicit_bit_vect")
        sparse_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("sparse_maccs", sparse_maccs),
            ],
        )
        dense_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("dense_maccs", dense_maccs),
            ],
        )
        explicit_bit_vect_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("explicit_bit_vect_maccs", explicit_bit_vect_maccs),
            ],
        )

        sparse_output = sparse_pipeline.fit_transform(test_smiles)
        dense_output = dense_pipeline.fit_transform(test_smiles)
        explicit_bit_vect_maccs_output = explicit_bit_vect_pipeline.fit_transform(
            test_smiles
        )

        self.assertTrue(np.all(sparse_output.toarray() == dense_output))

        self.assertTrue(
            np.equal(
                dense_output,
                np.array(explicit_bit_vect_maccs_output),
            ).all()
        )

    def test_setter_getter(self) -> None:
        """Test if the setters and getters work as expected."""
        mol_fp = MolToMACCSFP()
        params: dict[str, Any] = {
            "return_as": "dense",
        }
        mol_fp.set_params(**params)
        self.assertEqual(mol_fp.get_params()["return_as"], "dense")

    def test_setter_getter_error_handling(self) -> None:
        """Test if the setters and getters work as expected when errors are encountered."""

        mol_fp = MolToMACCSFP()
        params: dict[str, Any] = {
            "return_as": "invalid-option",
        }
        self.assertRaises(ValueError, mol_fp.set_params, **params)

    def test_feature_names(self) -> None:
        """Test if the feature names are correct."""
        mol_fp = MolToMACCSFP()
        feature_names = mol_fp.feature_names
        self.assertEqual(len(feature_names), mol_fp.n_bits)
        # feature names should be unique
        self.assertEqual(len(feature_names), len(set(feature_names)))


if __name__ == "__main__":
    unittest.main()
