"""Tests for the MolToPathFingerprint pipeline element."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import Mol2PathFP

# pylint: disable=duplicate-code

test_smiles = [
    "c1ccccc1",
    "c1ccccc1C",
    "NCCOCCCC(=O)O",
]


class TestMol2PathFingerprint(unittest.TestCase):
    """Unittest for Mol2PathFP, which calculates the RDKit Path Fingerprint."""

    def test_can_be_constructed(self) -> None:
        """Test if the Mol2PathFP pipeline element can be constructed.

        Returns
        -------
        None
        """
        mol_fp = Mol2PathFP()
        mol_fp_copy = mol_fp.copy()
        self.assertTrue(mol_fp_copy is not mol_fp)
        for key, value in mol_fp.get_params().items():
            self.assertEqual(value, mol_fp_copy.get_params()[key])
        mol_fp_recreated = Mol2PathFP(**mol_fp.get_params())
        for key, value in mol_fp.get_params().items():
            self.assertEqual(value, mol_fp_recreated.get_params()[key])

    def test_output_types(self) -> None:
        """Test equality of different output_types."""

        smi2mol = SmilesToMol()
        sparse_path_fp = Mol2PathFP(n_bits=1024, return_as="sparse")
        dense_path_fp = Mol2PathFP(n_bits=1024, return_as="dense")
        explicit_bit_vect_path_fp = Mol2PathFP(
            n_bits=1024, return_as="explicit_bit_vect"
        )
        sparse_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("sparse_path_fp", sparse_path_fp),
            ],
        )
        dense_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("dense_path_fp", dense_path_fp),
            ],
        )
        explicit_bit_vect_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("explicit_bit_vect_path_fp", explicit_bit_vect_path_fp),
            ],
        )

        sparse_output = sparse_pipeline.fit_transform(test_smiles)
        dense_output = dense_pipeline.fit_transform(test_smiles)
        explicit_bit_vect_path_fp_output = explicit_bit_vect_pipeline.fit_transform(
            test_smiles
        )

        self.assertTrue(np.all(sparse_output.toarray() == dense_output))

        self.assertTrue(
            np.equal(
                dense_output,
                np.array(explicit_bit_vect_path_fp_output),
            ).all()
        )

    def test_counted_bits(self) -> None:
        """Test if the option counted bits works as expected.

        Returns
        -------
        None
        """
        mol_fp = Mol2PathFP(n_bits=1024, return_as="dense")
        smi2mol = SmilesToMol()
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("mol_fp", mol_fp),
            ],
        )
        output_binary = pipeline.fit_transform(test_smiles)
        pipeline.set_params(mol_fp__counted=True)
        output_counted = pipeline.fit_transform(test_smiles)
        self.assertTrue(
            np.all(np.flatnonzero(output_counted) == np.flatnonzero(output_binary))
        )
        self.assertTrue(np.all(output_counted >= output_binary))
        self.assertTrue(np.any(output_counted > output_binary))

    def test_setter_getter(self) -> None:
        """Test if the setters and getters work as expected."""
        mol_fp = Mol2PathFP()
        params: dict[str, Any] = {
            "min_path": 10,
            "max_path": 12,
            "use_hs": False,
            "branched_paths": False,
            "use_bond_order": False,
            "count_simulation": True,
            "num_bits_per_feature": 4,
            "counted": True,
            "n_bits": 1024,
        }
        mol_fp.set_params(**params)
        self.assertEqual(mol_fp.get_params()["min_path"], 10)
        self.assertEqual(mol_fp.get_params()["max_path"], 12)
        self.assertEqual(mol_fp.get_params()["use_hs"], False)
        self.assertEqual(mol_fp.get_params()["branched_paths"], False)
        self.assertEqual(mol_fp.get_params()["use_bond_order"], False)
        self.assertEqual(mol_fp.get_params()["count_simulation"], True)
        self.assertEqual(mol_fp.get_params()["num_bits_per_feature"], 4)
        self.assertEqual(mol_fp.get_params()["counted"], True)
        self.assertEqual(mol_fp.get_params()["n_bits"], 1024)

    def test_setter_getter_error_handling(self) -> None:
        """Test if the setters and getters work as expected when errors are encountered."""

        mol_fp = Mol2PathFP()
        params: dict[str, Any] = {
            "min_path": 2,
            "n_bits": 1024,
            "return_as": "invalid-option",
        }
        self.assertRaises(ValueError, mol_fp.set_params, **params)

    def test_feature_names(self) -> None:
        """Test if the feature names are correct."""
        mol_fp = Mol2PathFP(n_bits=1024)
        feature_names = mol_fp.feature_names
        self.assertEqual(len(feature_names), 1024)
        # feature names should be unique
        self.assertEqual(len(feature_names), len(set(feature_names)))


if __name__ == "__main__":
    unittest.main()
