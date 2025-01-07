"""Tests for the MolToFoldedMorganFingerprint pipeline element."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import MolToMorganFP
from tests.utils.fingerprints import fingerprints_to_numpy

test_smiles = [
    "c1ccccc1",
    "c1ccccc1C",
    "NCCOCCCC(=O)O",
]


class TestMol2MorganFingerprint(unittest.TestCase):
    """Unittest for MolToFoldedMorganFingerprint, which calculates folded Morgan Fingerprints."""

    def test_can_be_constructed(self) -> None:
        """Test if the MolToFoldedMorganFingerprint pipeline element can be constructed.

        Returns
        -------
        None
        """
        mol_fp = MolToMorganFP()
        mol_fp_copy = mol_fp.copy()
        self.assertTrue(mol_fp_copy is not mol_fp)
        for key, value in mol_fp.get_params().items():
            self.assertEqual(value, mol_fp_copy.get_params()[key])
        mol_fp_recreated = MolToMorganFP(**mol_fp.get_params())
        for key, value in mol_fp.get_params().items():
            self.assertEqual(value, mol_fp_recreated.get_params()[key])

    def test_counted_bits(self) -> None:
        """Test if the option counted bits works as expected.

        Returns
        -------
        None
        """
        mol_fp = MolToMorganFP(radius=2, n_bits=1024, return_as="dense")
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

    def test_output_types(self) -> None:
        """Test equality of different output_types."""

        smi2mol = SmilesToMol()
        sparse_morgan = MolToMorganFP(radius=2, n_bits=1024, return_as="sparse")
        dense_morgan = MolToMorganFP(radius=2, n_bits=1024, return_as="dense")
        explicit_bit_vect_morgan = MolToMorganFP(
            radius=2, n_bits=1024, return_as="explicit_bit_vect"
        )
        sparse_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("sparse_morgan", sparse_morgan),
            ],
        )
        dense_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("dense_morgan", dense_morgan),
            ],
        )
        explicit_bit_vect_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("explicit_bit_vect_morgan", explicit_bit_vect_morgan),
            ],
        )

        sparse_output = sparse_pipeline.fit_transform(test_smiles)
        dense_output = dense_pipeline.fit_transform(test_smiles)
        explicit_bit_vect_morgan_output = explicit_bit_vect_pipeline.fit_transform(
            test_smiles
        )

        self.assertTrue(np.all(sparse_output.toarray() == dense_output))

        self.assertTrue(
            np.equal(
                dense_output,
                np.array(explicit_bit_vect_morgan_output),
            ).all()
        )

    def test_setter_getter(self) -> None:
        """Test if the setters and getters work as expected."""
        mol_fp = MolToMorganFP()
        params: dict[str, Any] = {
            "radius": 2,
            "n_bits": 1024,
            "return_as": "dense",
        }
        mol_fp.set_params(**params)
        self.assertEqual(mol_fp.get_params()["radius"], 2)
        self.assertEqual(mol_fp.get_params()["n_bits"], 1024)
        self.assertEqual(mol_fp.get_params()["return_as"], "dense")

    def test_setter_getter_error_handling(self) -> None:
        """Test if the setters and getters work as expected when errors are encountered."""

        mol_fp = MolToMorganFP()
        params: dict[str, Any] = {
            "radius": 2,
            "n_bits": 1024,
            "return_as": "invalid-option__11!",
        }
        self.assertRaises(ValueError, mol_fp.set_params, **params)

    def test_bit2atom_mapping(self) -> None:
        """Test that the mapping from bits to atom weights works as intended."""
        n_bits = 2048
        sparse_morgan = MolToMorganFP(radius=2, n_bits=n_bits, return_as="sparse")
        dense_morgan = MolToMorganFP(radius=2, n_bits=n_bits, return_as="dense")
        explicit_bit_vect_morgan = MolToMorganFP(
            radius=2, n_bits=n_bits, return_as="explicit_bit_vect"
        )

        smi2mol = SmilesToMol()
        for test_smi in test_smiles:
            for fp_gen in [sparse_morgan, dense_morgan, explicit_bit_vect_morgan]:
                for counted in [False, True]:
                    mol = smi2mol.transform([test_smi])[0]
                    fp_gen.set_params(counted=counted)
                    fp = fp_gen.transform([mol])
                    mapping = fp_gen.bit2atom_mapping(mol)
                    np_fp = fingerprints_to_numpy(fp)
                    self.assertEqual(np.nonzero(np_fp)[0].shape[0], len(mapping))  # type: ignore

    def test_feature_names(self) -> None:
        """Test if the feature names are correct."""
        mol_fp = MolToMorganFP(n_bits=1024)
        feature_names = mol_fp.feature_names
        self.assertEqual(len(feature_names), 1024)
        # feature names should be unique
        self.assertEqual(len(feature_names), len(set(feature_names)))


if __name__ == "__main__":
    unittest.main()
