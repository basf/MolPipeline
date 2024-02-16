"""Tests for the MolToFoldedMorganFingerprint pipeline element."""

from __future__ import annotations

import unittest

import numpy as np
import numpy.typing as npt
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect

from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.mol2any.mol2morgan_fingerprint import (
    MolToFoldedMorganFingerprint,
)

test_smiles = [
    "c1ccccc1",
    "c1ccccc1C",
    "NCCOCCCC(=O)O",
]


def _explicit_bit_vect_list_to_numpy(
    explicit_bit_vect_list: list[ExplicitBitVect],
) -> npt.NDArray[np.int_]:
    """Convert explicitBitVect manually to numpy

    It is assumed all fingerprints in the list have the same length.

    Parameters
    ----------
    explicit_bit_vect_list: list[ExplicitBitVect]
        List of fingerprints

    Returns
    -------
    npt.NDArray
        Numpy fingerprint matrix.
    """
    if len(explicit_bit_vect_list) == 0:
        return np.empty(
            (
                0,
                0,
            ),
            dtype=int,
        )
    mat = np.empty(
        (len(explicit_bit_vect_list), len(explicit_bit_vect_list[0])), dtype=int
    )
    for i, fingerprint in enumerate(explicit_bit_vect_list):
        DataStructs.ConvertToNumpyArray(fingerprint, mat[i, :])
    return mat


class TestMol2MorganFingerprint(unittest.TestCase):
    """Unittest for MolToFoldedMorganFingerprint, which calculates folded Morgan Fingerprints."""

    def test_can_be_constructed(self) -> None:
        """Test if the MolToFoldedMorganFingerprint pipeline element can be constructed.

        Returns
        -------
        None
        """
        mol_fp = MolToFoldedMorganFingerprint()
        mol_fp_copy = mol_fp.copy()
        self.assertTrue(mol_fp_copy is not mol_fp)
        for key, value in mol_fp.get_params().items():
            self.assertEqual(value, mol_fp_copy.get_params()[key])
        mol_fp_recreated = MolToFoldedMorganFingerprint(**mol_fp.get_params())
        for key, value in mol_fp.get_params().items():
            self.assertEqual(value, mol_fp_recreated.get_params()[key])

    def test_sparse_dense_accordance(self) -> None:
        """Test if the calculation of Morgan fingprints in dense and sparse are equal.

        Compared to precalculated values.

        Returns
        -------
        None
        """
        smi2mol = SmilesToMolPipelineElement()
        sparse_morgan = MolToFoldedMorganFingerprint(
            radius=2, n_bits=1024, sparse_output=True
        )
        dense_morgan = MolToFoldedMorganFingerprint(
            radius=2, n_bits=1024, sparse_output=False
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

        sparse_output = sparse_pipeline.fit_transform(test_smiles)
        dense_output = dense_pipeline.fit_transform(test_smiles)

        self.assertTrue(np.all(sparse_output.toarray() == dense_output))

    def test_output_types(self) -> None:
        """Test equality of different output_types."""

        smi2mol = SmilesToMolPipelineElement()
        sparse_morgan = MolToFoldedMorganFingerprint(
            radius=2, n_bits=1024, output_type="sparse"
        )
        dense_morgan = MolToFoldedMorganFingerprint(
            radius=2, n_bits=1024, output_type="dense"
        )
        explicit_bit_vect_morgan = MolToFoldedMorganFingerprint(
            radius=2, n_bits=1024, output_type="explicit_bit_vect"
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
                _explicit_bit_vect_list_to_numpy(explicit_bit_vect_morgan_output),
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
