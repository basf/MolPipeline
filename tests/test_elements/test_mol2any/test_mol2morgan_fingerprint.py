"""Tests for the MolToFoldedMorganFingerprint pipeline element."""
import unittest
import numpy as np

from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.mol2any.mol2morgan_fingerprint import (
    MolToFoldedMorganFingerprint,
)
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement

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


if __name__ == "__main__":
    unittest.main()
