"""Tests for the MolToFoldedMorganFingerprint pipeline element."""

import numpy as np
import unittest

from molpipeline.pipeline import MolPipeline
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
    def test_spase_dense_accordance(self) -> None:
        """Test if the calculation of RDKitPhysChem Descriptors works as expected.

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
        sparse_pipeline = MolPipeline([smi2mol, sparse_morgan])
        dense_pipeline = MolPipeline([smi2mol, dense_morgan])

        sparse_output = sparse_pipeline.fit_transform(test_smiles)
        dense_output = dense_pipeline.fit_transform(test_smiles)

        self.assertTrue(np.all(sparse_output.toarray() == dense_output))


if __name__ == "__main__":
    unittest.main()
