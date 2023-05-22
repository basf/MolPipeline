"""Unittest for handling Nones."""

import unittest

import numpy as np
from rdkit import RDLogger

from molpipeline.pipeline import MolPipeline
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.mol2any.mol2smiles import MolToSmilesPipelineElement
from molpipeline.pipeline_elements.mol2any.mol2morgan_fingerprint import (
    MolToFoldedMorganFingerprint,
)
from molpipeline.pipeline_elements.mol2any.mol2rdkit_phys_chem import (
    MolToRDKitPhysChem,
)

rdlog = RDLogger.logger()
rdlog.setLevel(RDLogger.CRITICAL)

TEST_SMILES = ["NCCCO", "abc"]
EXPECTED_OUTPUT = ["NCCCO", None]


class NoneTest(unittest.TestCase):
    """Unittest for None Handling."""

    def test_none_dummy_fill_molpipeline(self) -> None:
        """Assert that invalid smiles are transformed to None."""
        pipeline = MolPipeline(
            [
                SmilesToMolPipelineElement(),
                MolToSmilesPipelineElement(),
            ],
            none_handling="fill_dummy",
            fill_value=None,
        )
        out = pipeline.transform(TEST_SMILES)
        for pred_val, true_val in zip(out, EXPECTED_OUTPUT):
            self.assertEqual(pred_val, true_val)

    def test_none_dummy_remove_record_molpipeline(self) -> None:
        """Assert that invalid smiles are transformed to None."""
        pipeline = MolPipeline(
            [
                SmilesToMolPipelineElement(),
                MolToSmilesPipelineElement(),
            ],
            none_handling="record_remove",
        )
        out = pipeline.transform(TEST_SMILES)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0], EXPECTED_OUTPUT[0])
        self.assertEqual(len(pipeline.none_collector.none_indices), 1)
        self.assertEqual(pipeline.none_collector.none_indices[0], 1)

    def test_dummy_remove_morgan_record_molpipeline(self) -> None:
        """Assert that invalid smiles are transformed to None."""
        pipeline = MolPipeline(
            [
                SmilesToMolPipelineElement(),
                MolToFoldedMorganFingerprint(),
            ],
            none_handling="record_remove",
        )
        out = pipeline.transform(TEST_SMILES)
        self.assertEqual(out.shape, (1, 2048))

    def test_dummy_remove_physchem_record_molpipeline(self) -> None:
        """Assert that invalid smiles are transformed to None."""
        pipeline = MolPipeline(
            [
                SmilesToMolPipelineElement(),
                MolToRDKitPhysChem(),
            ],
            none_handling="record_remove",
        )
        pipeline2 = pipeline.copy()
        pipeline.fit(TEST_SMILES)
        out = pipeline.transform(TEST_SMILES)
        out2 = pipeline2.fit_transform(TEST_SMILES)
        self.assertEqual(out.shape, out2.shape)
        self.assertTrue(np.max(np.abs(out - out2)) < 0.000001)

    def test_dummy_fill_physchem_record_molpipeline(self) -> None:
        """Assert that invalid smiles are transformed to None."""
        pipeline = MolPipeline(
            [
                SmilesToMolPipelineElement(),
                MolToRDKitPhysChem(),
            ],
            none_handling="fill_dummy",
            fill_value=10,
        )
        pipeline2 = pipeline.copy()
        pipeline.fit(TEST_SMILES)
        out = pipeline.transform(TEST_SMILES)
        out2 = pipeline2.fit_transform(TEST_SMILES)
        self.assertEqual(out.shape, out2.shape)
        self.assertEqual(out.shape, (2, 207))
        self.assertTrue(np.nanmax(np.abs(out - out2)) < 0.000001)
