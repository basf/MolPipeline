"""Unittest for handling Nones."""

import unittest

import numpy as np
from rdkit import RDLogger

from sklearn.base import clone

from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.mol2any.mol2smiles import MolToSmilesPipelineElement
from molpipeline.pipeline_elements.mol2any.mol2morgan_fingerprint import (
    MolToFoldedMorganFingerprint,
)
from molpipeline.pipeline_elements.mol2any.mol2rdkit_phys_chem import (
    MolToRDKitPhysChem,
)
from molpipeline.pipeline_elements.post_prediction import PostPredictionWrapper
from molpipeline.pipeline_elements.error_handling import ErrorFilter, ErrorReplacer

rdlog = RDLogger.logger()
rdlog.setLevel(RDLogger.CRITICAL)

TEST_SMILES = ["NCCCO", "abc", "c1ccccc1"]
EXPECTED_OUTPUT = ["NCCCO", None, "c1ccccc1"]


class NoneTest(unittest.TestCase):
    """Unittest for None Handling."""

    def test_error_dummy_fill_molpipeline(self) -> None:
        """Assert that invalid smiles are transformed to None."""

        smi2mol = SmilesToMolPipelineElement()
        mol2smi = MolToSmilesPipelineElement()
        remove_error = ErrorFilter.from_element_list([smi2mol, mol2smi])
        replace_error = PostPredictionWrapper(
            ErrorReplacer.from_error_filter(remove_error, fill_value=None)
        )

        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("mol2smi", mol2smi),
                ("remove_error", remove_error),
                ("replace_error", replace_error),
            ]
        )
        out = pipeline.fit_transform(TEST_SMILES)
        for pred_val, true_val in zip(out, EXPECTED_OUTPUT):
            self.assertEqual(pred_val, true_val)

    def test_error_dummy_remove_record_molpipeline(self) -> None:
        """Assert that invalid smiles are transformed to None."""
        smi2mol = SmilesToMolPipelineElement()
        mol2smi = MolToSmilesPipelineElement()
        error_filter = ErrorFilter.from_element_list([smi2mol, mol2smi])
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("mol2smi", mol2smi),
                ("error_filter", error_filter),
            ]
        )
        out = pipeline.transform(TEST_SMILES)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], EXPECTED_OUTPUT[0])

    def test_dummy_remove_morgan_record_molpipeline(self) -> None:
        """Assert that invalid smiles are transformed to None."""
        smi2mol = SmilesToMolPipelineElement()
        mol2morgan = MolToFoldedMorganFingerprint()
        error_filter = ErrorFilter.from_element_list([smi2mol, mol2morgan])
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("mol2morgan", mol2morgan),
                ("error_filter", error_filter),
            ],
        )
        out = pipeline.transform(TEST_SMILES)
        self.assertEqual(out.shape, (2, 2048))

    def test_dummy_remove_physchem_record_molpipeline(self) -> None:
        """Assert that invalid smiles are transformed to None."""
        smi2mol = SmilesToMolPipelineElement()
        mol2physchem = MolToRDKitPhysChem()
        remove_none = ErrorFilter.from_element_list([smi2mol, mol2physchem])
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("mol2physchem", mol2physchem),
                ("remove_none", remove_none),
            ],
        )
        pipeline2 = clone(pipeline)
        pipeline.fit(TEST_SMILES)
        out = pipeline.transform(TEST_SMILES)
        out2 = pipeline2.fit_transform(TEST_SMILES)
        self.assertEqual(out.shape, out2.shape)
        self.assertTrue(np.max(np.abs(out - out2)) < 0.000001)

    def test_dummy_remove_physchem_record_autodetect_molpipeline(self) -> None:
        """Assert that invalid smiles are transformed to None."""
        smi2mol = SmilesToMolPipelineElement()
        mol2physchem = MolToRDKitPhysChem()
        remove_none = ErrorFilter(filter_everything=True)
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("mol2physchem", mol2physchem),
                ("remove_none", remove_none),
            ],
        )
        pipeline2 = clone(pipeline)
        pipeline.fit(TEST_SMILES)
        out = pipeline.transform(TEST_SMILES)
        print(pipeline2["remove_none"].filter_everything)
        out2 = pipeline2.fit_transform(TEST_SMILES)
        self.assertEqual(out.shape, out2.shape)
        self.assertTrue(np.max(np.abs(out - out2)) < 0.000001)

    def test_dummy_fill_physchem_record_molpipeline(self) -> None:
        """Assert that invalid smiles are transformed to None."""

        smi2mol = SmilesToMolPipelineElement()
        mol2physchem = MolToRDKitPhysChem()
        remove_none = ErrorFilter.from_element_list([smi2mol, mol2physchem])
        fill_none = PostPredictionWrapper(
            ErrorReplacer.from_error_filter(remove_none, fill_value=10)
        )

        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("mol2physchem", mol2physchem),
                ("remove_none", remove_none),
                ("fill_none", fill_none),
            ],
            n_jobs=1,
        )
        pipeline2 = clone(pipeline)
        pipeline.fit(TEST_SMILES)
        out = pipeline.transform(TEST_SMILES)
        out2 = pipeline2.fit_transform(TEST_SMILES)
        self.assertEqual(out.shape, out2.shape)
        self.assertEqual(out.shape, (3, 207))
        self.assertTrue(np.nanmax(np.abs(out - out2)) < 0.000001)