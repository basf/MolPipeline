"""Unittest for handling Nones."""

import unittest
from typing import Any

import numpy as np
from rdkit import RDLogger
from sklearn.base import clone

from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.error_handling import ErrorFilter, ErrorReplacer
from molpipeline.pipeline_elements.mol2any.mol2morgan_fingerprint import (
    MolToFoldedMorganFingerprint,
)
from molpipeline.pipeline_elements.mol2any.mol2rdkit_phys_chem import MolToRDKitPhysChem
from molpipeline.pipeline_elements.mol2any.mol2smiles import MolToSmilesPipelineElement
from molpipeline.pipeline_elements.post_prediction import PostPredictionWrapper
from tests.utils.mock_element import MockTransformingPipelineElement

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
        self.assertEqual(out.shape, (3, 208))
        self.assertTrue(np.nanmax(np.abs(out - out2)) < 0.000001)

    def test_replace_mixed_datatypes(self) -> None:
        """Assert that invalid values are replaced by fill value."""

        # test values
        test_values = [
            123,
            20099,  # index 1 will be replaced in this test
            1337,
        ]

        test_tuples = [
            (42, test_values, False),  # test scalar fill value with list returned
            (np.nan, test_values, False),  # test float fill value with list returned
            ([], test_values, False),  # test object list fill value with list returned
            ({}, test_values, False),  # test object dict fill value with list returned
            (None, test_values, False),  # test None fill value with list returned
            (42, test_values, True),  # test scalar fill value with numpy array returned
            (
                np.nan,
                test_values,
                True,
            ),  # test float fill value with numpy array returned
            (
                {},
                test_values,
                True,
            ),  # test object fill value with numpy array returned
            (
                None,
                test_values,
                True,
            ),  # test object fill value with numpy array returned
        ]

        for fill_value, test_values, to_numpy in test_tuples:
            this_test_values: Any = test_values
            if to_numpy:
                this_test_values = np.array(test_values)

            mock2mock = MockTransformingPipelineElement(
                invalid_values={
                    test_values[1]
                },  # replaces element at index 1 with an invalid instance
                to_numpy=to_numpy,
                uuid="test1234",
            )
            error_filter = ErrorFilter.from_element_list([mock2mock])
            error_replacer = ErrorReplacer.from_error_filter(
                error_filter=error_filter, fill_value=fill_value
            )
            pipeline = Pipeline(
                [
                    ("mock2mock", mock2mock),
                    ("error_filter", error_filter),
                    ("error_replacer", error_replacer),
                ],
                n_jobs=1,
            )
            pipeline2 = clone(pipeline)
            pipeline.fit(this_test_values)
            out = pipeline.transform(this_test_values)
            out2 = pipeline2.fit_transform(this_test_values)

            if to_numpy:
                self.assertTrue(isinstance(out, np.ndarray))
                self.assertTrue(isinstance(out2, np.ndarray))
            else:
                self.assertTrue(isinstance(out, list))
                self.assertTrue(isinstance(out2, list))

            self.assertEqual(len(out), len(out2))
            self.assertEqual(len(this_test_values), len(out))
            self.assertEqual(out[0], this_test_values[0])
            self.assertEqual(out2[0], this_test_values[0])
            self.assertEqual(out[2], this_test_values[2])
            self.assertEqual(out2[2], this_test_values[2])

            if isinstance(fill_value, float) and np.isnan(fill_value).all():
                self.assertTrue(np.isnan(out[1]).all())
                self.assertTrue(np.isnan(out2[1]).all())
            else:
                if fill_value is None:
                    self.assertIsNone(out[1])
                    self.assertIsNone(out2[1])
                else:
                    self.assertEqual(out[1], fill_value)
                    self.assertEqual(out2[1], fill_value)

    def test_replace_mixed_datatypes_expected_failures(self) -> None:
        """Test expected failures when replacing invalid values."""

        # test values
        test_values = [
            123,
            20099,  # index 1 will be replaced in this test
            1337,
        ]

        mock2mock = MockTransformingPipelineElement(
            invalid_values={test_values[1]},
            to_numpy=True,
            uuid="test1234",
        )
        error_filter = ErrorFilter.from_element_list([mock2mock])
        error_replacer = ErrorReplacer.from_error_filter(
            error_filter=error_filter, fill_value=[]
        )
        pipeline = Pipeline(
            [
                ("mock2mock", mock2mock),
                ("error_filter", error_filter),
                ("error_replacer", error_replacer),
            ],
            n_jobs=1,
        )
        pipeline2 = clone(pipeline)

        self.assertRaises(ValueError, pipeline.fit, test_values)
        self.assertRaises(ValueError, pipeline.transform, test_values)
        self.assertRaises(ValueError, pipeline2.fit_transform, test_values)
