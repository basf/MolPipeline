"""Test SubpipelineExtractor."""

import unittest

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from molpipeline import ErrorFilter, FilterReinserter, Pipeline, PostPredictionWrapper
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import MolToMorganFP, MolToSmiles
from molpipeline.utils.subpipeline import SubpipelineExtractor


class TestSubpipelineExtractor(unittest.TestCase):
    """Test SubpipelineExtractor."""

    def test_get_molecule_reader_element(self) -> None:
        """Test extracting molecule reader element from pipelines."""

        # test basic example
        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("morgan", MolToMorganFP(radius=1, n_bits=64)),
                ("model", RandomForestClassifier()),
            ]
        )
        extractor = SubpipelineExtractor(pipeline)
        self.assertEqual(extractor.get_molecule_reader_element(), pipeline.steps[0][1])

        # test with multiple molecule readers
        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("mol2smi", MolToSmiles()),
                ("smi2mol2", SmilesToMol()),
                ("morgan", MolToMorganFP(radius=1, n_bits=64)),
                ("model", RandomForestClassifier()),
            ]
        )
        extractor = SubpipelineExtractor(pipeline)
        self.assertEqual(extractor.get_molecule_reader_element(), pipeline.steps[2][1])

    def test_get_model_element(self) -> None:
        """Test extracting model element from pipeline."""

        # test basic example
        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("morgan", MolToMorganFP(radius=1, n_bits=64)),
                ("model", RandomForestClassifier()),
            ]
        )
        extractor = SubpipelineExtractor(pipeline)
        self.assertEqual(extractor.get_model_element(), pipeline.steps[2][1])

        # test with PostPredictionWrapper
        error_filter = ErrorFilter()
        error_reinserter = PostPredictionWrapper(
            FilterReinserter.from_error_filter(error_filter, None)
        )
        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("error_filter", error_filter),
                ("morgan", MolToMorganFP(radius=1, n_bits=64)),
                ("model", RandomForestClassifier()),
                (
                    "error_reinserter",
                    error_reinserter,
                ),
            ]
        )
        extractor = SubpipelineExtractor(pipeline)
        self.assertEqual(extractor.get_model_element(), pipeline.steps[3][1])

    def test_get_molecule_reader_subpipeline(self) -> None:
        """Test extracting subpipeline up to the molecule reader element from pipelines."""

        # test basic example
        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("morgan", MolToMorganFP(radius=1, n_bits=64)),
                ("model", RandomForestClassifier()),
            ]
        )
        extractor = SubpipelineExtractor(pipeline)
        subpipeline = extractor.get_molecule_reader_subpipeline()
        self.assertIsInstance(subpipeline, Pipeline)
        self.assertEqual(len(subpipeline.steps), 1)  # type: ignore[union-attr]
        self.assertIs(subpipeline.steps[0], pipeline.steps[0])  # type: ignore[union-attr]

        # test with multiple molecule readers
        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("mol2smi", MolToSmiles()),
                ("smi2mol2", SmilesToMol()),
                ("morgan", MolToMorganFP(radius=1, n_bits=64)),
                ("model", RandomForestClassifier()),
            ]
        )
        extractor = SubpipelineExtractor(pipeline)
        subpipeline = extractor.get_molecule_reader_subpipeline()
        self.assertIsInstance(subpipeline, Pipeline)
        self.assertEqual(len(subpipeline.steps), 3)  # type: ignore[union-attr]
        for i, subpipe_step in enumerate(subpipeline.steps):  # type: ignore[union-attr]
            self.assertIs(subpipe_step, pipeline.steps[i])

    def test_get_model_subpipeline(self) -> None:
        """Test extracting subpipeline up to the the model element from pipelines."""

        # test basic example
        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("morgan", MolToMorganFP(radius=1, n_bits=64)),
                ("model", RandomForestClassifier()),
            ]
        )
        extractor = SubpipelineExtractor(pipeline)
        subpipeline = extractor.get_model_subpipeline()
        self.assertIsInstance(subpipeline, Pipeline)
        self.assertEqual(len(subpipeline.steps), 3)  # type: ignore[union-attr]
        for i, subpipe_step in enumerate(subpipeline.steps):  # type: ignore[union-attr]
            self.assertIs(subpipe_step, pipeline.steps[i])

        # test with PostPredictionWrapper
        error_filter = ErrorFilter()
        error_reinserter = PostPredictionWrapper(
            FilterReinserter.from_error_filter(error_filter, None)
        )
        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("error_filter", error_filter),
                ("morgan", MolToMorganFP(radius=1, n_bits=64)),
                ("model", RandomForestClassifier()),
                (
                    "error_reinserter",
                    error_reinserter,
                ),
            ]
        )
        extractor = SubpipelineExtractor(pipeline)
        subpipeline = extractor.get_model_subpipeline()
        self.assertIsInstance(subpipeline, Pipeline)
        self.assertEqual(len(subpipeline.steps), 4)  # type: ignore[union-attr]
        for i, subpipe_step in enumerate(subpipeline.steps):  # type: ignore[union-attr]
            self.assertIs(subpipe_step, pipeline.steps[i])

    def test_get_all_filter_reinserter_fill_values(self) -> None:
        """Test extracting all FilterReinserter fill values from pipelines."""

        test_fill_values = [None, np.nan]

        for test_fill_value in test_fill_values:
            error_filter = ErrorFilter()
            error_reinserter = PostPredictionWrapper(
                FilterReinserter.from_error_filter(error_filter, test_fill_value)
            )
            pipeline = Pipeline(
                [
                    ("smi2mol", SmilesToMol()),
                    ("error_filter", error_filter),
                    ("morgan", MolToMorganFP(radius=1, n_bits=64)),
                    ("model", RandomForestClassifier()),
                    (
                        "error_reinserter",
                        error_reinserter,
                    ),
                ]
            )
            extractor = SubpipelineExtractor(pipeline)
            fill_values = extractor.get_all_filter_reinserter_fill_values()
            self.assertEqual(fill_values, [test_fill_value])
