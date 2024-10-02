"""Test the comparison functions."""

from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.error_handling import ErrorFilter, FilterReinserter
from molpipeline.mol2any import (
    MolToConcatenatedVector,
    MolToMorganFP,
    MolToRDKitPhysChem,
)
from molpipeline.post_prediction import PostPredictionWrapper
from molpipeline.utils.comparison import check_pipelines_equivalent


def get_test_pipeline() -> Pipeline:
    """Get a test pipeline."""

    error_filter = ErrorFilter(filter_everything=True)
    pipeline = Pipeline(
        [
            ("smi2mol", SmilesToMol()),
            (
                "mol2fp",
                MolToConcatenatedVector(
                    [
                        ("morgan", MolToMorganFP(n_bits=2048)),
                        ("physchem", MolToRDKitPhysChem()),
                    ]
                ),
            ),
            ("error_filter", error_filter),
            ("rf", RandomForestClassifier()),
            (
                "filter_reinserter",
                PostPredictionWrapper(
                    FilterReinserter.from_error_filter(error_filter, None)
                ),
            ),
        ],
        n_jobs=1,
    )

    # Set up pipeline
    return pipeline


class TestComparison(TestCase):
    """Test if functional equivalent pipelines are detected as such."""

    def test_are_equal(self) -> None:
        """Test if two equivalent pipelines are detected as such."""

        pipeline_a = get_test_pipeline()
        pipeline_b = get_test_pipeline()
        self.assertTrue(check_pipelines_equivalent(pipeline_a, pipeline_b))

    def test_are_not_equal(self) -> None:
        """Test if two different pipelines are detected as such."""
        # Test changed parameters
        pipeline_a = get_test_pipeline()
        pipeline_b = get_test_pipeline()
        pipeline_b.set_params(mol2fp__morgan__n_bits=1024)
        self.assertFalse(check_pipelines_equivalent(pipeline_a, pipeline_b))

        # Test changed steps
        pipeline_b = get_test_pipeline()
        last_step = pipeline_b.steps[-1]
        pipeline_b.steps = pipeline_b.steps[:-1]
        self.assertFalse(check_pipelines_equivalent(pipeline_a, pipeline_b))

        # Test if adding the step back makes the pipelines equivalent
        pipeline_b.steps.append(last_step)
        self.assertTrue(check_pipelines_equivalent(pipeline_a, pipeline_b))
