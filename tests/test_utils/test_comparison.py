"""Test the comparison functions."""

from typing import Callable
from unittest import TestCase

from molpipeline import Pipeline
from molpipeline.utils.comparison import check_pipelines_equivalent
from tests.utils.default_models import (
    get_morgan_physchem_rf_pipeline,
    get_standardization_pipeline,
)


class TestComparison(TestCase):
    """Test if functional equivalent pipelines are detected as such."""

    def test_are_equal(self) -> None:
        """Test if two equivalent pipelines are detected as such."""
        # Test standardization pipelines
        pipline_method_list: list[Callable[[int], Pipeline]] = [
            get_standardization_pipeline,
            get_morgan_physchem_rf_pipeline,
        ]
        for pipeline_method in pipline_method_list:
            pipeline_a = pipeline_method(1)
            pipeline_b = pipeline_method(1)
            self.assertTrue(check_pipelines_equivalent(pipeline_a, pipeline_b))

    def test_are_not_equal(self) -> None:
        """Test if two different pipelines are detected as such."""
        # Test changed parameters
        pipeline_a = get_morgan_physchem_rf_pipeline()
        pipeline_b = get_morgan_physchem_rf_pipeline()
        pipeline_b.set_params(mol2fp__morgan__n_bits=1024)
        self.assertFalse(check_pipelines_equivalent(pipeline_a, pipeline_b))

        # Test changed steps
        pipeline_b = get_morgan_physchem_rf_pipeline()
        last_step = pipeline_b.steps[-1]
        pipeline_b.steps = pipeline_b.steps[:-1]
        self.assertFalse(check_pipelines_equivalent(pipeline_a, pipeline_b))

        # Test if adding the step back makes the pipelines equivalent
        pipeline_b.steps.append(last_step)
        self.assertTrue(check_pipelines_equivalent(pipeline_a, pipeline_b))
