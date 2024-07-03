"""Test Chemprop neural fingerprint."""

import logging
import unittest

from sklearn.base import clone

from molpipeline.estimators.chemprop.neural_fingerprint import ChempropNeuralFP
from molpipeline.utils.json_operations import recursive_from_json, recursive_to_json

# pylint: disable=relative-beyond-top-level
from test_extras.test_chemprop.chemprop_test_utils.compare_models import compare_params
from test_extras.test_chemprop.chemprop_test_utils.default_models import (
    get_neural_fp_encoder,
)

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)


class TestChempropNeuralFingerprint(unittest.TestCase):
    """Test the Chemprop model."""

    def test_clone(self) -> None:
        """Test the clone method."""
        chemprop_fp_encoder = get_neural_fp_encoder()
        cloned_encoder = clone(chemprop_fp_encoder)
        self.assertIsInstance(cloned_encoder, ChempropNeuralFP)
        compare_params(self, chemprop_fp_encoder, cloned_encoder)

    def test_json_serialization(self) -> None:
        """Test the to_json and from_json methods."""
        chemprop_fp_encoder = get_neural_fp_encoder()
        chemprop_json = recursive_to_json(chemprop_fp_encoder)
        chemprop_encoder_copy = recursive_from_json(chemprop_json)
        compare_params(self, chemprop_fp_encoder, chemprop_encoder_copy)
