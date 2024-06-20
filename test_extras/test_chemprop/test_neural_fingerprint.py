"""Test Chemprop neural fingerprint."""

import logging
import unittest
from typing import Iterable

import torch
from sklearn.base import clone

from molpipeline.estimators.chemprop.neural_fingerprint import ChempropNeuralFP
from molpipeline.utils.json_operations import recursive_from_json, recursive_to_json

# pylint: disable=relative-beyond-top-level
from .chemprop_test_utils.compare_models import compare_params
from .chemprop_test_utils.constant_vars import NO_IDENTITY_CHECK
from .chemprop_test_utils.default_models import get_classification_mpnn

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)


def get_neural_fp_encoder() -> ChempropNeuralFP:
    """Get the Chemprop model.

    Returns
    -------
    ChempropNeuralFP
        The Chemprop model.
    """
    mpnn = get_classification_mpnn()
    chemprop_model = ChempropNeuralFP(model=mpnn, lightning_trainer__accelerator="cpu")
    return chemprop_model


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
        original_params = chemprop_fp_encoder.get_params(deep=True)
        recreated_params = chemprop_encoder_copy.get_params(deep=True)

        self.assertSetEqual(set(original_params.keys()), set(recreated_params.keys()))
        for param_name, param in original_params.items():
            if param_name in NO_IDENTITY_CHECK:
                self.assertIsInstance(recreated_params[param_name], type(param))
                if isinstance(param, Iterable):
                    for i, p in enumerate(param):
                        self.assertIsInstance(recreated_params[param_name][i], type(p))
            elif param_name == "model__predictor__task_weights":
                self.assertTrue(torch.allclose(param, recreated_params[param_name]))
            else:
                self.assertEqual(
                    recreated_params[param_name], param, f"Test failed for {param_name}"
                )
