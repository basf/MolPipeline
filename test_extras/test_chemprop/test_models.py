"""Test Chemprop component wrapper."""

import logging
import unittest
from typing import Iterable

import torch
from chemprop.nn.loss import MSELoss
from sklearn.base import clone

from molpipeline.estimators.chemprop.component_wrapper import (
    MPNN,
    MeanAggregation,
    RegressionFFN,
)
from molpipeline.estimators.chemprop.models import (
    ChempropClassifier,
    ChempropModel,
    ChempropRegressor,
)
from molpipeline.estimators.chemprop.neural_fingerprint import ChempropNeuralFP
from molpipeline.utils.json_operations import recursive_from_json, recursive_to_json

# pylint: disable=relative-beyond-top-level
from test_extras.test_chemprop.chemprop_test_utils.compare_models import compare_params
from test_extras.test_chemprop.chemprop_test_utils.constant_vars import DEFAULT_PARAMS, NO_IDENTITY_CHECK
from test_extras.test_chemprop.chemprop_test_utils.default_models import (
    get_chemprop_model_binary_classification_mpnn,
)

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)


class TestChempropModel(unittest.TestCase):
    """Test the Chemprop model."""

    def test_get_params(self) -> None:
        """Test the get_params and set_params methods."""
        chemprop_model = get_chemprop_model_binary_classification_mpnn()
        orig_params = chemprop_model.get_params(deep=True)
        expected_params = dict(DEFAULT_PARAMS)  # Shallow copy

        self.assertSetEqual(set(orig_params), set(expected_params))
        # Check if the parameters are as expected
        for param_name, param in expected_params.items():
            if param_name in NO_IDENTITY_CHECK:
                if isinstance(param, Iterable):
                    self.assertIsInstance(orig_params[param_name], type(param))
                    for i, p in enumerate(param):
                        self.assertIsInstance(orig_params[param_name][i], p)
                elif isinstance(param, type):
                    self.assertIsInstance(orig_params[param_name], param)
                else:
                    raise ValueError(f"{param_name} should be a type.")
            else:
                self.assertEqual(
                    orig_params[param_name], param, f"Test failed for {param_name}"
                )

        new_params = {
            "batch_size": 32,
            "model__agg": MeanAggregation(),
            "model__message_passing__activation": "tanh",
            "model__message_passing__depth": 2,
        }
        # Check setting new parameters
        chemprop_model.set_params(**new_params)
        model_params = chemprop_model.get_params(deep=True)
        for param_name, param in new_params.items():
            if param_name in ["model__agg"]:
                self.assertIsInstance(model_params[param_name], type(param))
                continue
            self.assertEqual(param, model_params[param_name])

    def test_clone(self) -> None:
        """Test the clone method."""
        chemprop_model = get_chemprop_model_binary_classification_mpnn()
        cloned_model = clone(chemprop_model)
        self.assertIsInstance(cloned_model, ChempropModel)
        compare_params(self, chemprop_model, cloned_model)

    def test_classifier_methods(self) -> None:
        """Test the classifier methods."""
        chemprop_model = get_chemprop_model_binary_classification_mpnn()
        # pylint: disable=protected-access
        self.assertTrue(chemprop_model._is_binary_classifier())
        self.assertFalse(chemprop_model._is_multiclass_classifier())
        # pylint: enable=protected-access
        self.assertTrue(hasattr(chemprop_model, "predict_proba"))

    def test_neural_fp(self) -> None:
        """Test the to_encoder method."""
        chemprop_model = get_chemprop_model_binary_classification_mpnn()
        neural_fp = chemprop_model.to_encoder()
        self.assertIsInstance(neural_fp, ChempropNeuralFP)
        self.assertIsInstance(neural_fp.model, MPNN)
        # the model should be cloned
        self.assertNotEqual(id(chemprop_model.model), id(neural_fp.model))
        self.assertEqual(neural_fp.disable_fitting, True)

    def test_json_serialization(self) -> None:
        """Test the to_json and from_json methods."""
        chemprop_model = get_chemprop_model_binary_classification_mpnn()
        chemprop_json = recursive_to_json(chemprop_model)
        chemprop_model_copy = recursive_from_json(chemprop_json)
        param_dict = chemprop_model_copy.get_params(deep=True)

        self.assertSetEqual(set(param_dict.keys()), set(DEFAULT_PARAMS.keys()))
        for param_name, param in DEFAULT_PARAMS.items():
            if param_name in NO_IDENTITY_CHECK:
                if isinstance(param, Iterable):
                    self.assertIsInstance(param_dict[param_name], type(param))
                    for i, p in enumerate(param):
                        self.assertIsInstance(param_dict[param_name][i], p)
                elif isinstance(param, type):
                    self.assertIsInstance(param_dict[param_name], param)
                else:
                    raise ValueError(f"{param_name} should be a type.")
            elif param_name == "model__predictor__task_weights":
                self.assertTrue(torch.allclose(param, param_dict[param_name]))
            else:
                self.assertEqual(
                    param_dict[param_name], param, f"Test failed for {param_name}"
                )


class TestChempropClassifier(unittest.TestCase):
    """Test the Chemprop classifier model."""

    def test_get_params(self) -> None:
        """Test the get_params and set_params methods."""
        chemprop_model = ChempropClassifier(lightning_trainer__accelerator="cpu")
        param_dict = chemprop_model.get_params(deep=True)
        expected_params = dict(DEFAULT_PARAMS)  # Shallow copy
        self.assertSetEqual(set(param_dict.keys()), set(expected_params.keys()))
        for param_name, param in expected_params.items():
            if param_name in NO_IDENTITY_CHECK:
                if isinstance(param, Iterable):
                    self.assertIsInstance(param_dict[param_name], type(param))
                    for i, p in enumerate(param):
                        self.assertIsInstance(param_dict[param_name][i], p)
                elif isinstance(param, type):
                    self.assertIsInstance(param_dict[param_name], param)
                else:
                    raise ValueError(f"{param_name} should be a type.")
            else:
                self.assertEqual(
                    param_dict[param_name], param, f"Test failed for {param_name}"
                )


class TestChempropRegressor(unittest.TestCase):
    """Test the Chemprop regressor model."""

    def test_get_params(self) -> None:
        """Test the get_params and set_params methods."""
        chemprop_model = ChempropRegressor(lightning_trainer__accelerator="cpu")
        param_dict = chemprop_model.get_params(deep=True)
        expected_params = dict(DEFAULT_PARAMS)
        expected_params["model__predictor"] = RegressionFFN
        expected_params["model__predictor__criterion"] = MSELoss
        self.assertSetEqual(set(param_dict.keys()), set(expected_params.keys()))
        for param_name, param in expected_params.items():
            if param_name in NO_IDENTITY_CHECK:
                if isinstance(param, Iterable):
                    self.assertIsInstance(param_dict[param_name], type(param))
                    for i, p in enumerate(param):
                        self.assertIsInstance(param_dict[param_name][i], p)
                elif isinstance(param, type):
                    self.assertIsInstance(param_dict[param_name], param)
                else:
                    raise ValueError(f"{param_name} should be a type.")
            else:
                self.assertEqual(
                    param_dict[param_name], param, f"Test failed for {param_name}"
                )
