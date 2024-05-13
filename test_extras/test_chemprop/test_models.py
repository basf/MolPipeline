"""Test Chemprop component wrapper."""

import logging
import unittest

from chemprop.nn.loss import BCELoss, LossFunction, MSELoss
from lightning import pytorch as pl
from sklearn.base import clone
from torch import Tensor, nn

from molpipeline.estimators.chemprop.component_wrapper import (
    MPNN,
    BinaryClassificationFFN,
    BondMessagePassing,
    MeanAggregation,
    RegressionFFN,
    SumAggregation,
)
from molpipeline.estimators.chemprop.models import (
    ChempropClassifier,
    ChempropModel,
    ChempropRegressor,
)
from molpipeline.estimators.chemprop.neural_fingerprint import ChempropNeuralFP

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)


def get_model() -> ChempropModel:
    """Get the Chemprop model.

    Returns
    -------
    ChempropModel
        The Chemprop model.
    """
    binary_clf_ffn = BinaryClassificationFFN()
    aggregate = SumAggregation()
    bond_message_passing = BondMessagePassing()
    mpnn = MPNN(
        message_passing=bond_message_passing,
        agg=aggregate,
        predictor=binary_clf_ffn,
    )
    chemprop_model = ChempropModel(model=mpnn)
    return chemprop_model


DEFAULT_PARAMS = {
    "batch_size": 64,
    "lightning_trainer": pl.Trainer,
    "model": MPNN,
    "model__agg__dim": 0,
    "model__agg": SumAggregation,
    "model__batch_norm": True,
    "model__final_lr": 0.0001,
    "model__init_lr": 0.0001,
    "model__max_lr": 0.001,
    "model__message_passing__activation": "relu",
    "model__message_passing__bias": False,
    "model__message_passing__d_e": 14,
    "model__message_passing__d_h": 300,
    "model__message_passing__d_v": 72,
    "model__message_passing__d_vd": None,
    "model__message_passing__depth": 3,
    "model__message_passing__dropout_rate": 0.0,
    "model__message_passing__undirected": False,
    "model__message_passing": BondMessagePassing,
    "model__metric_list": None,
    "model__predictor__activation": "relu",
    "model__warmup_epochs": 2,
    "model__predictor": BinaryClassificationFFN,
    "model__predictor__criterion": BCELoss,
    "model__predictor__dropout": 0,
    "model__predictor__hidden_dim": 300,
    "model__predictor__input_dim": 300,
    "model__predictor__n_layers": 1,
    "model__predictor__n_tasks": 1,
    "model__predictor__output_transform": nn.Identity,
    "model__predictor__task_weights": Tensor([1.0]),
    "model__predictor__threshold": None,
    "n_jobs": 1,
}

NO_IDENTITY_CHECK = [
    "model__agg",
    "model__message_passing",
    "lightning_trainer",
    "model",
    "model__predictor",
    "model__predictor__criterion",
    "model__predictor__output_transform",
]


class TestChempropModel(unittest.TestCase):
    """Test the Chemprop model."""

    def test_get_params(self) -> None:
        """Test the get_params and set_params methods."""
        chemprop_model = get_model()
        orig_params = chemprop_model.get_params(deep=True)
        expected_params = dict(DEFAULT_PARAMS)  # Shallow copy

        self.assertSetEqual(set(orig_params), set(expected_params))
        # Check if the parameters are as expected
        for param_name, param in expected_params.items():
            if param_name in NO_IDENTITY_CHECK:
                if not isinstance(param, type):
                    raise ValueError(f"{param_name} should be a type.")
                self.assertIsInstance(orig_params[param_name], param)
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
        chemprop_model = get_model()
        cloned_model = clone(chemprop_model)
        self.assertIsInstance(cloned_model, ChempropModel)
        cloned_model_params = cloned_model.get_params(deep=True)
        for param_name, param in chemprop_model.get_params(deep=True).items():
            cloned_param = cloned_model_params[param_name]
            if hasattr(param, "get_params"):
                self.assertEqual(param.__class__, cloned_param.__class__)
                self.assertNotEqual(id(param), id(cloned_param))
            elif isinstance(param, pl.Trainer):
                self.assertIsInstance(cloned_param, pl.Trainer)
            elif isinstance(param, LossFunction):
                self.assertEqual(
                    param.state_dict()["task_weights"],
                    cloned_param.state_dict()["task_weights"],
                )
                self.assertEqual(type(param), type(cloned_param))
            elif isinstance(param, nn.Identity):
                self.assertEqual(type(param), type(cloned_param))
            else:
                self.assertEqual(param, cloned_param, f"Test failed for {param_name}")

    def test_classifier_methods(self) -> None:
        """Test the classifier methods."""
        chemprop_model = get_model()
        # pylint: disable=protected-access
        self.assertTrue(chemprop_model._is_binary_classifier())
        self.assertFalse(chemprop_model._is_multiclass_classifier())
        # pylint: enable=protected-access
        self.assertTrue(hasattr(chemprop_model, "predict_proba"))

    def test_neural_fp(self) -> None:
        """Test the to_encoder method."""
        chemprop_model = get_model()
        neural_fp = chemprop_model.to_encoder()
        self.assertIsInstance(neural_fp, ChempropNeuralFP)
        self.assertIsInstance(neural_fp.model, MPNN)
        # the model should be cloned
        self.assertNotEqual(id(chemprop_model.model), id(neural_fp.model))
        self.assertEqual(neural_fp.disable_fitting, True)


class TestChempropClassifier(unittest.TestCase):
    """Test the Chemprop classifier model."""

    def test_get_params(self) -> None:
        """Test the get_params and set_params methods."""
        chemprop_model = ChempropClassifier()
        param_dict = chemprop_model.get_params(deep=True)
        expected_params = dict(DEFAULT_PARAMS)  # Shallow copy
        self.assertSetEqual(set(param_dict.keys()), set(expected_params.keys()))
        for param_name, param in expected_params.items():
            if param_name in NO_IDENTITY_CHECK:
                if not isinstance(param, type):
                    raise ValueError(f"{param_name} should be a type.")
                self.assertIsInstance(param_dict[param_name], param)
            else:
                self.assertEqual(
                    param_dict[param_name], param, f"Test failed for {param_name}"
                )


class TestChempropRegressor(unittest.TestCase):
    """Test the Chemprop regressor model."""

    def test_get_params(self) -> None:
        """Test the get_params and set_params methods."""
        chemprop_model = ChempropRegressor()
        param_dict = chemprop_model.get_params(deep=True)
        expected_params = dict(DEFAULT_PARAMS)
        expected_params["model__predictor"] = RegressionFFN
        expected_params["model__predictor__criterion"] = MSELoss
        self.assertSetEqual(set(param_dict.keys()), set(expected_params.keys()))
        for param_name, param in expected_params.items():
            if param_name in NO_IDENTITY_CHECK:
                if not isinstance(param, type):
                    raise ValueError(f"{param_name} should be a type.")
                self.assertIsInstance(param_dict[param_name], param)
            else:
                self.assertEqual(
                    param_dict[param_name], param, f"Test failed for {param_name}"
                )
