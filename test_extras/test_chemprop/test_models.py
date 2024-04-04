"""Test Chemprop component wrapper."""

import logging
import unittest

from sklearn.base import clone

try:
    from lightning import pytorch as pl

    from molpipeline.estimators.chemprop.component_wrapper import (
        MPNN,
        BinaryClassificationFFN,
        BondMessagePassing,
        MeanAggregation,
        SumAggregation,
    )
    from molpipeline.estimators.chemprop.models import ChempropModel
    from molpipeline.estimators.chemprop.neural_fingerprint import ChempropNeuralFP

    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False


def get_model() -> "ChempropModel":
    """Get the Chemprop model.

    Return type is a string to avoid errors when Chemprop is not available.

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


class TestChempropModel(unittest.TestCase):
    """Test the Chemprop model."""

    def test_get_params(self) -> None:
        """Test the get_params and set_params methods."""
        chemprop_model = get_model()
        oring_params = chemprop_model.get_params(deep=True)

        expected_params = {
            "batch_size": 64,
            "lightning_trainer": pl.Trainer,
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
            "model__message_passing__d_v": 133,
            "model__message_passing__d_vd": None,
            "model__message_passing__depth": 3,
            "model__message_passing__dropout_rate": 0.0,
            "model__message_passing__undirected": False,
            "model__message_passing": BondMessagePassing,
        }

        # Check if the parameters are as expected
        for param_name, param in expected_params.items():
            if param_name in [
                "model__agg",
                "model__message_passing",
                "lightning_trainer",
            ]:
                if not isinstance(param, type):
                    raise ValueError(f"{param_name} should be a type.")
                self.assertIsInstance(oring_params[param_name], param)
                continue
            self.assertEqual(oring_params[param_name], param)

    def test_set_params(self) -> None:
        """Test the set_params method."""
        chemprop_model = get_model()

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
            else:
                self.assertEqual(param, cloned_param)

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
