"""Test Chemprop component wrapper."""

import unittest

from chemprop.nn.loss import LossFunction
from sklearn.base import clone
from torch import nn

from molpipeline.estimators.chemprop.component_wrapper import (
    MPNN,
    BinaryClassificationFFN,
    BondMessagePassing,
    MeanAggregation,
    SumAggregation,
)


class BinaryClassificationFFNTest(unittest.TestCase):
    """Test the BinaryClassificationFFN class."""

    def test_get_set_params(self) -> None:
        """Test the get_params and set_params methods."""
        binary_clf_ffn = BinaryClassificationFFN()
        orig_params = binary_clf_ffn.get_params(deep=True)
        new_params = {
            "activation": "relu",
            "dropout": 0.5,
            "hidden_dim": 400,
            "input_dim": 300,
            "n_layers": 2,
            "n_tasks": 1,
        }
        # Check setting new parameters
        binary_clf_ffn.set_params(**new_params)
        model_params = binary_clf_ffn.get_params(deep=True)
        for param_name, param in new_params.items():
            self.assertEqual(param, model_params[param_name])

        # Check setting original parameters
        binary_clf_ffn.set_params(**orig_params)
        model_params = binary_clf_ffn.get_params(deep=True)
        for param_name, param in orig_params.items():
            self.assertEqual(param, model_params[param_name])


class BondMessagePassingTest(unittest.TestCase):
    """Test the BondMessagePassing class."""

    def test_get_set_params(self) -> None:
        """Test the get_params and set_params methods."""
        bond_message_passing = BondMessagePassing()
        orig_params = bond_message_passing.get_params(deep=True)
        new_params = {
            "activation": "relu",
            "bias": True,
            "d_e": 14,
            "d_h": 300,
            "d_v": 133,
            "d_vd": None,
            "depth": 4,
            "dropout_rate": 0.5,
            "undirected": False,
        }
        # Check setting new parameters
        bond_message_passing.set_params(**new_params)
        model_params = bond_message_passing.get_params(deep=True)
        for param_name, param in new_params.items():
            self.assertEqual(param, model_params[param_name])

        # Check setting original parameters
        bond_message_passing.set_params(**orig_params)
        model_params = bond_message_passing.get_params(deep=True)
        for param_name, param in orig_params.items():
            self.assertEqual(param, model_params[param_name])


class MPNNTest(unittest.TestCase):
    """Test the MPNN class."""

    def test_get_set_params(self) -> None:
        """Test the get_params and set_params methods."""
        mpnn1 = MPNN(
            message_passing=BondMessagePassing(depth=2),
            agg=SumAggregation(),
            predictor=BinaryClassificationFFN(n_layers=1),
        )
        params1 = mpnn1.get_params(deep=True)

        mpnn2 = MPNN(
            message_passing=BondMessagePassing(depth=1),
            agg=MeanAggregation(),
            predictor=BinaryClassificationFFN(n_layers=4),
        )
        mpnn2.set_params(**params1)
        for param_name, param in mpnn1.get_params(deep=True).items():
            param2 = mpnn2.get_params(deep=True)[param_name]
            # Classes are cloned, so they are not equal, but they should be the same class
            # Since (here) objects are identical if their parameters are identical, and since all
            # their parameters are listed flat in the params dicts, all objects are identical if
            # param dicts are identical.
            if hasattr(param, "get_params"):
                self.assertEqual(param.__class__, param2.__class__)
            else:
                self.assertEqual(param, param2)

    def test_clone(self) -> None:
        """Test the clone method."""
        mpnn = MPNN(
            message_passing=BondMessagePassing(),
            agg=SumAggregation(),
            predictor=BinaryClassificationFFN(),
        )
        mpnn_clone = clone(mpnn)
        for param_name, param in mpnn.get_params(deep=True).items():
            clone_param = mpnn_clone.get_params(deep=True)[param_name]
            if hasattr(param, "get_params"):
                self.assertEqual(param.__class__, clone_param.__class__)
            elif isinstance(param, LossFunction):
                self.assertEqual(
                    param.state_dict()["task_weights"],
                    clone_param.state_dict()["task_weights"],
                )
                self.assertEqual(type(param), type(clone_param))
            elif isinstance(param, nn.Identity):
                self.assertEqual(type(param), type(clone_param))
            else:
                self.assertEqual(param, clone_param)


if __name__ == "__main__":
    unittest.main()
