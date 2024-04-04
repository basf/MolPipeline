"""Test Chemprop component wrapper."""

import unittest

from sklearn.base import clone

try:
    from molpipeline.estimators.chemprop.component_wrapper import (
        MPNN,
        BinaryClassificationFFN,
        BondMessagePassing,
        MeanAggregation,
        SumAggregation,
    )

    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False


class BinaryClassificationFFNTest(unittest.TestCase):
    """Test the BinaryClassificationFFN class."""

    def test_get_set_params(self) -> None:
        """Test the get_params and set_params methods."""
        binary_clf_ffn = BinaryClassificationFFN()
        oring_params = binary_clf_ffn.get_params(deep=True)
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
        for param_name, param in new_params.items():
            self.assertEqual(param, binary_clf_ffn.get_params(deep=True)[param_name])

        # Check setting original parameters
        binary_clf_ffn.set_params(**oring_params)
        for param_name, param in oring_params.items():
            self.assertEqual(param, binary_clf_ffn.get_params(deep=True)[param_name])


class MPNNTest(unittest.TestCase):
    """Test the MPNN class."""

    def test_get_set_params(self) -> None:
        """Test the get_params and set_params methods."""
        mpnn1 = MPNN(
            message_passing=BondMessagePassing(),
            agg=SumAggregation(),
            predictor=BinaryClassificationFFN(),
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
            else:
                self.assertEqual(param, clone_param)


if __name__ == "__main__":
    unittest.main()
