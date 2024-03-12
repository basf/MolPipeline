"""Test Chemprop component wrapper."""

import unittest

from sklearn.base import clone

from molpipeline.sklearn_estimators.chemprop.component_wrapper import (
    MPNN,
    BinaryClassificationFFN,
    BondMessagePassing,
    SumAggregation,
)


class BinaryClassificationFFNTest(unittest.TestCase):
    """Test the BinaryClassificationFFN class."""

    def test_get_set_params(self):
        """Test the get_params and set_params methods."""
        binary_clf_ffn = BinaryClassificationFFN()
        params = binary_clf_ffn.get_params(deep=True)
        binary_clf_ffn.set_params(**params)


class MPNNTest(unittest.TestCase):
    """Test the MPNN class."""

    def test_get_set_params(self):
        """Test the get_params and set_params methods."""
        mpnn = MPNN(
            message_passing=BondMessagePassing(),
            agg=SumAggregation(),
            predictor=BinaryClassificationFFN(),
        )
        params = mpnn.get_params(deep=True)
        mpnn.set_params(**params)

    def test_clone(self):
        """Test the clone method."""
        mpnn = MPNN(
            message_passing=BondMessagePassing(),
            agg=SumAggregation(),
            predictor=BinaryClassificationFFN(),
        )
        print(mpnn.get_params(deep=True))
        mpnn_clone = clone(mpnn)
        self.assertEqual(mpnn, mpnn_clone)


if __name__ == "__main__":
    unittest.main()
