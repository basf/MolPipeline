"""Unit tests for StratifiedRegressionKFold splitter."""

import unittest

import numpy as np

from molpipeline.experimental.model_selection.splitter.stratified_regression import (
    StratifiedRegressionKFold,
)


class TestStratifiedRegressionKFold(unittest.TestCase):
    """Unit test for the functionality of the StratifiedRegressionKFold splitter."""

    def test_stratification(self) -> None:
        """Test that the stratification creates balanced folds."""
        rng = np.random.default_rng(42)
        feature_mat = rng.random(size=(100, 10))
        y = rng.uniform(low=0.1, high=0.9, size=100)
        # The first batch of 10 is between 0.1 and 0.9, the next between 1 and 9, etc.
        for i in range(1, 10):
            y[i * 10 :] *= 10

        splitter = StratifiedRegressionKFold(n_splits=5, n_groups=10, random_state=42)

        for _, test_idx in splitter.split(feature_mat, y):
            y_test = y[test_idx]
            # Check that y contains values from all quantile groups in the test set
            # Since there are 10 groups for 10 orders of magnitude, which are split
            # into 5 folds, we expect each fold to contain values from 2 groups.
            magintues = np.ceil(np.log10(y_test))
            values, counts = np.unique(magintues, return_counts=True)
            # Check that all groups are represented
            self.assertListEqual(sorted(values.tolist()), list(range(10)))
            # Ensure that all counts are 2
            expected_counts = np.full(10, 2)
            self.assertTrue(np.array_equal(counts, expected_counts))
