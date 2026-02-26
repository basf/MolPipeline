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

    def test_get_n_splits(self) -> None:
        """Test that the get_n_splits method returns the correct number of splits."""
        splitter = StratifiedRegressionKFold(n_splits=42, n_groups=10)
        self.assertEqual(splitter.get_n_splits(), 42)

    def test_error_too_few_unique_y_values(self) -> None:
        """Test that a Error is raised if n_groups is too large for y.

        There can be at most as many groups as there are unique target values, otherwise
        the stratification cannot be performed.

        """
        y = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        x = np.random.default_rng(42).random((6, 2))
        splitter = StratifiedRegressionKFold(n_splits=2, n_groups=10)
        expected_error_msg = (
            r"n_groups \(10\) is greater than the number of unique "
            r"target values \(3\)"
        )
        with self.assertRaisesRegex(ValueError, expected_error_msg):
            list(splitter.split(x, y))

    def test_many_identical_y_values(self) -> None:
        """Test that the splitter can handle many identical y values."""
        y = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0] + [3.0] * 100)
        x = np.random.default_rng(42).random((106, 2))
        splitter = StratifiedRegressionKFold(n_splits=5, n_groups=3)
        splits = list(splitter.split(x, y))
        self.assertEqual(len(splits), 5)

    def test_nan_handling(self) -> None:
        """Test that NaN values in y are handled appropriately."""
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        x = np.random.default_rng(42).random((5, 2))
        splitter = StratifiedRegressionKFold(n_splits=2, n_groups=3)
        expected_error_msg = "Input y contains NaN"
        with self.assertRaisesRegex(ValueError, expected_error_msg):
            list(splitter.split(x, y))
