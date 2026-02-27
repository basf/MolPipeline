"""Unit tests for StratifiedRegressionKFold splitter."""

import unittest

import numpy as np

from molpipeline.experimental.model_selection.splitter.stratified_regression import (
    PercentileStratifiedKFold,
)


class TestPercentileStratifiedKFold(unittest.TestCase):
    """Unit test for the PercentileStratifiedKFold splitter."""

    def test_stratification(self) -> None:
        """Test that the stratification creates balanced folds."""
        rng = np.random.default_rng(42)
        feature_mat = rng.random(size=(100, 10))
        y = rng.uniform(low=0.1, high=0.9, size=100)
        # The first batch of 10 is between 0.1 and 0.9, the next between 1 and 9, etc.
        for i in range(1, 10):
            y[i * 10 :] *= 10

        splitter = PercentileStratifiedKFold(n_splits=5, n_groups=10, random_state=42)

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

    def test_stratification_deterministic(self) -> None:
        """Test that the stratification creates balanced folds and is deterministic.

        Defining 5 groups for the values 1-10, results in the groups [1, 2], [3, 4], ...
        Using n_splits=2, results in each fold containing 1 value from each group.

        """
        y = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        expected_groups = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        rng = np.random.default_rng(42)
        feature_mat = rng.random(size=(10, 10))
        splitter = PercentileStratifiedKFold(n_splits=2, n_groups=5, random_state=42)
        splits = list(splitter.split(feature_mat, y))
        for _, test_idx in splits:
            groups_test = expected_groups[test_idx]
            values, counts = np.unique(groups_test, return_counts=True)
            # Check that all groups are represented and are only represented once
            self.assertListEqual(sorted(values.tolist()), list(range(5)))
            self.assertTrue(np.all(counts == 1))

        # Explicitly check the indices
        expected_splits = [
            (np.array([0, 2, 3, 4, 6]), np.array([1, 5, 7, 8, 9])),
            (np.array([1, 5, 7, 8, 9]), np.array([0, 2, 3, 4, 6])),
        ]
        self.assertTrue(np.array_equal(splits, expected_splits))

    def test_get_n_splits(self) -> None:
        """Test that the get_n_splits method returns the correct number of splits."""
        splitter = PercentileStratifiedKFold(n_splits=42, n_groups=10)
        self.assertEqual(splitter.get_n_splits(), 42)

    def test_error_too_few_unique_y_values(self) -> None:
        """Test that a Error is raised if n_groups is too large for y.

        There can be at most as many groups as there are unique target values, otherwise
        the stratification cannot be performed.

        """
        y = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        x = np.random.default_rng(42).random((6, 2))
        splitter = PercentileStratifiedKFold(n_splits=2, n_groups=10)
        expected_error_msg = (
            r"n_groups \(10\) is greater than the number of unique "
            r"target values \(3\)"
        )
        with self.assertRaisesRegex(ValueError, expected_error_msg):
            list(splitter.split(x, y))

    def test_many_identical_y_values(self) -> None:
        """Test that an error is raised if there are too manz identical y values."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0] + [6.0] * 5)
        x = np.random.default_rng(42).random((10, 2))
        splitter = PercentileStratifiedKFold(n_splits=5, n_groups=3)
        expceted_error_msg = (
            "Too many identical values in y for the specified number of groups!"
        )
        with self.assertRaisesRegex(ValueError, expceted_error_msg):
            list(splitter.split(x, y))

    def test_nan_handling(self) -> None:
        """Test that NaN values in y are handled appropriately."""
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        x = np.random.default_rng(42).random((5, 2))
        splitter = PercentileStratifiedKFold(n_splits=2, n_groups=3)
        expected_error_msg = "Input y contains NaN"
        with self.assertRaisesRegex(ValueError, expected_error_msg):
            list(splitter.split(x, y))
