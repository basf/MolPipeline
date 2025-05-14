"""Tests for stochastic filters."""

import unittest

import numpy as np

from molpipeline.estimators.samplers.stochastic_filter import (
    GlobalClassBalanceFilter,
    GroupSizeFilter,
    LocalGroupClassBalanceFilter,
)


class TestGlobalClassBalanceFilter(unittest.TestCase):
    """Test stochastic sampler."""

    def test_global_class_balance_filter_multiclass(self) -> None:
        """Test GlobalClassBalanceFilter with multiple classes."""
        filter_policy = GlobalClassBalanceFilter()
        x_matrix = np.zeros((10, 2))
        y = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 3])

        # Class counts: class 0=2, class 1=4, class 2=3, class 3=1
        probs = filter_policy.calculate_probabilities(x_matrix, y)

        # Calculate expected probabilities:
        # inverse frequencies are [1/2, 1/4, 1/3, 1]
        # sum = 1/2 + 1/4 + 1/3 + 1 = 25/12
        # normalized values are [1/2, 1/4, 1/3, 1] / (25/12) = [6/25, 3/25, 4/25, 12/25]
        expected = np.array(
            [
                6 / 25,
                6 / 25,
                3 / 25,
                3 / 25,
                3 / 25,
                3 / 25,
                4 / 25,
                4 / 25,
                4 / 25,
                12 / 25,
            ],
        )
        self.assertTrue(np.allclose(probs, expected))

    def test_global_class_balance_filter_single_class(self) -> None:
        """Test GlobalClassBalanceFilter with only one class."""
        filter_policy = GlobalClassBalanceFilter()
        x_matrix = np.zeros((5, 2))
        y = np.ones(5)  # All samples belong to class 1

        probs = filter_policy.calculate_probabilities(x_matrix, y)

        # With only one class, all samples should have equal probability of 1
        expected = np.ones(5)
        self.assertTrue(np.allclose(probs, expected))

    def test_global_class_balance_filter_extreme_imbalance(self) -> None:
        """Test GlobalClassBalanceFilter with extremely imbalanced classes."""
        filter_policy = GlobalClassBalanceFilter()
        x_matrix = np.zeros((101, 2))
        y = np.zeros(101)
        y[0] = 1  # Only one sample of class 1

        probs = filter_policy.calculate_probabilities(x_matrix, y)

        # inverse frequencies are [1/100, 1]
        # sum = 1/100 + 1 = 1.01
        expected = np.array([1 / 1.01, *([(1 / 100) / 1.01] * 100)])
        self.assertTrue(np.allclose(probs, expected))


class TestLocalGroupClassBalanceFilter(unittest.TestCase):
    """Test LocalGroupClassBalanceFilter."""

    def test_multiple_groups_multiple_classes(self) -> None:
        """Test probability calculation with multiple groups and classes."""
        # Groups: 3 groups with varying class distribution
        # Group 0: class 0=1, class 1=2  (inv_freq: [1, 1/2])
        # Group 1: class 0=2, class 1=1  (inv_freq: [1/2, 1])
        # Group 2: class 0=1, class 1=1  (inv_freq: [1, 1])
        groups = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        y = np.array([0, 1, 1, 0, 0, 1, 0, 1])

        filter_policy = LocalGroupClassBalanceFilter(groups)
        x_matrix = np.zeros((len(groups), 2))

        probs = filter_policy.calculate_probabilities(x_matrix, y)

        # Expected probabilities after normalization:
        # Total sum of inv_freqs: 1 + 1/2 + 1/2 + 1 + 1 + 1 = 5
        # Group 0: [1/5, 1/10, 1/10]
        # Group 1: [1/10, 1/10, 1/5]
        # Group 2: [1/5, 1/5]
        expected = np.array(
            [1 / 5, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 5, 1 / 5, 1 / 5],
        )
        self.assertTrue(np.allclose(probs, expected))

    def test_single_group_multiple_classes(self) -> None:
        """Test with a single group containing multiple classes."""
        groups = np.array([1, 1, 1, 1])
        y = np.array([0, 0, 1, 2])  # Class 0=2, Class 1=1, Class 2=1

        filter_policy = LocalGroupClassBalanceFilter(groups)
        x_matrix = np.zeros((4, 2))

        probs = filter_policy.calculate_probabilities(x_matrix, y)

        # Expected inverse frequencies are [1/2, 1, 1]
        # Sum = 1/2 + 1 + 1 = 2.5
        # Normalized values are [1/5, 1/5, 2/5, 2/5]
        expected = np.array([1 / 5, 1 / 5, 2 / 5, 2 / 5])
        self.assertTrue(np.allclose(probs, expected))

    def test_multiple_groups_single_class(self) -> None:
        """Test with multiple groups each containing a single class."""
        groups = np.array([1, 1, 2, 2, 3, 3])
        y = np.ones(6)  # All samples are class 1

        filter_policy = LocalGroupClassBalanceFilter(groups)
        x_matrix = np.zeros((6, 2))

        probs = filter_policy.calculate_probabilities(x_matrix, y)

        # Expected inverse frequencies are [1/2, 1/2, 1/2]
        # Sum = 1/2 + 1/2 + 1/2 = 1.5
        # Normalized values are [1/3, 1/3, 1/3, 1/3, 1/3, 1/3]

        # With single class in each group, all samples should have equal probability
        expected = np.ones(6) / 3
        self.assertTrue(np.allclose(probs, expected))

    def test_extreme_imbalance_within_groups(self) -> None:
        """Test with extreme class imbalance within groups."""
        groups = np.array([1, 1, 1, 1, 1, 2, 2, 2])
        y = np.array([0, 0, 0, 0, 1, 0, 0, 1])  # Group 1: 4:1, Group 2: 2:1

        filter_policy = LocalGroupClassBalanceFilter(groups)
        x_matrix = np.zeros((8, 2))

        probs = filter_policy.calculate_probabilities(x_matrix, y)

        # Group 1: inv_freqs = [1/4, 1] = [0.25, 1]
        # Group 2: inv_freqs = [1/2, 1] = [0.5, 1]
        # Sum = 1/4 + 1 + 1/2 + 1 = 2.75
        # Normalized values are [0.25/2.75, 0.25/2.75, 0.25/2.75, 0.25/2.75, 1/2.75,
        #                        0.5/2.75, 0.5/2.75, 1/2.75]
        expected = np.array(
            [
                0.25 / 2.75,
                0.25 / 2.75,
                0.25 / 2.75,
                0.25 / 2.75,
                1 / 2.75,
                0.5 / 2.75,
                0.5 / 2.75,
                1 / 2.75,
            ],
        )
        self.assertTrue(np.allclose(probs, expected))

    def test_error_mismatched_lengths(self) -> None:
        """Test error handling for mismatched length between groups and y."""
        groups = np.array([1, 1, 2, 2])
        filter_policy = LocalGroupClassBalanceFilter(groups)
        x_matrix = np.zeros((5, 2))
        y = np.array([0, 0, 1, 1, 1])

        with self.assertRaises(ValueError):
            filter_policy.calculate_probabilities(x_matrix, y)

    def test_empty_groups(self) -> None:
        """Test handling of empty input."""
        groups = np.array([])
        filter_policy = LocalGroupClassBalanceFilter(groups)
        x_matrix = np.zeros((0, 2))
        y = np.array([])

        probs = filter_policy.calculate_probabilities(x_matrix, y)
        self.assertEqual(len(probs), 0)


class TestGroupSizeFilter(unittest.TestCase):
    """Test GroupSizeFilter."""

    def test_group_size_filter_basic(self) -> None:
        """Test GroupSizeFilter with groups of different sizes."""
        # Groups: size 3, 2, 5
        groups = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 3])
        filter_policy = GroupSizeFilter(groups)
        x_matrix = np.zeros((10, 2))
        y = np.ones(10)  # All samples belong to class 1 but doesn't matter

        probs = filter_policy.calculate_probabilities(x_matrix, y)

        # Expected inverse sizes are [1/3, 1/2, 1/5]
        # Sum = 1/3 + 1/2 + 1/5 = 31/30
        # Normalized values are [1/3, 1/2, 1/5] / (31/30) = [10/31, 15/31, 6/31]
        expected = np.array(
            [
                10 / 31,
                10 / 31,
                10 / 31,
                15 / 31,
                15 / 31,
                6 / 31,
                6 / 31,
                6 / 31,
                6 / 31,
                6 / 31,
            ],
        )
        self.assertTrue(np.allclose(probs, expected))

    def test_group_size_filter_single_group(self) -> None:
        """Test GroupSizeFilter with only one group."""
        groups = np.ones(5)
        filter_policy = GroupSizeFilter(groups)
        x_matrix = np.zeros((5, 2))
        y = np.ones(5)

        probs = filter_policy.calculate_probabilities(x_matrix, y)

        # With one group, all samples should have equal probability
        expected = np.ones(5)
        self.assertTrue(np.allclose(probs, expected))

    def test_group_size_filter_equal_sizes(self) -> None:
        """Test GroupSizeFilter with equally sized groups."""
        groups = np.array([1, 1, 2, 2, 3, 3])
        filter_policy = GroupSizeFilter(groups)
        x_matrix = np.zeros((6, 2))
        y = np.ones(6)

        probs = filter_policy.calculate_probabilities(x_matrix, y)

        # All groups have equal size, so all samples should have equal probability of
        # 1 / 3
        expected = np.zeros(6) + (1 / 3)
        self.assertTrue(np.allclose(probs, expected))

    def test_group_size_filter_extreme_imbalance(self) -> None:
        """Test GroupSizeFilter with extremely imbalanced group sizes."""
        groups = np.zeros(101)
        groups[0] = 1  # One sample in group 1, 100 samples in group 0

        filter_policy = GroupSizeFilter(groups)
        x_matrix = np.ones((101, 2))
        y = np.zeros(101)

        probs = filter_policy.calculate_probabilities(x_matrix, y)

        # inverse frequencies are [1/100, 1]
        # sum = 1/100 + 1 = 1.01
        expected = np.array([1 / 1.01, *([(1 / 100) / 1.01] * 100)])
        self.assertTrue(np.allclose(probs, expected))

    def test_group_size_filter_error_handling(self) -> None:
        """Test GroupSizeFilter error handling for mismatched lengths."""
        groups = np.array([1, 1, 2, 2])
        filter_policy = GroupSizeFilter(groups)
        x_matrix = np.zeros((5, 2))
        y = np.ones(5)

        with self.assertRaises(ValueError):
            filter_policy.calculate_probabilities(x_matrix, y)
