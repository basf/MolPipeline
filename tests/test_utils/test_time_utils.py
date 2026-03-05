"""Unit test for time_utils module."""

import unittest

import numpy as np
import pandas as pd

from molpipeline.utils.time_utils import (
    resolve_named_time_stamps,
    split_intervals,
    timestamp_to_group,
)


class TestSplitIntervals(unittest.TestCase):
    """Tests for split_intervals function."""

    def test_basic_split(self) -> None:
        """Test splitting a 5-day range into 5 intervals."""
        start = pd.Timestamp("2024-01-01 00:00:00")
        end = pd.Timestamp("2024-01-6 00:00:00")
        n_intervals = 5
        thresholds = split_intervals(start, end, n_intervals)
        expected_thresholds = [
            "2024-01-02 00:00:00",
            "2024-01-03 00:00:00",
            "2024-01-04 00:00:00",
            "2024-01-05 00:00:00",
        ]
        # Fencepost error: 4 thresholds -> 5 intervals
        self.assertEqual(n_intervals, len(thresholds) + 1)
        self.assertTrue(np.array_equal(thresholds, pd.to_datetime(expected_thresholds)))


class TestTimestampToGroup(unittest.TestCase):
    """Tests for timestamp_to_group function."""

    def setUp(self) -> None:
        """Set up common test data."""
        self.groups = pd.to_datetime(
            [
                "2024-01-01 12:00:00",  # Group 0
                "2024-01-02 12:00:00",  # Group 1
                "2024-01-03 12:00:00",  # Group 2
                "2024-01-04 12:00:00",  # Group 3
            ],
        )
        self.threshold_list = pd.to_datetime(
            [
                "2024-01-02 00:00:00",
                "2024-01-03 00:00:00",
                "2024-01-04 00:00:00",
            ],
        )

    def test_basic_grouping(self) -> None:
        """Test grouping timestamps based on thresholds."""
        group_indices = timestamp_to_group(self.groups, self.threshold_list)
        expected_indices = np.array([0, 1, 2, 3])
        self.assertTrue(np.array_equal(group_indices, expected_indices))

    def test_shuffle_groups(self) -> None:
        """Test that the function correctly groups timestamps regardless of order."""
        shuffle_order = [2, 0, 3, 1]  # New order of indices
        shuffled_groups = self.groups[shuffle_order]  # Shuffle the order
        group_indices = timestamp_to_group(shuffled_groups, self.threshold_list)
        self.assertTrue(np.array_equal(group_indices, np.array(shuffle_order)))

    def test_shuffle_threshold_list(self) -> None:
        """Test that the function correctly groups regardless of threshold order."""
        shuffle_order = [2, 0, 1]  # New order of indices
        shuffled_threshold_list = self.threshold_list[shuffle_order]
        group_indices = timestamp_to_group(self.groups, shuffled_threshold_list)
        self.assertTrue(np.array_equal(group_indices, np.array([0, 1, 2, 3])))

    def test_shuffle_groups_and_thresholds(self) -> None:
        """Test that the function correctly groups regardless of both orders."""
        group_shuffle_order = [2, 0, 3, 1]  # New order of group indices
        threshold_shuffle_order = [2, 0, 1]  # New order of threshold indices
        shuffled_groups = self.groups[group_shuffle_order]
        shuffled_threshold_list = self.threshold_list[threshold_shuffle_order]
        group_indices = timestamp_to_group(shuffled_groups, shuffled_threshold_list)
        self.assertTrue(np.array_equal(group_indices, np.array(group_shuffle_order)))


class TestResolveSpecialTimeStrings(unittest.TestCase):
    """Tests for resolve_special_time_strings function."""

    def test_resolve_special_time_strings(self) -> None:
        """Test that special time strings are resolved correctly."""
        now = pd.Timestamp.now()
        test_cases = {
            "now": now,
            "today": now.normalize(),
            "Q1": pd.Timestamp(year=now.year, month=1, day=1),
            "Q2": pd.Timestamp(year=now.year, month=4, day=1),
            "Q3": pd.Timestamp(year=now.year, month=7, day=1),
            "Q4": pd.Timestamp(year=now.year, month=10, day=1),
        }
        for input_str, expected in test_cases.items():
            resolved = resolve_named_time_stamps(input_str)  # type: ignore
            if input_str == "now":
                # Allow for a small time difference due to execution time
                self.assertTrue(abs(resolved - expected) < pd.Timedelta(seconds=1))
            else:
                self.assertEqual(resolved, expected)
