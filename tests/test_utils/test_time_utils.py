"""Unit test for time_utils module."""

import unittest

import numpy as np
import pandas as pd

from molpipeline.utils.time_utils import floor_timestamp, split_intervals


class TestFloorTimestamp(unittest.TestCase):
    """Tests for round_timestamp function."""

    def setUp(self) -> None:
        """Set up test variables."""
        self.timestamp = pd.Timestamp("2024-06-15 12:34:56")

    def test_round_to_day(self) -> None:
        """Test rounding to day."""
        expected = pd.Timestamp("2024-06-15")
        self.assertEqual(floor_timestamp(self.timestamp, "day"), expected)

    def test_round_to_hour(self) -> None:
        """Test rounding to hour."""
        expected = pd.Timestamp("2024-06-15 12:00:00")
        self.assertEqual(floor_timestamp(self.timestamp, "hour"), expected)

    def test_round_to_month(self) -> None:
        """Test rounding to month."""
        expected = pd.Timestamp("2024-06-01")
        self.assertEqual(floor_timestamp(self.timestamp, "month"), expected)

    def test_no_rounding(self) -> None:
        """Test no rounding when round_to is None."""
        self.assertEqual(floor_timestamp(self.timestamp, None), self.timestamp)

    def test_invalid_round_to(self) -> None:
        """Test that invalid round_to value raises ValueError."""
        with self.assertRaises(ValueError):
            floor_timestamp(self.timestamp, "invalid_option")


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
