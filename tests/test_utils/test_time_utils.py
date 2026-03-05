"""Unit test for time_utils module."""

import unittest

import numpy as np
import pandas as pd

from molpipeline.utils.time_utils import split_intervals

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
