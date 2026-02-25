"""Unit test for time_utils module."""

import unittest

import pandas as pd

from molpipeline.utils.time_utils import floor_timestamp


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
