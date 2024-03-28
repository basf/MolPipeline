"""Test the ignore error scorer wrapper."""

import unittest

import numpy as np

from molpipeline.metrics import ignored_value_scorer


class IgnoreErrorScorerTest(unittest.TestCase):
    """Test the ignore error scorer wrapper."""

    def test_filter_nan(self) -> None:
        """Test that filtering np.nan works."""
        y_true = np.array([1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, np.nan])
        ba_score = ignored_value_scorer("balanced_accuracy", np.nan)
        value = ba_score._score_func(y_true, y_pred)  # pylint: disable=protected-access
        self.assertAlmostEqual(value, 1.0)

    def test_filter_none(self) -> None:
        """Test that filtering None works."""
        y_true = np.array([1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, None])
        ba_score = ignored_value_scorer("balanced_accuracy", None)
        value = ba_score._score_func(y_true, y_pred)  # pylint: disable=protected-access
        self.assertAlmostEqual(value, 1.0)

    def test_filter_nan_with_none(self) -> None:
        """Test that filtering NaN with None works."""
        y_true = np.array([1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, None])
        ba_score = ignored_value_scorer("balanced_accuracy", np.nan)
        self.assertAlmostEqual(
            ba_score._score_func(y_true, y_pred),  # pylint: disable=protected-access
            1.0,
        )

    def test_filter_none_with_nan(self) -> None:
        """Test that filtering None with NaN works."""
        y_true = np.array([1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, np.nan])
        ba_score = ignored_value_scorer("balanced_accuracy", None)
        self.assertAlmostEqual(
            ba_score._score_func(y_true, y_pred),  # pylint: disable=protected-access
            1.0,
        )
