"""Test the ignore error scorer wrapper."""

import unittest

import numpy as np
from sklearn import linear_model
from sklearn.metrics import get_scorer

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

    def test_correct_init_mse(self) -> None:
        """Test that initialization is correct as we access via protected vars."""
        x_train = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).reshape(
            -1, 1
        )
        y_train = np.array([0.1, 0.3, 0.3, 0.4, 0.5, 0.5, 0.7, 0.88, 0.9, 1])
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)
        cix_scorer = ignored_value_scorer("neg_mean_squared_error", None)
        scikit_scorer = get_scorer("neg_mean_squared_error")
        self.assertEqual(
            cix_scorer(regr, x_train, y_train), scikit_scorer(regr, x_train, y_train)
        )

    def test_correct_init_rmse(self) -> None:
        """Test that initialization is correct as we access via protected vars."""
        x_train = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).reshape(
            -1, 1
        )
        y_train = np.array([0.1, 0.3, 0.3, 0.4, 0.5, 0.5, 0.7, 0.88, 0.9, 1])
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)
        cix_scorer = ignored_value_scorer("neg_root_mean_squared_error", None)
        scikit_scorer = get_scorer("neg_root_mean_squared_error")
        self.assertEqual(
            cix_scorer(regr, x_train, y_train), scikit_scorer(regr, x_train, y_train)
        )

    def test_correct_init_inheritance(self) -> None:
        """Test that initialization is correct if we pass an initialized scorer."""
        x_train = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).reshape(
            -1, 1
        )
        y_train = np.array([0.1, 0.3, 0.3, 0.4, 0.5, 0.5, 0.7, 0.88, 0.9, 1])
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)
        scikit_scorer = get_scorer("neg_root_mean_squared_error")
        cix_scorer = ignored_value_scorer(
            get_scorer("neg_root_mean_squared_error"), None
        )
        self.assertEqual(
            cix_scorer(regr, x_train, y_train), scikit_scorer(regr, x_train, y_train)
        )
