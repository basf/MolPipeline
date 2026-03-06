"""Test the ignore error scorer wrapper."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn import linear_model
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from molpipeline.metrics import ignored_value_scorer


class IgnoreErrorScorerTest(unittest.TestCase):
    """Test the ignore error scorer wrapper."""

    def test_filter_nan(self) -> None:
        """Test that filtering np.nan works."""
        y_true = np.array([1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, np.nan])
        ba_score = ignored_value_scorer("balanced_accuracy", np.nan)
        value = ba_score._score_func(y_true, y_pred)
        self.assertAlmostEqual(value, 1.0)

    def test_filter_none(self) -> None:
        """Test that filtering None works."""
        y_true = np.array([1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, None])
        ba_score = ignored_value_scorer("balanced_accuracy", None)
        value = ba_score._score_func(y_true, y_pred)
        self.assertAlmostEqual(value, 1.0)

    def test_filter_nan_with_none(self) -> None:
        """Test that filtering NaN with None works."""
        y_true = np.array([1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, None])
        ba_score = ignored_value_scorer("balanced_accuracy", np.nan)
        self.assertAlmostEqual(ba_score._score_func(y_true, y_pred), 1.0)

    def test_filter_none_with_nan(self) -> None:
        """Test that filtering None with NaN works."""
        y_true = np.array([1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, np.nan])
        ba_score = ignored_value_scorer("balanced_accuracy", None)
        self.assertAlmostEqual(ba_score._score_func(y_true, y_pred), 1.0)

    def test_correct_init_mse(self) -> None:
        """Test that initialization is correct as we access via protected vars."""
        x_train = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).reshape(
            -1,
            1,
        )
        y_train = np.array([0.1, 0.3, 0.3, 0.4, 0.5, 0.5, 0.7, 0.88, 0.9, 1])
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)
        cix_scorer = ignored_value_scorer("neg_mean_squared_error", None)
        scikit_scorer = get_scorer("neg_mean_squared_error")
        self.assertEqual(
            cix_scorer(regr, x_train, y_train),
            scikit_scorer(regr, x_train, y_train),
        )

    def test_correct_init_rmse(self) -> None:
        """Test that initialization is correct as we access via protected vars."""
        x_train = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).reshape(
            -1,
            1,
        )
        y_train = np.array([0.1, 0.3, 0.3, 0.4, 0.5, 0.5, 0.7, 0.88, 0.9, 1])
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)
        cix_scorer = ignored_value_scorer("neg_root_mean_squared_error", None)
        scikit_scorer = get_scorer("neg_root_mean_squared_error")
        self.assertEqual(
            cix_scorer(regr, x_train, y_train),
            scikit_scorer(regr, x_train, y_train),
        )

    def test_correct_init_inheritance(self) -> None:
        """Test that initialization is correct if we pass an initialized scorer."""
        x_train = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).reshape(
            -1,
            1,
        )
        y_train = np.array([0.1, 0.3, 0.3, 0.4, 0.5, 0.5, 0.7, 0.88, 0.9, 1])
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)
        scikit_scorer = get_scorer("neg_root_mean_squared_error")
        cix_scorer = ignored_value_scorer(
            get_scorer("neg_root_mean_squared_error"),
            None,
        )
        self.assertEqual(
            cix_scorer(regr, x_train, y_train),
            scikit_scorer(regr, x_train, y_train),
        )

    def test_pickle_roundtrip(self) -> None:
        """Test that ignored_value_scorer result can be pickled and unpickled."""
        original_f = ignored_value_scorer(get_scorer("balanced_accuracy"), np.nan)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "scorer.joblib"
            joblib.dump(original_f, path)
            unpickled_f = joblib.load(path)

        x = np.ones((5, 1))
        y_true = np.array([1, 0, 0, 1, 0])
        estimator = linear_model.LogisticRegression()
        estimator.fit(x, y_true)
        original_score = original_f(estimator, x, y_true)
        unpickled_score = unpickled_f(estimator, x, y_true)
        self.assertAlmostEqual(original_score, unpickled_score)

    def test_grid_search_cv(self) -> None:
        """Test that ignored_value_scorer works as scoring in GridSearchCV."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((50, 5))
        y = (x[:, 0] > 0).astype(int)

        scorer = ignored_value_scorer("balanced_accuracy", np.nan)

        grid_search = GridSearchCV(
            estimator=SVC(),
            param_grid={"C": [0.1, 1.0]},
            scoring=scorer,
            cv=2,
            n_jobs=1,
        )
        grid_search.fit(x, y)

        self.assertIn("mean_test_score", grid_search.cv_results_)
        self.assertIsNotNone(grid_search.best_score_)
        self.assertGreater(grid_search.best_score_, 0.0)

    def test_grid_search_cv_multimetric(self) -> None:
        """Test that ignored_value_scorer works in multimetric GridSearchCV.

        There was a bug with multimetric scoring because GridSearchCV will call
        __name__ on the scorer for its __repr__, and if the scorer is an instance of
        ignored_value_scorer, then _IgnoreValueScoreFunc must have a __name__ attribute.
        """
        rng = np.random.default_rng(42)
        x = rng.standard_normal((50, 5))
        y = (x[:, 0] > 0).astype(int)

        scoring = {
            "ba": ignored_value_scorer("balanced_accuracy", np.nan),
            "f1": ignored_value_scorer("f1", np.nan),
        }

        grid_search = GridSearchCV(
            estimator=SVC(),
            param_grid={"C": [0.1, 1.0]},
            scoring=scoring,
            refit="ba",
            cv=2,
            n_jobs=1,
        )
        grid_search.fit(x, y)

        # Verify that __name__ is correctly forwarded through GridSearchCV's
        # public scorer_ attribute
        self.assertEqual(
            grid_search.scorer_["ba"]._score_func.__name__,
            scoring["ba"]._score_func.__name__,
        )
        self.assertEqual(
            grid_search.scorer_["ba"]._score_func.__name__, "balanced_accuracy_score",
        )
        self.assertEqual(
            grid_search.scorer_["f1"]._score_func.__name__,
            scoring["f1"]._score_func.__name__,
        )
        self.assertEqual(grid_search.scorer_["f1"]._score_func.__name__, "f1_score")

        self.assertIn("mean_test_ba", grid_search.cv_results_)
        self.assertIn("mean_test_f1", grid_search.cv_results_)
        self.assertIsNotNone(grid_search.best_score_)
        self.assertGreater(grid_search.best_score_, 0.0)
