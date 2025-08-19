"""Tests for the GhostClassificationThreshold class."""

import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from molpipeline import Pipeline, PostPredictionWrapper
from molpipeline.estimators.ghost_classification_threshold import (
    Ghost,
    GhostPostPredictionWrapper,
)


class TestGhostClassificationThreshold(unittest.TestCase):
    """Tests for the GhostClassificationThreshold class."""

    def setUp(self):
        """Set up test data and objects."""
        # Create a simple binary classification dataset
        self.X, self.y = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        # Create a simple classifier
        self.clf = LogisticRegression(random_state=42)
        self.clf.fit(self.X_train, self.y_train)
        self.y_pred_train = self.clf.predict_proba(self.X_train)

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        ghost_clf = Ghost()
        self.assertEqual(ghost_clf.optimization_metric, "Kappa")
        self.assertEqual(
            ghost_clf.thresholds, list(np.round(np.arange(0.05, 0.55, 0.05), 2))
        )
        self.assertIsNone(ghost_clf.decision_threshold)

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        ghost_clf = Ghost(
            thresholds=thresholds, optimization_metric="ROC", random_state=42
        )
        self.assertEqual(ghost_clf.optimization_metric, "ROC")
        self.assertEqual(ghost_clf.thresholds, thresholds)
        self.assertIsNotNone(ghost_clf.random_seed)

    def test_invalid_thresholds(self):
        """Test initialization with invalid thresholds."""
        # Test with invalid thresholds (outside [0,1])
        with self.assertRaises(ValueError):
            Ghost(thresholds=[-0.1, 0.5, 1.2])

        # Test with duplicated thresholds
        with self.assertRaises(ValueError):
            Ghost(thresholds=[0.1, 0.5, 0.1])

    def test_invalid_optimization_metric(self):
        """Test initialization with invalid optimization metric."""
        with self.assertRaises(ValueError):
            Ghost(optimization_metric="Invalid")

    def test_fit_and_transform_interface(self) -> None:
        """Test basic usage of Ghost."""
        estimator = Ghost()

        # Call fit and check decision threshold is set
        self.assertIsNone(estimator.decision_threshold)
        estimator.fit(self.y_pred_train, self.y_train)
        self.assertIsInstance(estimator.decision_threshold, float)
        self.assertTrue(0 <= estimator.decision_threshold <= 1)

        # Call transform and check output
        y_pred_transformed = estimator.transform(self.y_pred_train)
        self.assertIsInstance(y_pred_transformed, np.ndarray)
        self.assertEqual(y_pred_transformed.shape, (len(self.y_train),))
        self.assertTrue(np.all(np.isin(y_pred_transformed, [0, 1])))

    def test_fit_transform_interface(self) -> None:
        """Test basic usage of Ghost."""
        estimator = Ghost()

        # Call fit_transform and check results and decision threshold
        self.assertIsNone(estimator.decision_threshold)
        y_pred_transformed = estimator.fit_transform(self.y_pred_train, self.y_train)
        self.assertIsInstance(estimator.decision_threshold, float)
        self.assertTrue(0 <= estimator.decision_threshold <= 1)
        self.assertIsInstance(y_pred_transformed, np.ndarray)
        self.assertEqual(y_pred_transformed.shape, (len(self.y_train),))
        self.assertTrue(np.all(np.isin(y_pred_transformed, [0, 1])))

    def test_transform_without_fit(self):
        """Test error when predicting without fitting first."""
        ghost_clf = Ghost()
        with self.assertRaises(ValueError):
            ghost_clf.transform(self.X_test)

    def test_pipeline_using_prediction_wrapper(self):
        """Test integration with Pipeline."""
        pipeline = Pipeline(
            [
                ("clf", LogisticRegression(random_state=42)),
                (
                    "clf_threshold",
                    PostPredictionWrapper(
                        Ghost(
                            random_state=42,
                        )
                    ),
                ),
            ]
        )

        # Fit and predict
        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)

        print()
