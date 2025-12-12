"""Unit tests for CalibratedClassifierCV with emphasis on class_weight.

Tests use a small imbalanced dataset.
"""

import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from molpipeline.estimators.calibration.calibrated_classifier import (
    CalibratedClassifierCV,
)

SEED = 42


class TestCalibratedClassifierCV(unittest.TestCase):
    """Unit tests for CalibratedClassifierCV with emphasis on class_weight.

    Tests use a small imbalanced dataset.
    """

    def setUp(self) -> None:
        """Set up any necessary components before each test."""
        x, y = make_classification(
            n_samples=1000,
            n_features=3,
            n_informative=3,
            n_redundant=0,
            n_clusters_per_class=3,
            weights=[0.9, 0.1],
            random_state=SEED,
        )
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=SEED,
            stratify=y,
        )

    def test_calibrated_classifiercv_with_class_weight(self) -> None:
        """Test CalibratedClassifierCV with class_weight on imbalanced data.

        Since we account for class imbalance, we expect sensitivity and selectivity
        to be reasonably balanced and both above 0.9.
        This should pass consistently for all parameter combinations tested here.

        """
        classes = np.array([0, 1])
        class_weight_arr = compute_class_weight(
            "balanced",
            y=self.y_train,
            classes=classes,
        )
        class_weight_dict = dict(zip(classes, class_weight_arr, strict=True))
        param_grid = ParameterGrid(
            {
                "class_weight": ["balanced", class_weight_dict],
                "ensemble": [True, False, "auto"],
                "method": ["isotonic", "sigmoid"],
            },
        )
        for params in param_grid:
            with self.subTest(params=params):
                clf = RandomForestClassifier(
                    random_state=SEED,
                    class_weight=params["class_weight"],
                )
                calibrated = CalibratedClassifierCV(
                    clf,
                    cv=2,
                    **params,
                )
                calibrated.fit(self.x_train, self.y_train)
                preds = calibrated.predict(self.x_test)
                probas = calibrated.predict_proba(self.x_test)
                self.assertEqual(preds.shape, (self.x_test.shape[0],))
                self.assertEqual(probas.shape, (self.x_test.shape[0], 2))

                self.assertGreater(balanced_accuracy_score(self.y_test, preds), 0.7)

                sensitivity = recall_score(self.y_test, preds, pos_label=1)
                selectivity = recall_score(self.y_test, preds, pos_label=0)
                self.assertGreater(sensitivity, 0.9)
                self.assertGreater(selectivity, 0.9)
                self.assertLess(abs(sensitivity - selectivity), 0.1)

    def test_calibrated_classifiercv_without_class_weight(self) -> None:
        """Test CalibratedClassifierCV without class_weight on imbalanced data.

        Since we do not account for class imbalance, we expect that sensitivity, which
        measures performance on the minority class, will be lower than selectivity.

        """
        clf = RandomForestClassifier(random_state=SEED)
        calibrated = CalibratedClassifierCV(
            clf,
            cv=2,
            method="isotonic",
            class_weight=None,
            ensemble=False,
        )
        calibrated.fit(self.x_train, self.y_train)
        preds = calibrated.predict(self.x_test)
        probas = calibrated.predict_proba(self.x_test)
        self.assertEqual(preds.shape, (self.x_test.shape[0],))
        self.assertEqual(probas.shape, (self.x_test.shape[0], 2))

        self.assertGreater(balanced_accuracy_score(self.y_test, preds), 0.6)
        sensitivity = recall_score(self.y_test, preds, pos_label=1)
        selectivity = recall_score(self.y_test, preds, pos_label=0)
        self.assertLess(sensitivity, 0.7)
        self.assertGreater(selectivity, 0.9)
        self.assertGreater(selectivity - sensitivity, 0.1)


if __name__ == "__main__":
    unittest.main()
