"""Unit tests for CalibratedClassifierCV with emphasis on class_weight.

Tests use a small imbalanced dataset.
"""

import unittest

from loguru import logger
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, accuracy_score
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

        n_c0 = 900
        n_c1 = 100

        self.selectivity = 0.8
        self.sensitivity = 0.8

        # To reach desired selectivity/sensitivity, we need to adjust the number of
        # samples before flipping labels
        # We sample from class 0: True negatives + False positives
        # We sample from class 1: True positives + False negatives

        true_pos = int(n_c1 * self.sensitivity)
        false_pos = int(n_c1 - true_pos)
        true_neg = int(n_c0 * self.selectivity)
        false_neg = int(n_c0 - true_neg)

        logger.debug(
            f"TP: {true_pos}, FN: {false_neg}, TN: {true_neg}, FP: {false_pos}"
        )

        n_c0_sample = true_neg + false_pos
        n_c1_sample = true_pos + false_neg

        sample_weight = n_c0_sample / (n_c0_sample + n_c1_sample)
        x, y = make_classification(
            n_samples=n_c0_sample + n_c1_sample,
            n_features=1,
            n_informative=1,
            n_redundant=0,
            class_sep=2,
            flip_y=0,
            n_clusters_per_class=1,
            weights=[sample_weight, 1 / sample_weight],
            random_state=SEED,
        )

        logger.debug(
            f"Before flipping: Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}"
        )
        rng = np.random.default_rng(SEED)
        flip_c0_indices = rng.choice(np.where(y == 0)[0], size=false_pos, replace=False)
        flip_c1_indices = rng.choice(np.where(y == 1)[0], size=false_neg, replace=False)
        y[flip_c0_indices] = 1
        y[flip_c1_indices] = 0
        logger.debug(
            f"Flipped {false_pos} labels from class 0 to 1 and "
            f"{false_neg} labels from class 1 to 0."
        )
        logger.debug(
            f"After flipping: Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}"
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
                clf = LogisticRegression(
                    random_state=SEED,
                    class_weight=None,
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
                sensitivity = recall_score(self.y_test, preds, pos_label=1)
                selectivity = recall_score(self.y_test, preds, pos_label=0)
                ba = balanced_accuracy_score(self.y_test, preds)
                logger.info(
                    f"Params: {params}, Balanced Accuracy: {ba:.3f}, "
                    f"Sensitivity: {sensitivity:.3f}, Selectivity: {selectivity:.3f}",
                )

                self.assertGreater(ba, (self.sensitivity + self.selectivity) / 2 - 0.05)
                self.assertGreater(sensitivity, self.sensitivity - 0.05)
                self.assertGreater(selectivity, self.selectivity - 0.05)
                self.assertLess(abs(sensitivity - selectivity), 0.15)

    def test_calibrated_classifiercv_without_class_weight(self) -> None:
        """Test CalibratedClassifierCV without class_weight on imbalanced data.

        Since we do not account for class imbalance, we expect that sensitivity, which
        measures performance on the minority class, will be lower than selectivity.

        """
        clf = LogisticRegression(random_state=SEED, class_weight=None)
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

        ba = balanced_accuracy_score(self.y_test, preds)
        ac = accuracy_score(self.y_test, preds)
        sensitivity = recall_score(self.y_test, preds, pos_label=1)
        selectivity = recall_score(self.y_test, preds, pos_label=0)
        logger.debug(
            f"Without class_weight - Balanced Accuracy: "
            f"{balanced_accuracy_score(self.y_test, preds):.3f}, "
            f"Accuracy: {ac:.3f}, "
            f"Sensitivity: {sensitivity:.3f}, Selectivity: {selectivity:.3f}",
        )
        self.assertGreater(ac, (self.sensitivity + self.selectivity) / 2 - 0.1)
        self.assertLess(ba, (self.sensitivity + self.selectivity) / 2 - 0.1)
        self.assertLess(sensitivity, self.sensitivity - 0.1)
        self.assertGreater(selectivity, self.selectivity - 0.1)
        self.assertGreater(selectivity - sensitivity, 0.1)


if __name__ == "__main__":
    unittest.main()
