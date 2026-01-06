"""Unit tests for CalibratedClassifierCV with emphasis on class_weight.

Tests use a small imbalanced dataset.
"""

import unittest

import numpy as np
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV as SKCalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils import compute_sample_weight
from sklearn.utils.class_weight import compute_class_weight

from molpipeline.estimators.calibration.calibrated_classifier import (
    CalibratedClassifierCV,
)
from tests.utils.logging import capture_logs

# Parameters for the tests
SEED = 42
TOLERANCE = 0.05


class TestCalibratedClassifierCV(unittest.TestCase):
    """Unit tests for CalibratedClassifierCV with emphasis on class_weight.

    Tests use a small imbalanced dataset.
    """

    def setUp(self) -> None:
        """Set up any necessary components before each test.

        Raises
        ------
        AssertionError
            If generated class distributions do not match expected values.
        AssertionError
            If final class distributions do not match requested values.

        """
        n_c0 = 900  # Class 0 is majority class
        n_c1 = 100  # Class 1 is minority class

        self.selectivity = 0.8
        self.sensitivity = 0.8
        self.expected_ba = (self.selectivity + self.sensitivity) / 2

        # To reach desired selectivity/sensitivity, we need to adjust the number of
        # samples before flipping labels
        # We sample from class 0: True negatives + False positives
        # We sample from class 1: True positives + False negatives

        true_pos = int(n_c1 * self.sensitivity)
        false_pos = int(n_c1 - true_pos)
        true_neg = int(n_c0 * self.selectivity)
        false_neg = int(n_c0 - true_neg)

        logger.debug(
            f"TP: {true_pos}, FN: {false_neg}, TN: {true_neg}, FP: {false_pos}",
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
            weights=[sample_weight, 1 - sample_weight],
            random_state=SEED,
        )

        if not (np.sum(y == 0) == n_c0_sample and np.sum(y == 1) == n_c1_sample):
            raise AssertionError(
                "Generated class distribution does not match expected.",
            )
        rng = np.random.default_rng(SEED)
        flip_c0_indices = rng.choice(np.where(y == 0)[0], size=false_pos, replace=False)
        flip_c1_indices = rng.choice(np.where(y == 1)[0], size=false_neg, replace=False)
        y[flip_c0_indices] = 1
        y[flip_c1_indices] = 0
        if not (np.sum(y == 0) == n_c0 and np.sum(y == 1) == n_c1):
            raise AssertionError(
                "Final class distribution does not match requested distribution.",
            )

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=SEED,
            stratify=y,
        )

    def test_temperature_and_class_weight_warning(self) -> None:
        """Test that warning is raised when using temperature method with class_weight.

        Using class_weight with temperature method seems to have no effect.

        """
        with capture_logs(level="WARNING") as log_capture:
            CalibratedClassifierCV(
                method="temperature",
                class_weight="balanced",
            )
        warning_messages = "\n".join(log_capture)
        self.assertIn(
            "At the moment temperature scaling with class weights seems to have no "
            "effect.",
            warning_messages,
        )

    def test_with_class_weight(self) -> None:
        """Test CalibratedClassifierCV with class_weight on imbalanced data.

        Since we account for class imbalance, we expect sensitivity and selectivity
        to be reasonably balanced and both above 0.8.
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
                "method": ["isotonic", "sigmoid", "temperature"],
            },
        )
        for params in param_grid:
            with self.subTest(params=params):
                clf = LogisticRegression(random_state=SEED, class_weight=None)
                calibrated = CalibratedClassifierCV(clf, cv=2, **params)
                calibrated.fit(self.x_train, self.y_train)
                preds = calibrated.predict(self.x_test)
                probas = calibrated.predict_proba(self.x_test)
                self.assertEqual(preds.shape, (self.x_test.shape[0],))
                self.assertEqual(probas.shape, (self.x_test.shape[0], 2))
                sensitivity = recall_score(self.y_test, preds, pos_label=1)
                selectivity = recall_score(self.y_test, preds, pos_label=0)
                ba = balanced_accuracy_score(self.y_test, preds)
                logger.debug(
                    f"Params: {params}, Balanced Accuracy: {ba:.3f}, "
                    f"Sensitivity: {sensitivity:.3f}, Selectivity: {selectivity:.3f}",
                )
                self.assertGreater(selectivity, self.selectivity - TOLERANCE)

                # Temperature seems to neglect minority class despite class_weight
                if params["method"] != "temperature":
                    self.assertGreater(ba, self.expected_ba - TOLERANCE)
                    self.assertGreater(sensitivity, self.sensitivity - TOLERANCE)
                    self.assertLess(abs(sensitivity - selectivity), TOLERANCE * 2)
                else:
                    # This is not what should happen but what happens.
                    # If these tests fail we should test if the minority class is
                    # respected and update the tests.
                    self.assertLessEqual(ba, 0.5)
                    self.assertLessEqual(sensitivity, 0.1)

    def test_without_class_weight(self) -> None:
        """Test CalibratedClassifierCV without class_weight on imbalanced data.

        Without class_weighting our implementation should match sklearn's.

        Since we do not account for class imbalance, we expect that sensitivity, which
        measures performance on the minority class, will be lower than selectivity.

        """
        param_grid = ParameterGrid(
            {
                "ensemble": [True, False, "auto"],
                "method": ["isotonic", "sigmoid", "temperature"],
            },
        )
        for params in param_grid:
            with self.subTest(params=params):
                clf = LogisticRegression(random_state=SEED, class_weight=None)
                calibrated = CalibratedClassifierCV(
                    clf,
                    cv=2,
                    class_weight=None,
                    **params,
                )
                sk_calibrated = SKCalibratedClassifierCV(
                    clf,
                    cv=2,
                    **params,
                )
                calibrated.fit(self.x_train, self.y_train)
                sk_calibrated.fit(self.x_train, self.y_train)
                preds = calibrated.predict(self.x_test)
                probas = calibrated.predict_proba(self.x_test)
                sk_probas = sk_calibrated.predict_proba(self.x_test)
                self.assertEqual(preds.shape, (self.x_test.shape[0],))
                self.assertEqual(probas.shape, (self.x_test.shape[0], 2))
                self.assertEqual(sk_probas.shape, (self.x_test.shape[0], 2))
                self.assertTrue(np.allclose(probas, sk_probas))

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
                # Accuracy should be high and balanced accuracy low due to imbalance
                self.assertGreater(ac, self.expected_ba - TOLERANCE)
                self.assertLess(ba, self.expected_ba - TOLERANCE)
                self.assertLess(sensitivity, self.sensitivity - TOLERANCE)
                self.assertGreater(selectivity, self.selectivity - TOLERANCE)
                self.assertGreater(selectivity - sensitivity, TOLERANCE)

    def test_without_class_weight_and_with_sample_weight(self) -> None:
        """Test CalibratedClassifierCV without class_weight but with sample_weight.

        Sample weights are accepted by sklearn's CalibratedClassifierCV, so our
        implementation should match sklearn's behavior.

        Since we provide sample weights, we expect sensitivity and selectivity
        to be reasonably balanced and both above 0.8. Temperature scaling seems to
        neglect the minority class despite sample weights, though. Hence, we exclude it.

        Since the sample_weight is also forwarded to the logistic regression,
        we invert its class weights to evaluate only the effect of sample_weight for
        the calibration itself.

        """
        sample_weight = compute_sample_weight("balanced", self.y_train)
        class_weight = compute_class_weight(
            "balanced",
            y=self.y_train,
            classes=np.array([0, 1]),
        )
        param_grid = ParameterGrid(
            {
                "ensemble": [True, False, "auto"],
                "method": ["isotonic", "sigmoid", "temperature"],
            },
        )
        for params in param_grid:
            with self.subTest(params=params):
                clf = LogisticRegression(
                    # Invert class weights to isolate effect of sample_weight
                    # on calibration: sample_weight * 1/class_weight = uniform weights
                    class_weight={0: 1 / class_weight[0], 1: 1 / class_weight[1]},
                    random_state=SEED,
                )
                calibrated = CalibratedClassifierCV(
                    clf,
                    cv=2,
                    **params,
                )
                sk_calibrated = SKCalibratedClassifierCV(
                    clf,
                    cv=2,
                    **params,
                )
                calibrated.fit(
                    self.x_train,
                    self.y_train,
                    sample_weight=sample_weight,
                )
                sk_calibrated.fit(
                    self.x_train,
                    self.y_train,
                    sample_weight=sample_weight,
                )
                preds = calibrated.predict(self.x_test)
                probas = calibrated.predict_proba(self.x_test)
                sk_probas = sk_calibrated.predict_proba(self.x_test)
                self.assertEqual(preds.shape, (self.x_test.shape[0],))
                self.assertEqual(probas.shape, (self.x_test.shape[0], 2))
                self.assertEqual(sk_probas.shape, (self.x_test.shape[0], 2))
                self.assertTrue(np.allclose(probas, sk_probas))

                sensitivity = recall_score(self.y_test, preds, pos_label=1)
                selectivity = recall_score(self.y_test, preds, pos_label=0)
                ba = balanced_accuracy_score(self.y_test, preds)
                logger.debug(
                    f"With sample_weight - Params: {params}, Balanced Accuracy: "
                    f"{ba:.3f}, Sensitivity: {sensitivity:.3f}, "
                    f"Selectivity: {selectivity:.3f}",
                )
                # Temperature seems to neglect minority class despite class_weight
                if params["method"] != "temperature":
                    self.assertGreater(ba, self.expected_ba - TOLERANCE)
                    self.assertGreater(sensitivity, self.sensitivity - TOLERANCE)
                    self.assertLess(abs(sensitivity - selectivity), TOLERANCE * 2)
                else:
                    # This is not what should happen but what happens.
                    # If these tests fail we should test if the minority class is
                    # respected and update the tests.
                    self.assertLessEqual(ba, 0.5)
                    self.assertLessEqual(sensitivity, 0.1)


if __name__ == "__main__":
    unittest.main()
