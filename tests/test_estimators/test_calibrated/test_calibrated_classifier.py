"""Unit tests for CalibratedClassifierCV with emphasis on class_weight.

Tests use a small imbalanced dataset.
"""

import unittest

import numpy as np
import numpy.typing as npt
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV as SKCalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils import compute_sample_weight
from sklearn.utils.class_weight import compute_class_weight

from molpipeline.estimators.calibration.calibrated_classifier import (
    CalibratedClassifierCV,
)
from tests.utils.logging import capture_logs

# Parameters for the tests
SEED = 42
TOLERANCE = 0.1


def make_specific_classification(  # pylint: disable=too-many-locals
    n_positive: int,
    n_negative: int,
    selectivity: float,
    sensitivity: float,
    random_state: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Create a classification dataset with specific class distribution and performance.

    Parameters
    ----------
    n_positive : int
        Number of positive samples.
    n_negative : int
        Number of negative samples.
    selectivity : float
        Desired selectivity (true negative rate).
    sensitivity : float
        Desired sensitivity (true positive rate).
    random_state : int | None, optional
        Random state for reproducibility, by default None.

    Returns
    -------
    npt.NDArray[np.float64]
        Features of the dataset.
    npt.NDArray[np.int64]
        Labels of the dataset.

    Raises
    ------
    AssertionError
        If the generated class distribution does not match expected.
    AssertionError
        If the final class distribution does not match requested.

    """
    # To reach desired selectivity/sensitivity, we need to adjust the number of
    # samples before flipping labels
    # We sample from class 0: True negatives + False positives
    # We sample from class 1: True positives + False negatives

    true_pos = int(n_positive * sensitivity)
    false_pos = int(n_positive - true_pos)
    true_neg = int(n_negative * selectivity)
    false_neg = int(n_negative - true_neg)

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
        random_state=random_state,
    )

    if not (np.sum(y == 0) == n_c0_sample and np.sum(y == 1) == n_c1_sample):
        raise AssertionError(
            "Generated class distribution does not match expected.",
        )
    rng = np.random.default_rng(random_state)
    flip_c0_indices = rng.choice(np.where(y == 0)[0], size=false_pos, replace=False)
    flip_c1_indices = rng.choice(np.where(y == 1)[0], size=false_neg, replace=False)
    y[flip_c0_indices] = 1
    y[flip_c1_indices] = 0
    if not (np.sum(y == 0) == n_negative and np.sum(y == 1) == n_positive):
        raise AssertionError(
            "Final class distribution does not match requested distribution.",
        )
    return x, y


class TestCalibratedClassifierCV(unittest.TestCase):  # pylint: disable=too-many-instance-attributes
    """Unit tests for CalibratedClassifierCV with emphasis on class_weight.

    Tests use a small imbalanced dataset.
    """

    def setUp(self) -> None:
        """Set up any necessary components before each test."""
        n_c0 = 900  # Class 0 is majority class
        n_c1 = 100  # Class 1 is minority class

        self.selectivity = 0.8
        self.sensitivity = 0.8
        self.expected_ba = (self.selectivity + self.sensitivity) / 2

        x, y = make_specific_classification(
            n_positive=n_c1,
            n_negative=n_c0,
            selectivity=self.selectivity,
            sensitivity=self.sensitivity,
            random_state=SEED,
        )

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x,
            y,
            test_size=0.3,
            random_state=SEED,
            stratify=y,
        )

        classes = np.array([0, 1])
        class_weight_arr = compute_class_weight(
            "balanced",
            y=self.y_train,
            classes=classes,
        )
        class_weight_dict = dict(zip(classes, class_weight_arr, strict=True))

        self.cal_param_dict = {
            "ensemble": [True, False, "auto"],
            "method": ["isotonic", "sigmoid", "temperature"],
            "estimator": [LogisticRegression(random_state=SEED, class_weight=None)],
            "class_weight": ["balanced", class_weight_dict],
        }

    def evaluate_predictions(
        self,
        y_true: npt.NDArray[np.int64],
        pred: npt.NDArray[np.int64],
        score: npt.NDArray[np.float64] | None = None,
        balanced_model: bool = True,
        log_info: str = "",
    ) -> None:
        """Evaluate predictions and check the metrics.

        Parameters
        ----------
        y_true : npt.NDArray[np.int64]
            True labels.
        pred : npt.NDArray[np.int64]
            Predicted labels.
        score : npt.NDArray[np.float64] | None, optional
            Score/probabilities for positive class, by default None.
        balanced_model : bool, default=True
            Whether the model is unbalanced.
        log_info : str, optional
            Additional info to log, by default "".

        """
        sensitivity = recall_score(y_true, pred, pos_label=1)
        selectivity = recall_score(y_true, pred, pos_label=0)
        if score is not None:
            roc_auc = f"ROC AUC: {roc_auc_score(y_true, score):.3f} "
        else:
            roc_auc = ""
        ba = balanced_accuracy_score(y_true, pred)
        logger.debug(
            f"{log_info}\n"
            f"Balanced Accuracy: {ba:.3f}, {roc_auc}"
            f"Sensitivity: {sensitivity:.3f}, Selectivity: {selectivity:.3f}",
        )

        self.assertGreater(selectivity, self.selectivity - TOLERANCE)
        if balanced_model:
            self.assertGreater(ba, self.expected_ba - TOLERANCE)
            self.assertGreater(sensitivity, self.sensitivity - TOLERANCE)
            self.assertLess(abs(sensitivity - selectivity), TOLERANCE * 2)
        else:
            self.assertLess(ba, self.expected_ba - TOLERANCE)
            self.assertLess(sensitivity, self.sensitivity - TOLERANCE)
            self.assertGreater(abs(sensitivity - selectivity), TOLERANCE * 2)

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
        for params in ParameterGrid(self.cal_param_dict):
            with self.subTest(params=params):
                calibrated = CalibratedClassifierCV(cv=2, **params)
                calibrated.fit(self.x_train, self.y_train)
                preds = calibrated.predict(self.x_test)
                probas = calibrated.predict_proba(self.x_test)
                self.assertEqual(preds.shape, (self.x_test.shape[0],))
                self.assertEqual(probas.shape, (self.x_test.shape[0], 2))
                self.evaluate_predictions(
                    self.y_test,
                    preds,
                    score=probas[:, 1],
                    balanced_model=params["method"] != "temperature",
                    log_info=params,
                )

    def test_without_class_weight(self) -> None:
        """Test CalibratedClassifierCV without class_weight on imbalanced data.

        Without class_weighting our implementation should match sklearn's.

        Since we do not account for class imbalance, we expect that sensitivity, which
        measures performance on the minority class, will be lower than selectivity.

        """
        param_grid = dict(self.cal_param_dict)
        param_grid.pop("class_weight")
        for params in ParameterGrid(param_grid):
            with self.subTest(params=params):
                calibrated = CalibratedClassifierCV(cv=2, class_weight=None, **params)
                sk_calibrated = SKCalibratedClassifierCV(cv=2, **params)
                calibrated.fit(self.x_train, self.y_train)
                sk_calibrated.fit(self.x_train, self.y_train)
                preds = calibrated.predict(self.x_test)
                probas = calibrated.predict_proba(self.x_test)
                sk_probas = sk_calibrated.predict_proba(self.x_test)
                self.assertEqual(preds.shape, (self.x_test.shape[0],))
                self.assertEqual(probas.shape, (self.x_test.shape[0], 2))
                self.assertEqual(sk_probas.shape, (self.x_test.shape[0], 2))
                self.assertTrue(np.allclose(probas, sk_probas))
                self.evaluate_predictions(
                    self.y_test,
                    preds,
                    probas[:, 1],
                    balanced_model=False,
                    log_info=params,
                )

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
        param_dict = dict(self.cal_param_dict)
        param_dict.pop("class_weight")
        for params in ParameterGrid(param_dict):
            params["estimator"].set_params(
                class_weight={
                    0: 1 / class_weight[0],
                    1: 1 / class_weight[1],
                },
            )
            with self.subTest(params=params):
                calibrated = CalibratedClassifierCV(cv=2, **params, class_weight=None)
                sk_calibrated = SKCalibratedClassifierCV(cv=2, **params)
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
                self.evaluate_predictions(
                    self.y_test,
                    preds,
                    probas[:, 1],
                    balanced_model=params["method"] != "temperature",
                    log_info=params,
                )

    def test_balanced_model_does_not_regress(self) -> None:
        """Test that a balanced model does not regress when calibrated."""
        for params in ParameterGrid(self.cal_param_dict):
            with self.subTest(params=params):
                # Uncalibrated balanced model
                base_model = params["estimator"]
                base_model.set_params(class_weight="balanced")
                base_model.fit(self.x_train, self.y_train)
                base_preds = base_model.predict(self.x_test)
                base_probas = base_model.predict_proba(self.x_test)

                # Evaluate uncalibrated model. This is just a sanity check.
                self.evaluate_predictions(
                    self.y_test,
                    base_preds,
                    score=base_probas[:, 1],
                    balanced_model=True,
                    log_info=f"Uncalibrated {params}",
                )

                # Evaluate calibrated balanced model
                calibrated = CalibratedClassifierCV(cv=2, **params)
                calibrated.fit(self.x_train, self.y_train)
                preds = calibrated.predict(self.x_test)
                probas = calibrated.predict_proba(self.x_test)

                self.evaluate_predictions(
                    self.y_test,
                    preds,
                    score=probas[:, 1],
                    balanced_model=True,
                    log_info=f"Calibrated {params}",
                )

                # Test that sklearn's CalibratedClassifierCV regresses
                # Sklearn's implementation does not account for class_weight internally.
                # Hence, the calibrated model is unbalanced.
                sk_params = dict(params)
                sk_params.pop("class_weight")
                sk_calibrated = SKCalibratedClassifierCV(cv=2, **sk_params)
                sk_calibrated.fit(self.x_train, self.y_train)
                sk_preds = sk_calibrated.predict(self.x_test)
                sk_probas = sk_calibrated.predict_proba(self.x_test)

                # Temperature scaling with sklearn seems to have only litte effect
                # on the calibration. So it is still a balanced model.
                self.evaluate_predictions(
                    self.y_test,
                    sk_preds,
                    score=sk_probas[:, 1],
                    balanced_model=params["method"] == "temperature",
                    log_info=f"Sklearn Calibrated {params}",
                )


if __name__ == "__main__":
    unittest.main()
