"""Comprehensive tests for conformal prediction wrappers and pipeline integration."""
# pylint: disable=too-many-lines

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from molpipeline import ErrorFilter, FilterReinserter, Pipeline, PostPredictionWrapper
from molpipeline.any2mol import SmilesToMol
from molpipeline.experimental.model_selection.splitter import (
    create_continuous_stratified_folds,
)
from molpipeline.experimental.uncertainty import (
    ConformalClassifier,
    ConformalRegressor,
    CrossConformalClassifier,
    CrossConformalRegressor,
    LogNonconformity,
    SVMMarginNonconformity,
    create_nonconformity_function,
)
from molpipeline.mol2any import MolToMorganFP
from tests import TEST_DATA_DIR

FP_RADIUS = 2
FP_SIZE = 1024


class BaseConformalTestData(unittest.TestCase):
    """Base class for conformal prediction tests with unified data setup."""

    x_clf: npt.NDArray[np.int_]
    y_clf: npt.NDArray[np.float64]
    x_reg: npt.NDArray[np.int_]
    y_reg: npt.NDArray[np.float64]
    smiles_clf: list[str]
    smiles_reg: list[str]

    @classmethod
    def setUpClass(cls) -> None:  # pylint: disable=too-many-locals
        """Set up test data once for all tests.

        Provides both fingerprint matrices and SMILES lists for comprehensive testing.
        """
        bbbp_df = pd.read_csv(
            TEST_DATA_DIR / "molecule_net_bbbp.tsv.gz",
            sep="\t",
            compression="gzip",
            nrows=200,
        )
        logd_df = pd.read_csv(
            TEST_DATA_DIR / "molecule_net_logd.tsv.gz",
            sep="\t",
            compression="gzip",
            nrows=200,
        )

        error_filter_clf = ErrorFilter()
        pipeline_clf = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("error_filter", error_filter_clf),
                (
                    "morgan",
                    MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense"),
                ),
                (
                    "filter_reinserter",
                    FilterReinserter.from_error_filter(
                        error_filter_clf,
                        fill_value=np.nan,
                    ),
                ),
            ],
        )

        smiles_list_clf = bbbp_df["smiles"]
        labels_clf = bbbp_df["p_np"].to_numpy()
        fingerprint_matrix_clf = pipeline_clf.fit_transform(smiles_list_clf)
        valid_mask_clf = ~np.isnan(fingerprint_matrix_clf).any(axis=1)
        cls.x_clf = fingerprint_matrix_clf[valid_mask_clf].astype(np.int_)
        cls.y_clf = labels_clf[valid_mask_clf].astype(np.float64)
        cls.smiles_clf = [
            smiles_list_clf.iloc[i]
            for i in range(len(smiles_list_clf))
            if valid_mask_clf[i]
        ]

        error_filter_reg = ErrorFilter()
        pipeline_reg = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("error_filter", error_filter_reg),
                (
                    "morgan",
                    MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense"),
                ),
                (
                    "filter_reinserter",
                    FilterReinserter.from_error_filter(
                        error_filter_reg,
                        fill_value=np.nan,
                    ),
                ),
            ],
        )

        smiles_list_reg = logd_df["smiles"]
        labels_reg = logd_df["exp"].to_numpy()
        fingerprint_matrix_reg = pipeline_reg.fit_transform(smiles_list_reg)
        valid_mask_reg = ~np.isnan(fingerprint_matrix_reg).any(axis=1)
        cls.x_reg = fingerprint_matrix_reg[valid_mask_reg].astype(np.int_)
        cls.y_reg = labels_reg[valid_mask_reg].astype(np.float64)
        cls.smiles_reg = [
            smiles_list_reg.iloc[i]
            for i in range(len(smiles_list_reg))
            if valid_mask_reg[i]
        ]

    def _check_prediction_sets_content(
        self,
        prediction_sets: npt.NDArray[np.int_],
        n_classes: int,
    ) -> None:
        """Check that prediction sets contain valid binary arrays.

        Parameters
        ----------
        prediction_sets : npt.NDArray[np.int_]
            Binary prediction sets array where each row corresponds to a sample
            and each column to a class. Values are 0 or 1.
        n_classes : int
            Number of classes.

        """
        self.assertIsInstance(prediction_sets, np.ndarray)
        self.assertEqual(prediction_sets.shape[1], n_classes)
        self.assertTrue(np.all(np.isin(prediction_sets, [0, 1])))

    def _get_train_calib_test_splits(
        self,
        x_data: npt.NDArray[np.int_],
        y_data: npt.NDArray[np.float64],
    ) -> tuple[
        npt.NDArray[np.int_],
        npt.NDArray[np.int_],
        npt.NDArray[np.int_],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Split data into train, calibration, and test sets.

        Parameters
        ----------
        x_data : npt.NDArray[Any]
            Input features array.
        y_data : npt.NDArray[Any]
            Target values array.

        Returns
        -------
        tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any],
              npt.NDArray[Any], npt.NDArray[Any]]
            Tuple containing training features, calibration features, test features,
            training targets, calibration targets, and test targets.

        """
        x_train_all, x_test, y_train_all, y_test = train_test_split(
            x_data,
            y_data,
            test_size=0.2,
            random_state=42,
        )
        x_train, x_calib, y_train, y_calib = train_test_split(
            x_train_all,
            y_train_all,
            test_size=0.3,
            random_state=42,
        )
        return x_train, x_calib, x_test, y_train, y_calib, y_test


# Classification-specific tests
class TestConformalClassifier(BaseConformalTestData):
    """Core functionality tests for ConformalClassifier."""

    def test_evaluate_methods_classifier(self) -> None:
        """Test evaluate methods for ConformalClassifier with proper metrics."""
        x_train, x_calib, x_test, y_train, y_calib, y_test = (
            self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        )

        clf = RandomForestClassifier(random_state=42, n_estimators=50)
        cp = ConformalClassifier(clf)
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)

        # Test default metrics
        results = cp.evaluate(x_test, y_test)
        expected_metrics = {"error", "avg_c", "one_c", "empty", "ks_test"}
        self.assertTrue(expected_metrics.issubset(set(results.keys())))

        # Test specific metrics
        custom_metrics = ["error", "avg_c"]
        results_custom = cp.evaluate(x_test, y_test, metrics=custom_metrics)
        self.assertEqual(set(results_custom.keys()), set(custom_metrics))

        # Validate metric ranges
        self.assertGreaterEqual(results["error"], 0.0)
        self.assertLessEqual(results["error"], 1.0)
        self.assertGreaterEqual(results["avg_c"], 0.0)
        self.assertGreaterEqual(results["one_c"], 0.0)
        self.assertLessEqual(results["one_c"], 1.0)
        self.assertGreaterEqual(results["empty"], 0.0)
        self.assertLessEqual(results["empty"], 1.0)
        self.assertGreaterEqual(results["ks_test"], 0.0)
        self.assertLessEqual(results["ks_test"], 1.0)

    def test_evaluate_methods_cross_conformal(self) -> None:  # pylint: disable=too-many-locals  # noqa: PLR0914
        """Test evaluate methods for cross-conformal predictors (classification)."""
        # Test CrossConformalClassifier
        x_train_clf, x_test_clf, y_train_clf, y_test_clf = train_test_split(
            self.x_clf,
            self.y_clf,
            test_size=0.2,
            random_state=42,
        )

        clf = RandomForestClassifier(random_state=42, n_estimators=50)
        ccp_clf = CrossConformalClassifier(clf, n_folds=2)
        ccp_clf.fit_and_calibrate(x_train_clf, y_train_clf)

        results_clf = ccp_clf.evaluate(x_test_clf, y_test_clf)
        # Should have mean and std for each metric
        expected_keys = {
            "error_mean",
            "error_std",
            "avg_c_mean",
            "avg_c_std",
            "one_c_mean",
            "one_c_std",
            "empty_mean",
            "empty_std",
            "ks_test_mean",
            "ks_test_std",
        }
        self.assertTrue(expected_keys.issubset(set(results_clf.keys())))

    def test_conformal_prediction_classifier(self) -> None:
        """Test core prediction methods of ConformalClassifier."""
        x_train, x_calib, x_test, y_train, y_calib, y_test = (
            self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        )
        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        cp = ConformalClassifier(clf, mondrian=True)
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)
        preds = cp.predict(x_test)
        probs = cp.predict_proba(x_test)
        sets = cp.predict_set(x_test)
        p_values = cp.predict_p(x_test)
        self.assertEqual(len(preds), len(y_test))
        self.assertEqual(probs.shape[0], len(y_test))
        self.assertEqual(len(sets), len(y_test))
        self.assertEqual(len(p_values), len(y_test))

    def test_confidence_level_effect_classification(self) -> None:  # pylint: disable=too-many-locals  # noqa: PLR0914
        """Test effect of confidence level on prediction sets in ConformalClassifier."""
        x_train, x_calib, x_test, y_train, y_calib, y_test = (
            self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        )
        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        cp = ConformalClassifier(clf)
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)
        preds = cp.predict(x_test)
        probs = cp.predict_proba(x_test)
        sets = cp.predict_set(x_test)
        p_values = cp.predict_p(x_test)
        self.assertEqual(len(preds), len(y_test))
        self.assertEqual(probs.shape[0], len(y_test))
        self.assertEqual(len(sets), len(y_test))
        self.assertEqual(len(p_values), len(y_test))
        sets_90 = cp.predict_set(x_test, confidence=0.90)
        sets_95 = cp.predict_set(x_test, confidence=0.95)
        size_90 = float(np.mean([np.sum(set_row) for set_row in sets_90]))
        size_95 = float(np.mean([np.sum(set_row) for set_row in sets_95]))
        self.assertLessEqual(size_90, size_95)

    def test_class_specific_behavior(self) -> None:
        """Test that ConformalClassifier has classification-specific methods."""
        clf = RandomForestClassifier(random_state=42)
        cp_clf = ConformalClassifier(clf)
        self.assertTrue(hasattr(cp_clf, "predict_set"))
        self.assertTrue(hasattr(cp_clf, "predict_proba"))

    def test_fit_calibrate_state_validation_classifier(self) -> None:
        """Test that methods validate fit/calibrate state in ConformalClassifier."""
        x_train, x_calib, x_test, y_train, y_calib, y_test = (
            self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        )
        clf = RandomForestClassifier(random_state=42, n_estimators=50)
        cp_clf = ConformalClassifier(clf)
        with self.assertRaises(ValueError):
            cp_clf.predict(x_test)
        with self.assertRaises(ValueError):
            cp_clf.predict_proba(x_test)
        with self.assertRaises(ValueError):
            cp_clf.predict_set(x_test)
        with self.assertRaises(ValueError):
            cp_clf.predict_p(x_test)
        cp_clf.fit(x_train, y_train)
        with self.assertRaises(RuntimeError):
            cp_clf.predict_set(x_test)
        with self.assertRaises(RuntimeError):
            cp_clf.predict_p(x_test)
        preds = cp_clf.predict(x_test)
        probs = cp_clf.predict_proba(x_test)
        self.assertEqual(len(preds), len(y_test))
        self.assertEqual(probs.shape[0], len(y_test))
        cp_clf.calibrate(x_calib, y_calib)
        sets = cp_clf.predict_set(x_test)
        p_values = cp_clf.predict_p(x_test)
        self.assertEqual(len(sets), len(y_test))
        self.assertEqual(len(p_values), len(y_test))

    def test_nonconformity_functions(self) -> None:  # pylint: disable=too-many-locals  # noqa: PLR0914
        """Test different nonconformity functions in ConformalClassifier."""
        data_splits = self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        x_train, x_calib, x_test, y_train, y_calib, y_test = data_splits
        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        cp_hinge = ConformalClassifier(clf, nonconformity="hinge")
        cp_hinge.fit(x_train, y_train)
        cp_hinge.calibrate(x_calib, y_calib)
        sets_hinge = cp_hinge.predict_set(x_test)
        p_values_hinge = cp_hinge.predict_p(x_test)
        cp_margin = ConformalClassifier(clf, nonconformity="margin")
        cp_margin.fit(x_train, y_train)
        cp_margin.calibrate(x_calib, y_calib)
        sets_margin = cp_margin.predict_set(x_test)
        p_values_margin = cp_margin.predict_p(x_test)
        self.assertEqual(len(sets_hinge), len(y_test))
        self.assertEqual(len(sets_margin), len(y_test))
        self.assertEqual(len(p_values_hinge), len(y_test))
        self.assertEqual(len(p_values_margin), len(y_test))
        self.assertTrue(np.all((p_values_hinge >= 0) & (p_values_hinge <= 1)))
        self.assertTrue(np.all((p_values_margin >= 0) & (p_values_margin <= 1)))
        n_classes = len(np.unique(y_train))
        self._check_prediction_sets_content(sets_hinge, n_classes)
        self._check_prediction_sets_content(sets_margin, n_classes)

    def test_nonconformity_registry_create(self) -> None:
        """Test nonconformity function registry and creation utility."""
        self.assertIsInstance(create_nonconformity_function("log"), LogNonconformity)
        self.assertIsInstance(
            create_nonconformity_function("svm_margin"), SVMMarginNonconformity
        )

    def test_log_nc_function(self) -> None:
        """Test log_nc nonconformity function directly."""
        x_prob = np.array([[0.9, 0.1], [0.3, 0.7], [0.6, 0.4]])
        classes = np.array([0, 1])
        y = np.array([0, 1, 0])

        # Test calculate_nonconformity returns all class scores
        log_nc_func = LogNonconformity()
        scores_all = log_nc_func.calculate_nonconformity(x_prob)
        expected_all = -np.log(x_prob)
        self.assertTrue(np.allclose(scores_all, expected_all))

        # Extract true class scores using helper method
        scores = log_nc_func.extract_true_class_scores(x_prob, y, classes)
        expected = -np.log([0.9, 0.7, 0.6])
        self.assertTrue(np.allclose(scores, expected))

        y_series = pd.Series(y)
        scores_series = log_nc_func.extract_true_class_scores(
            x_prob, y_series.to_numpy(), classes
        )
        self.assertTrue(np.allclose(scores_series, expected))

        x_prob_small = np.array([[1e-15, 1.0], [1.0, 1e-15]])
        y_small = np.array([0, 1])
        scores_small = log_nc_func.extract_true_class_scores(
            x_prob_small, y_small, classes
        )
        self.assertTrue(np.all(np.isfinite(scores_small)))
        self.assertTrue(np.all(scores_small >= -np.log(1.0)))

    def test_log_nonconformity_class(self) -> None:  # pylint: disable=too-many-locals
        """Test LogNonconformity functor class."""
        x_train, x_calib, x_test, y_train, y_calib, _y_test = (
            self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        )

        classes = np.array(sorted(np.unique(y_train)))
        clf = RandomForestClassifier(random_state=42, n_estimators=10)
        clf.fit(x_train, y_train)

        # Get probability predictions
        probs_calib = clf.predict_proba(x_calib)
        probs_test = clf.predict_proba(x_test)

        log_nc_func = LogNonconformity()

        # Test with true labels (calibration)
        nc_calib = log_nc_func(probs_calib, classes=classes, y_true=y_calib)
        self.assertEqual(nc_calib.shape, (len(y_calib),))
        self.assertTrue(
            np.all(nc_calib >= 0)
        )  # Nonconformity scores should be non-negative

        # Test without true labels (test set - returns matrix)
        nc_test_all = log_nc_func(probs_test)
        self.assertEqual(nc_test_all.shape, (len(probs_test), len(classes)))
        self.assertTrue(np.all(nc_test_all >= 0))

        # Test get_name method
        self.assertEqual(log_nc_func.get_name(), "log")

        # Test that it works with ConformalClassifier
        cp = ConformalClassifier(clf, nonconformity="log")
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)
        sets = cp.predict_set(x_test)
        self.assertEqual(len(sets), len(probs_test))

    def test_svm_margin_binary_classification(self) -> None:  # pylint: disable=too-many-locals
        """Test SVMMarginNonconformity with binary SVM classification."""
        x_train, x_calib, x_test, y_train, y_calib, _y_test = (
            self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        )

        classes = np.array(sorted(np.unique(y_train)))
        self.assertEqual(len(classes), 2)

        svc = SVC(kernel="linear", probability=False, random_state=42)
        svc.fit(x_train, y_train)

        # Decision function values (signed distances to hyperplane)
        y_score_calib = svc.decision_function(x_calib)
        y_score_test = svc.decision_function(x_test)
        self.assertEqual(y_score_calib.ndim, 1)
        self.assertEqual(y_score_test.ndim, 1)

        # Test nonconformity for true labels on calibration set
        nc_calib_true = SVMMarginNonconformity()(
            y_score_calib, classes=classes, y_true=y_calib
        )
        self.assertEqual(nc_calib_true.shape, (len(y_score_calib),))

        y_mapped = np.where(y_calib == classes[1], 1, -1)
        expected_nc = np.where(y_mapped == 1, 1 - y_score_calib, y_score_calib + 1)
        self.assertTrue(np.allclose(nc_calib_true, expected_nc))

        nc_test_all = SVMMarginNonconformity()(y_score_test)
        self.assertEqual(nc_test_all.shape, (len(y_score_test), 2))
        expected_all = np.column_stack((y_score_test + 1, 1 - y_score_test))
        self.assertTrue(np.allclose(nc_test_all, expected_all))

        preds = svc.predict(x_test)
        confident_mask = np.abs(y_score_test) > 0.5

        pos_confident = confident_mask & (preds == classes[1])
        neg_confident = confident_mask & (preds == classes[0])
        self.assertGreater(
            np.sum(pos_confident),
            0,
            f"Expected confident positive predictions (got {np.sum(pos_confident)})",
        )
        self.assertGreater(
            np.sum(neg_confident),
            0,
            f"Expected confident negative predictions (got {np.sum(neg_confident)})",
        )

        self.assertTrue(
            np.all(nc_test_all[pos_confident, 1] < nc_test_all[pos_confident, 0]),
            "Confident positives should have lower NC for their true class",
        )
        self.assertTrue(
            np.all(nc_test_all[neg_confident, 0] < nc_test_all[neg_confident, 1]),
            "Confident negatives should have lower NC for their true class",
        )

    def test_cross_conformal_classifier(self) -> None:
        """Test CrossConformalClassifier."""
        x_train, x_test, y_train, y_test = train_test_split(
            self.x_clf,
            self.y_clf,
            test_size=0.2,
            random_state=42,
        )

        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        ccp = CrossConformalClassifier(clf, n_folds=2)
        ccp.fit_and_calibrate(x_train, y_train)

        preds = ccp.predict(x_test)
        probs = ccp.predict_proba(x_test)
        sets = ccp.predict_set(x_test)
        p_values = ccp.predict_p(x_test)

        self.assertEqual(len(preds), len(y_test))
        self.assertEqual(probs.shape[0], len(y_test))
        self.assertEqual(len(sets), len(y_test))
        self.assertEqual(len(p_values), len(y_test))

    def test_cross_conformal_confidence_effect_classification(self) -> None:
        """Test confidence level effect in cross-conformal classification."""
        x_train, x_test, y_train, _y_test = train_test_split(
            self.x_clf,
            self.y_clf,
            test_size=0.2,
            random_state=42,
        )

        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        ccp = CrossConformalClassifier(clf, n_folds=2)
        ccp.fit_and_calibrate(x_train, y_train)
        sets_90 = ccp.predict_set(x_test, confidence=0.90)
        sets_95 = ccp.predict_set(x_test, confidence=0.95)
        size_90 = float(np.mean([np.sum(set_row) for set_row in sets_90]))
        size_95 = float(np.mean([np.sum(set_row) for set_row in sets_95]))

        self.assertLessEqual(size_90, size_95)

    def test_pipeline_wrapped_by_conformal_classifier(self) -> None:  # pylint: disable=too-many-locals  # noqa: PLR0914
        """Test a MolPipeline wrapped by ConformalClassifier."""
        smi2mol = SmilesToMol()
        mol2morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense")
        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        error_filter = ErrorFilter(filter_everything=True)

        base_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("morgan", mol2morgan),
                ("error_filter", error_filter),
                ("classifier", clf),
                (
                    "error_replacer",
                    PostPredictionWrapper(
                        FilterReinserter.from_error_filter(error_filter, np.nan),
                    ),
                ),
            ],
        )

        conformal_pipeline = ConformalClassifier(base_pipeline)

        train_smiles, temp_smiles, y_train, y_temp = train_test_split(
            self.smiles_clf,
            self.y_clf,
            test_size=0.4,
            random_state=42,
        )
        calib_smiles, test_smiles, y_calib, y_test = train_test_split(
            temp_smiles,
            y_temp,
            test_size=0.5,
            random_state=42,
        )

        conformal_pipeline.fit(train_smiles, y_train)
        conformal_pipeline.calibrate(calib_smiles, y_calib)

        preds = conformal_pipeline.predict(test_smiles)
        sets = conformal_pipeline.predict_set(test_smiles)
        p_values = conformal_pipeline.predict_p(test_smiles)

        self.assertEqual(len(preds), len(y_test))
        self.assertEqual(len(sets), len(y_test))
        self.assertEqual(len(p_values), len(y_test))

        invalid_smiles = ["C1CC", "X", "."]
        test_smiles_with_invalid = test_smiles + invalid_smiles
        predicted_with_invalid = conformal_pipeline.predict(test_smiles_with_invalid)

        self.assertEqual(len(predicted_with_invalid), len(test_smiles_with_invalid))
        n_valid = len(test_smiles)
        n_invalid = len(invalid_smiles)

        valid_predictions = predicted_with_invalid[:n_valid]
        self.assertFalse(np.isnan(valid_predictions).any())

        invalid_predictions = predicted_with_invalid[n_valid:]
        self.assertTrue(np.isnan(invalid_predictions).all())
        self.assertEqual(len(invalid_predictions), n_invalid)


# Regression-specific tests
class TestConformalRegressor(BaseConformalTestData):
    """Core functionality tests for ConformalRegressor."""

    def test_confidence_effect_regression(self) -> None:
        """Test effect of confidence parameter on prediction intervals in ConformalRegressor."""
        x_train, x_calib, x_test, y_train, y_calib, y_test = (
            self._get_train_calib_test_splits(self.x_reg, self.y_reg)
        )
        reg = RandomForestRegressor(random_state=42, n_estimators=5)
        cp = ConformalRegressor(reg)
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)
        intervals_90 = cp.predict_int(x_test, confidence=0.90)
        self.assertEqual(intervals_90.shape[0], len(y_test))
        self.assertEqual(intervals_90.shape[1], 2)
        intervals_95 = cp.predict_int(x_test, confidence=0.95)
        width_90 = float(np.mean(intervals_90[:, 1] - intervals_90[:, 0]))
        width_95 = float(np.mean(intervals_95[:, 1] - intervals_95[:, 0]))
        self.assertLess(width_90, width_95)

    def test_class_specific_behavior(self) -> None:
        """Test that ConformalRegressor has regression-specific methods."""
        reg = RandomForestRegressor(random_state=42)
        cp_reg = ConformalRegressor(reg)
        self.assertTrue(hasattr(cp_reg, "predict_int"))
        self.assertFalse(hasattr(cp_reg, "predict_proba"))

    def test_fit_calibrate_state_validation_regressor(self) -> None:
        """Test that methods validate fit/calibrate state in ConformalRegressor."""
        x_train_reg, x_calib_reg, x_test_reg, y_train_reg, y_calib_reg, y_test_reg = (
            self._get_train_calib_test_splits(self.x_reg, self.y_reg)
        )
        reg = RandomForestRegressor(random_state=42, n_estimators=50)
        cp_reg = ConformalRegressor(reg)
        with self.assertRaises(ValueError):
            cp_reg.predict(x_test_reg)
        with self.assertRaises(ValueError):
            cp_reg.predict_int(x_test_reg)
        cp_reg.fit(x_train_reg, y_train_reg)
        with self.assertRaises(RuntimeError):
            cp_reg.predict_int(x_test_reg)
        preds_reg = cp_reg.predict(x_test_reg)
        self.assertEqual(len(preds_reg), len(y_test_reg))
        cp_reg.calibrate(x_calib_reg, y_calib_reg)
        intervals = cp_reg.predict_int(x_test_reg)
        self.assertEqual(intervals.shape, (len(y_test_reg), 2))

    def test_cross_conformal_regressor(self) -> None:
        """Test CrossConformalRegressor with stratified folds for regression."""
        splits = create_continuous_stratified_folds(
            self.y_reg, n_splits=2, random_state=42
        )
        (train_idx, test_idx) = splits[0]
        x_train, x_test = self.x_reg[train_idx], self.x_reg[test_idx]
        y_train, y_test = self.y_reg[train_idx], self.y_reg[test_idx]
        reg = RandomForestRegressor(random_state=42, n_estimators=5)
        ccp = CrossConformalRegressor(reg, n_folds=2)
        ccp.fit_and_calibrate(x_train, y_train)
        intervals = ccp.predict_int(x_test)
        for model in ccp.models_:
            model_intervals = model.predict_int(x_test)
            self.assertEqual(model_intervals.shape[0], len(y_test))
            self.assertEqual(model_intervals.shape[1], 2)
        self.assertEqual(intervals.shape[0], len(y_test))
        self.assertEqual(intervals.shape[1], 2)

    def test_cross_conformal_confidence_effect_regression(self) -> None:
        """Test confidence level effect in cross-conformal regression with stratified folds."""
        splits = create_continuous_stratified_folds(
            self.y_reg, n_splits=2, random_state=42
        )
        (train_idx, test_idx) = splits[0]
        x_train, x_test = self.x_reg[train_idx], self.x_reg[test_idx]
        y_train, _ = self.y_reg[train_idx], self.y_reg[test_idx]
        reg = RandomForestRegressor(random_state=42, n_estimators=5)
        ccp = CrossConformalRegressor(reg, n_folds=2)
        ccp.fit_and_calibrate(x_train, y_train)
        intervals_90 = ccp.predict_int(x_test, confidence=0.90)
        intervals_95 = ccp.predict_int(x_test, confidence=0.95)
        width_90 = float(np.mean(intervals_90[:, 1] - intervals_90[:, 0]))
        width_95 = float(np.mean(intervals_95[:, 1] - intervals_95[:, 0]))
        self.assertLess(width_90, width_95)

    def test_pipeline_wrapped_by_cross_conformal_regressor(self) -> None:
        """Test a MolPipeline wrapped by CrossConformalRegressor."""
        smi2mol = SmilesToMol()
        mol2morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense")
        reg = RandomForestRegressor(n_estimators=5, random_state=42)
        error_filter = ErrorFilter(filter_everything=True)

        base_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("morgan", mol2morgan),
                ("error_filter", error_filter),
                ("regressor", reg),
                (
                    "error_replacer",
                    PostPredictionWrapper(
                        FilterReinserter.from_error_filter(error_filter, np.nan),
                    ),
                ),
            ],
        )

        conformal_pipeline = CrossConformalRegressor(base_pipeline, n_folds=2)

        train_smiles, test_smiles, y_train, y_test = train_test_split(
            self.smiles_reg,
            self.y_reg,
            test_size=0.4,
            random_state=42,
        )

        conformal_pipeline.fit_and_calibrate(train_smiles, y_train)

        preds = conformal_pipeline.predict(test_smiles)
        intervals = conformal_pipeline.predict_int(test_smiles)

        self.assertEqual(len(preds), len(y_test))
        self.assertEqual(intervals.shape[0], len(y_test))
        self.assertEqual(intervals.shape[1], 2)

    def test_joblib_serialization_cross_conformal(self) -> None:
        """Test joblib serialization of CrossConformalRegressor."""
        x_train, x_test, y_train, _y_test = train_test_split(
            self.x_reg,
            self.y_reg,
            test_size=0.3,
            random_state=42,
        )
        reg = RandomForestRegressor(n_estimators=50, random_state=42)
        ccp = CrossConformalRegressor(reg, n_folds=2, random_state=42)
        ccp.fit_and_calibrate(x_train, y_train)
        original_preds = ccp.predict(x_test)
        original_intervals = ccp.predict_int(x_test)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "cross_conformal_model.joblib"
            joblib.dump(ccp, model_path)
            loaded_ccp = joblib.load(model_path)
            self.assertIsInstance(loaded_ccp, CrossConformalRegressor)
            loaded_preds = loaded_ccp.predict(x_test)
            loaded_intervals = loaded_ccp.predict_int(x_test)
            self.assertTrue(np.array_equal(original_preds, loaded_preds))
            self.assertTrue(np.allclose(original_intervals, loaded_intervals))

    def test_evaluate_methods_regressor(self) -> None:
        """Test evaluate methods for ConformalRegressor with proper metrics."""
        x_train, x_calib, x_test, y_train, y_calib, y_test = (
            self._get_train_calib_test_splits(self.x_reg, self.y_reg)
        )

        reg = RandomForestRegressor(random_state=42, n_estimators=50)
        cp = ConformalRegressor(reg)
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)

        # Test default metrics
        results = cp.evaluate(x_test, y_test)
        expected_metrics = {"error", "eff_mean", "eff_med", "ks_test"}
        self.assertTrue(expected_metrics.issubset(set(results.keys())))

        # Test specific metrics
        custom_metrics = ["error", "eff_mean"]
        results_custom = cp.evaluate(x_test, y_test, metrics=custom_metrics)
        self.assertEqual(set(results_custom.keys()), set(custom_metrics))

        # Validate metric ranges
        self.assertGreaterEqual(results["error"], 0.0)
        self.assertLessEqual(results["error"], 1.0)
        self.assertGreaterEqual(results["eff_mean"], 0.0)
        self.assertGreaterEqual(results["eff_med"], 0.0)
        self.assertGreaterEqual(results["ks_test"], 0.0)
        self.assertLessEqual(results["ks_test"], 1.0)

    def test_evaluate_methods_cross_conformal_regressor(self) -> None:
        """Test evaluate methods for cross-conformal predictors (regression) with stratified folds."""
        splits = create_continuous_stratified_folds(
            self.y_reg, n_splits=2, random_state=42
        )
        (train_idx, test_idx) = splits[0]
        x_train, x_test = self.x_reg[train_idx], self.x_reg[test_idx]
        y_train, y_test = self.y_reg[train_idx], self.y_reg[test_idx]
        reg = RandomForestRegressor(random_state=42, n_estimators=50)
        ccp_reg = CrossConformalRegressor(reg, n_folds=2)
        ccp_reg.fit_and_calibrate(x_train, y_train)
        results_reg = ccp_reg.evaluate(x_test, y_test)
        expected_keys_reg = {
            "error_mean",
            "error_std",
            "eff_mean_mean",
            "eff_mean_std",
            "eff_med_mean",
            "eff_med_std",
            "ks_test_mean",
            "ks_test_std",
        }
        self.assertTrue(expected_keys_reg.issubset(set(results_reg.keys())))


if __name__ == "__main__":
    unittest.main()
