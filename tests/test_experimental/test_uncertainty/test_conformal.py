"""Comprehensive tests for conformal prediction wrappers and pipeline integration."""
# pylint: disable=too-many-lines

import tempfile
import unittest
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from molpipeline import ErrorFilter, FilterReinserter, Pipeline, PostPredictionWrapper
from molpipeline.any2mol import SmilesToMol
from molpipeline.experimental.uncertainty.conformal import (
    ConformalClassifier,
    ConformalRegressor,
    CrossConformalClassifier,
    CrossConformalRegressor,
)
from molpipeline.mol2any import MolToMorganFP
from tests import TEST_DATA_DIR

FP_RADIUS = 2
FP_SIZE = 1024


class BaseConformalTestData(unittest.TestCase):
    """Base class for conformal prediction tests with unified data setup."""

    x_clf: npt.NDArray[Any]
    y_clf: npt.NDArray[Any]
    x_reg: npt.NDArray[Any]
    y_reg: npt.NDArray[Any]
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
        cls.x_clf = fingerprint_matrix_clf[valid_mask_clf]
        cls.y_clf = labels_clf[valid_mask_clf]
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
        cls.x_reg = fingerprint_matrix_reg[valid_mask_reg]
        cls.y_reg = labels_reg[valid_mask_reg]
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
        # Check all values are binary (0 or 1)
        self.assertTrue(np.all(np.isin(prediction_sets, [0, 1])))

    def _get_train_calib_test_splits(  # noqa: PLR6301
        self,
        x_data: npt.NDArray[Any],
        y_data: npt.NDArray[Any],
    ) -> tuple[
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any],
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


class TestConformalCore(BaseConformalTestData):
    """Core functionality tests for ConformalClassifier and ConformalRegressor."""

    def test_conformal_prediction_classifier(self) -> None:
        """Test ConformalClassifier."""
        x_train, x_calib, x_test, y_train, y_calib, y_test = (
            self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        )
        clf = RandomForestClassifier(random_state=42, n_estimators=100)
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

    def test_confidence_level_effect_regression(self) -> None:
        """Test that increasing confidence level increases interval width."""
        x_train, x_calib, x_test, y_train, y_calib, y_test = (
            self._get_train_calib_test_splits(self.x_reg, self.y_reg)
        )
        reg = RandomForestRegressor(random_state=42, n_estimators=100)
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

    def test_confidence_level_effect_classification(self) -> None:  # pylint: disable=too-many-locals  # noqa: PLR0914
        """Test that lower confidence level increases prediction set size."""
        x_train, x_calib, x_test, y_train, y_calib, y_test = (
            self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        )

        clf = RandomForestClassifier(random_state=42, n_estimators=100)
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

    def test_class_specific_behavior(self) -> None:  # pylint: disable=too-many-locals
        """Test that classifier and regressor classes behave appropriately."""
        clf = RandomForestClassifier(random_state=42)
        cp_clf = ConformalClassifier(clf)
        self.assertTrue(hasattr(cp_clf, "predict_set"))
        self.assertTrue(hasattr(cp_clf, "predict_proba"))

        reg = RandomForestRegressor(random_state=42)
        cp_reg = ConformalRegressor(reg)
        self.assertTrue(hasattr(cp_reg, "predict_int"))
        self.assertFalse(hasattr(cp_reg, "predict_proba"))

    def test_fit_calibrate_state_validation(self) -> None:  # pylint: disable=too-many-locals  # noqa: PLR0914
        """Test that fit and calibrate states are properly tracked and enforced."""
        x_train, x_calib, x_test, y_train, y_calib, y_test = (
            self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        )

        # Test classifier state validation
        clf = RandomForestClassifier(random_state=42, n_estimators=50)
        cp_clf = ConformalClassifier(clf)

        # Should raise error when trying to predict without fitting
        with self.assertRaises(ValueError):
            cp_clf.predict(x_test)
        with self.assertRaises(ValueError):
            cp_clf.predict_proba(x_test)
        with self.assertRaises(ValueError):
            cp_clf.predict_set(x_test)
        with self.assertRaises(ValueError):
            cp_clf.predict_p(x_test)

        # Fit but don't calibrate
        cp_clf.fit(x_train, y_train)

        # Should raise error when trying conformal predictions without calibration
        with self.assertRaises(RuntimeError):
            cp_clf.predict_set(x_test)
        with self.assertRaises(RuntimeError):
            cp_clf.predict_p(x_test)

        # Regular predictions should work after fitting
        preds = cp_clf.predict(x_test)
        probs = cp_clf.predict_proba(x_test)
        self.assertEqual(len(preds), len(y_test))
        self.assertEqual(probs.shape[0], len(y_test))

        # After calibration, all methods should work
        cp_clf.calibrate(x_calib, y_calib)
        sets = cp_clf.predict_set(x_test)
        p_values = cp_clf.predict_p(x_test)
        self.assertEqual(len(sets), len(y_test))
        self.assertEqual(len(p_values), len(y_test))

        # Test regressor state validation
        x_train_reg, x_calib_reg, x_test_reg, y_train_reg, y_calib_reg, y_test_reg = (
            self._get_train_calib_test_splits(self.x_reg, self.y_reg)
        )

        reg = RandomForestRegressor(random_state=42, n_estimators=50)
        cp_reg = ConformalRegressor(reg)

        # Should raise error when trying to predict without fitting
        with self.assertRaises(ValueError):
            cp_reg.predict(x_test_reg)
        with self.assertRaises(ValueError):
            cp_reg.predict_int(x_test_reg)

        # Fit but don't calibrate
        cp_reg.fit(x_train_reg, y_train_reg)

        # Should raise error when trying conformal predictions without calibration
        with self.assertRaises(RuntimeError):
            cp_reg.predict_int(x_test_reg)

        # Regular predictions should work after fitting
        preds_reg = cp_reg.predict(x_test_reg)
        self.assertEqual(len(preds_reg), len(y_test_reg))

        # After calibration, interval predictions should work
        cp_reg.calibrate(x_calib_reg, y_calib_reg)
        intervals = cp_reg.predict_int(x_test_reg)
        self.assertEqual(intervals.shape, (len(y_test_reg), 2))

    def test_nonconformity_functions(self) -> None:  # pylint: disable=too-many-locals
        """Test nonconformity functions for classification."""
        data_splits = self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        x_train, x_calib, x_test, y_train, y_calib, y_test = data_splits

        clf = RandomForestClassifier(random_state=42, n_estimators=100)

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

    def test_cross_conformal_classifier(self) -> None:
        """Test CrossConformalClassifier."""
        x_train, _, x_test, y_train, _, y_test = self._get_train_calib_test_splits(
            self.x_clf,
            self.y_clf,
        )

        clf = RandomForestClassifier(random_state=42, n_estimators=100)
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

    def test_cross_conformal_regressor(self) -> None:
        """Test CrossConformalRegressor."""
        x_train, _, x_test, y_train, _, y_test = self._get_train_calib_test_splits(
            self.x_reg,
            self.y_reg,
        )

        reg = RandomForestRegressor(random_state=42, n_estimators=100)
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
        """Test confidence level effect in cross-conformal regression."""
        x_train, _, x_test, y_train, _, _y_test = self._get_train_calib_test_splits(
            self.x_reg,
            self.y_reg,
        )

        reg = RandomForestRegressor(random_state=42, n_estimators=100)
        ccp = CrossConformalRegressor(reg, n_folds=2)
        ccp.fit_and_calibrate(x_train, y_train)
        intervals_90 = ccp.predict_int(x_test, confidence=0.90)
        intervals_95 = ccp.predict_int(x_test, confidence=0.95)
        width_90 = float(np.mean(intervals_90[:, 1] - intervals_90[:, 0]))
        width_95 = float(np.mean(intervals_95[:, 1] - intervals_95[:, 0]))
        self.assertLess(width_90, width_95)

    def test_cross_conformal_confidence_effect_classification(self) -> None:
        """Test confidence level effect in cross-conformal classification."""
        x_train, _, x_test, y_train, _, _y_test = self._get_train_calib_test_splits(
            self.x_clf,
            self.y_clf,
        )

        clf = RandomForestClassifier(random_state=42, n_estimators=100)
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
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
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
        # Check that invalid SMILES produce nan values at correct positions
        n_valid = len(test_smiles)
        n_invalid = len(invalid_smiles)

        # Valid predictions should not be nan
        valid_predictions = predicted_with_invalid[:n_valid]
        self.assertFalse(np.isnan(valid_predictions).any())

        # Invalid predictions should be nan
        invalid_predictions = predicted_with_invalid[n_valid:]
        self.assertTrue(np.isnan(invalid_predictions).all())
        self.assertEqual(len(invalid_predictions), n_invalid)

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

    def test_evaluate_methods_cross_conformal(self) -> None:  # pylint: disable=too-many-locals  # noqa: PLR0914
        """Test evaluate methods for cross-conformal predictors."""
        # Test CrossConformalClassifier
        x_train_clf, _, x_test_clf, y_train_clf, _, y_test_clf = (
            self._get_train_calib_test_splits(self.x_clf, self.y_clf)
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

        # Test CrossConformalRegressor
        x_train_reg, _, x_test_reg, y_train_reg, _, y_test_reg = (
            self._get_train_calib_test_splits(self.x_reg, self.y_reg)
        )

        reg = RandomForestRegressor(random_state=42, n_estimators=50)
        ccp_reg = CrossConformalRegressor(reg, n_folds=2)
        ccp_reg.fit_and_calibrate(x_train_reg, y_train_reg)

        results_reg = ccp_reg.evaluate(x_test_reg, y_test_reg)
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
