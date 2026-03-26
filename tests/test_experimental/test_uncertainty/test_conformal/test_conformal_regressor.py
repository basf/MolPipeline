"""Regression-focused tests for conformal prediction wrappers and integration."""
# pylint: disable=duplicate-code

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from molpipeline import ErrorFilter, FilterReinserter, Pipeline, PostPredictionWrapper
from molpipeline.any2mol import SmilesToMol
from molpipeline.experimental.model_selection.splitter import PercentileStratifiedKFold
from molpipeline.experimental.uncertainty import (
    ConformalRegressor,
    CrossConformalRegressor,
)
from molpipeline.mol2any import MolToMorganFP
from tests.test_experimental.test_uncertainty.test_conformal import (
    FP_RADIUS,
    FP_SIZE,
    RANDOM_SEED,
    BaseConformalTestData,
)


class TestConformalRegressor(BaseConformalTestData):
    """Core functionality tests for ConformalRegressor."""

    def test_confidence_effect_regression(self) -> None:
        """Test effect of confidence parameter on prediction intervals."""
        x_train, x_calib, x_test, y_train, y_calib, y_test = (
            self._get_train_calib_test_splits(self.x_reg, self.y_reg)
        )
        reg = RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=5)
        cp = ConformalRegressor(reg)
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)
        intervals_80 = cp.predict_int(x_test, confidence=0.80)
        self.assertEqual(intervals_80.shape[0], len(y_test))
        self.assertEqual(intervals_80.shape[1], 2)
        intervals_95 = cp.predict_int(x_test, confidence=0.95)
        width_80 = float(np.mean(intervals_80[:, 1] - intervals_80[:, 0]))
        width_95 = float(np.mean(intervals_95[:, 1] - intervals_95[:, 0]))
        self.assertLess(width_80, width_95)

    def test_class_specific_behavior(self) -> None:
        """Test that ConformalRegressor has regression-specific methods."""
        reg = RandomForestRegressor(random_state=RANDOM_SEED)
        cp_reg = ConformalRegressor(reg)
        self.assertTrue(hasattr(cp_reg, "predict_int"))
        self.assertFalse(hasattr(cp_reg, "predict_proba"))

    def test_fit_calibrate_state_validation_regressor(self) -> None:
        """Test that methods validate fit/calibrate state in ConformalRegressor."""
        x_train_reg, x_calib_reg, x_test_reg, y_train_reg, y_calib_reg, y_test_reg = (
            self._get_train_calib_test_splits(self.x_reg, self.y_reg)
        )
        reg = RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=5)
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

    def test_cross_conformal_regressor(self) -> None:  # noqa: PLR0914  # pylint: disable=too-many-locals
        """Test CrossConformalRegressor with stratified folds for regression."""
        splitter = PercentileStratifiedKFold(
            n_splits=2,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        splits = list(splitter.split(X=np.zeros(len(self.y_reg)), y=self.y_reg))
        (train_idx, test_idx) = splits[0]
        x_train, x_test = self.x_reg[train_idx], self.x_reg[test_idx]
        y_train, y_test = self.y_reg[train_idx], self.y_reg[test_idx]
        x_train_fit, x_calib, y_train_fit, y_calib = train_test_split(
            x_train,
            y_train,
            test_size=0.25,
            random_state=RANDOM_SEED,
        )
        reg = RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=5)
        ccp = CrossConformalRegressor(reg, n_folds=2)
        ccp.fit(x_train_fit, y_train_fit)
        ccp.cv_splits_ = [
            (np.array([], dtype=int), np.arange(len(x_calib), dtype=int))
            for _ in range(len(ccp.models_))
        ]
        ccp.calibrate(x_calib, y_calib)
        intervals = ccp.predict_int(x_test)
        for model in ccp.models_:
            model_intervals = model.predict_int(x_test)
            self.assertEqual(model_intervals.shape[0], len(y_test))
            self.assertEqual(model_intervals.shape[1], 2)
        self.assertEqual(intervals.shape[0], len(y_test))
        self.assertEqual(intervals.shape[1], 2)

    def test_cross_conformal_confidence_effect_regression(self) -> None:  # noqa: PLR0914  # pylint: disable=too-many-locals
        """Test confidence level effect in cross-conformal regression."""
        splitter = PercentileStratifiedKFold(
            n_splits=2,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        splits = list(splitter.split(X=np.zeros(len(self.y_reg)), y=self.y_reg))
        (train_idx, test_idx) = splits[0]
        x_train, x_test = self.x_reg[train_idx], self.x_reg[test_idx]
        y_train, _ = self.y_reg[train_idx], self.y_reg[test_idx]
        x_train_fit, x_calib, y_train_fit, y_calib = train_test_split(
            x_train,
            y_train,
            test_size=0.25,
            random_state=RANDOM_SEED,
        )
        reg = RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=5)
        ccp = CrossConformalRegressor(reg, n_folds=2)
        ccp.fit(x_train_fit, y_train_fit)
        ccp.cv_splits_ = [
            (np.array([], dtype=int), np.arange(len(x_calib), dtype=int))
            for _ in range(len(ccp.models_))
        ]
        ccp.calibrate(x_calib, y_calib)
        intervals_80 = ccp.predict_int(x_test, confidence=0.80)
        intervals_95 = ccp.predict_int(x_test, confidence=0.95)
        width_80 = float(np.mean(intervals_80[:, 1] - intervals_80[:, 0]))
        width_95 = float(np.mean(intervals_95[:, 1] - intervals_95[:, 0]))
        self.assertLess(width_80, width_95)

    def test_pipeline_wrapped_by_cross_conformal_regressor(self) -> None:  # noqa: PLR0914  # pylint: disable=too-many-locals
        """Test a MolPipeline wrapped by CrossConformalRegressor."""
        smi2mol = SmilesToMol()
        mol2morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense")
        reg = RandomForestRegressor(n_estimators=5, random_state=RANDOM_SEED)
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
            random_state=RANDOM_SEED,
        )
        train_smiles_fit, calib_smiles, y_train_fit, y_calib = train_test_split(
            train_smiles,
            y_train,
            test_size=0.25,
            random_state=RANDOM_SEED,
        )

        conformal_pipeline.fit(train_smiles_fit, y_train_fit)
        conformal_pipeline.cv_splits_ = [
            (np.array([], dtype=int), np.arange(len(calib_smiles), dtype=int))
            for _ in range(len(conformal_pipeline.models_))
        ]
        conformal_pipeline.calibrate(calib_smiles, y_calib)

        preds = conformal_pipeline.predict(test_smiles)
        intervals = conformal_pipeline.predict_int(test_smiles)

        self.assertEqual(len(preds), len(y_test))
        self.assertEqual(intervals.shape[0], len(y_test))
        self.assertEqual(intervals.shape[1], 2)

    def test_joblib_serialization_cross_conformal(self) -> None:  # noqa: PLR0914  # pylint: disable=too-many-locals
        """Test joblib serialization of CrossConformalRegressor."""
        x_train, x_test, y_train, _y_test = train_test_split(
            self.x_reg,
            self.y_reg,
            test_size=0.3,
            random_state=RANDOM_SEED,
        )
        reg = RandomForestRegressor(n_estimators=5, random_state=RANDOM_SEED)
        ccp = CrossConformalRegressor(reg, n_folds=2, random_state=RANDOM_SEED)
        x_train_fit, x_calib, y_train_fit, y_calib = train_test_split(
            x_train,
            y_train,
            test_size=0.25,
            random_state=RANDOM_SEED,
        )
        ccp.fit(x_train_fit, y_train_fit)
        ccp.cv_splits_ = [
            (np.array([], dtype=int), np.arange(len(x_calib), dtype=int))
            for _ in range(len(ccp.models_))
        ]
        ccp.calibrate(x_calib, y_calib)
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

        reg = RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=5)
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

    def test_evaluate_methods_cross_conformal_regressor(self) -> None:  # noqa: PLR0914  # pylint: disable=too-many-locals
        """Test evaluate methods for cross-conformal predictors (regression)."""
        splitter = PercentileStratifiedKFold(
            n_splits=2,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        splits = list(splitter.split(X=np.zeros(len(self.y_reg)), y=self.y_reg))
        (train_idx, test_idx) = splits[0]
        x_train, x_test = self.x_reg[train_idx], self.x_reg[test_idx]
        y_train, y_test = self.y_reg[train_idx], self.y_reg[test_idx]
        x_train_fit, x_calib, y_train_fit, y_calib = train_test_split(
            x_train,
            y_train,
            test_size=0.25,
            random_state=RANDOM_SEED,
        )
        reg = RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=5)
        ccp_reg = CrossConformalRegressor(reg, n_folds=2)
        ccp_reg.fit(x_train_fit, y_train_fit)
        ccp_reg.cv_splits_ = [
            (np.array([], dtype=int), np.arange(len(x_calib), dtype=int))
            for _ in range(len(ccp_reg.models_))
        ]
        ccp_reg.calibrate(x_calib, y_calib)
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
