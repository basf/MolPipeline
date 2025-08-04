"""Comprehensive tests for conformal prediction wrappers and pipeline integration."""

import tempfile
import unittest
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from crepes.extras import MondrianCategorizer, hinge, margin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from molpipeline import ErrorFilter, FilterReinserter, Pipeline, PostPredictionWrapper
from molpipeline.any2mol import SmilesToMol
from molpipeline.experimental.uncertainty.conformal import (
    ConformalPredictor,
    CrossConformalPredictor,
)
from molpipeline.mol2any import MolToMorganFP
from molpipeline.utils.json_operations import recursive_from_json, recursive_to_json

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "test_data"
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
            nrows=100,
        )
        logd_df = pd.read_csv(
            TEST_DATA_DIR / "molecule_net_logd.tsv.gz",
            sep="\t",
            compression="gzip",
            nrows=100,
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
                        error_filter_clf, fill_value=np.nan
                    ),
                ),
            ]
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
                        error_filter_reg, fill_value=np.nan
                    ),
                ),
            ]
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
        self, prediction_sets: list[npt.NDArray[np.int_]], n_classes: int
    ) -> None:
        """Check that prediction sets contain valid class indices.

        Parameters
        ----------
        prediction_sets : list[npt.NDArray[np.int_]]
            List of prediction sets.
        n_classes : int
            Number of classes.
        """
        for pred_set in prediction_sets:
            self.assertIsInstance(pred_set, np.ndarray)
            for class_idx in pred_set:
                self.assertIsInstance(class_idx, np.integer)
                self.assertGreaterEqual(class_idx, 0)
                self.assertLess(class_idx, n_classes)

    def _get_train_calib_test_splits(
        self, x_data: npt.NDArray[Any], y_data: npt.NDArray[Any]
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
        tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]
            Tuple containing training features, calibration features, test features,
            training targets, calibration targets, and test targets.
        """
        x_train_all, x_test, y_train_all, y_test = train_test_split(
            x_data, y_data, test_size=0.2, random_state=42
        )
        x_train, x_calib, y_train, y_calib = train_test_split(
            x_train_all, y_train_all, test_size=0.3, random_state=42
        )
        return x_train, x_calib, x_test, y_train, y_calib, y_test


class TestConformalPredictorCore(BaseConformalTestData):
    """Core functionality tests for ConformalPredictor (unit tests)."""

    def test_conformal_prediction_classifier(self) -> None:
        """Test ConformalPredictor with a classifier."""
        x_train, x_calib, x_test, y_train, y_calib, y_test = (
            self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        )
        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        cp = ConformalPredictor(clf, estimator_type="classifier")
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)
        preds = cp.predict(x_test)
        probs = cp.predict_proba(x_test)
        sets = cp.predict_conformal_set(x_test)
        p_values = cp.predict_p(x_test)
        self.assertEqual(len(preds), len(y_test))
        self.assertEqual(probs.shape[0], len(y_test))
        self.assertEqual(len(sets), len(y_test))
        self.assertEqual(len(p_values), len(y_test))

    def test_conformal_prediction_regressor(self) -> None:
        """Test ConformalPredictor with a regressor."""
        x_train, x_calib, x_test, y_train, y_calib, y_test = (
            self._get_train_calib_test_splits(self.x_reg, self.y_reg)
        )

        reg = RandomForestRegressor(random_state=42, n_estimators=5)
        cp = ConformalPredictor(reg, estimator_type="regressor")
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)
        intervals = cp.predict_int(x_test)
        self.assertEqual(intervals.shape[0], len(y_test))
        self.assertEqual(intervals.shape[1], 2)

    def test_confidence_level_effect_regression(self) -> None:
        """Test that increasing confidence level increases interval width."""
        x_train, x_calib, x_test, y_train, y_calib, _y_test = (
            self._get_train_calib_test_splits(self.x_reg, self.y_reg)
        )
        reg = RandomForestRegressor(random_state=42, n_estimators=5)
        cp = ConformalPredictor(reg, estimator_type="regressor")
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)
        intervals_90 = cp.predict_int(x_test, confidence=0.90)
        intervals_95 = cp.predict_int(x_test, confidence=0.95)
        width_90 = float(np.mean(intervals_90[:, 1] - intervals_90[:, 0]))
        width_95 = float(np.mean(intervals_95[:, 1] - intervals_95[:, 0]))
        self.assertLess(width_90, width_95)

    def test_confidence_level_effect_classification(self) -> None:
        """Test that lower confidence level increases prediction set size."""
        x_train, x_calib, x_test, y_train, y_calib, _y_test = (
            self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        )

        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        cp = ConformalPredictor(clf, estimator_type="classifier")
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)
        sets_90 = cp.predict_conformal_set(x_test, confidence=0.90)
        sets_95 = cp.predict_conformal_set(x_test, confidence=0.95)
        size_90 = float(np.mean([len(s) for s in sets_90]))
        size_95 = float(np.mean([len(s) for s in sets_95]))
        self.assertLessEqual(size_90, size_95)

    def test_auto_detection(self) -> None:  # pylint: disable=too-many-locals
        """Test automatic estimator type detection."""
        clf = RandomForestClassifier(random_state=42)
        cp_clf = ConformalPredictor(clf, estimator_type="auto")
        self.assertEqual(cp_clf.estimator_type, "classifier")

        reg = RandomForestRegressor(random_state=42)
        cp_reg = ConformalPredictor(reg, estimator_type="auto")
        self.assertEqual(cp_reg.estimator_type, "regressor")

    def test_nonconformity_functions(self) -> None:  # pylint: disable=too-many-locals
        """Test nonconformity functions for classification."""
        data_splits = self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        x_train, x_calib, x_test, y_train, y_calib, y_test = data_splits

        clf = RandomForestClassifier(random_state=42, n_estimators=5)

        cp_hinge = ConformalPredictor(
            clf, estimator_type="classifier", nonconformity=hinge
        )
        cp_hinge.fit(x_train, y_train)
        cp_hinge.calibrate(x_calib, y_calib)
        sets_hinge = cp_hinge.predict_conformal_set(x_test)
        p_values_hinge = cp_hinge.predict_p(x_test)
        cp_margin = ConformalPredictor(
            clf, estimator_type="classifier", nonconformity=margin
        )
        cp_margin.fit(x_train, y_train)
        cp_margin.calibrate(x_calib, y_calib)
        sets_margin = cp_margin.predict_conformal_set(x_test)
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

    def test_mondrian_conformal_classification(self) -> None:  # pylint: disable=too-many-locals
        """Test Mondrian conformal prediction for classification."""
        data_splits = self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        x_train, x_calib, x_test, y_train, y_calib, y_test = data_splits

        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        n_classes = len(np.unique(y_train))

        mc = MondrianCategorizer()
        mc.fit(
            x_calib, f=lambda x: (x[:, 0] > np.median(x[:, 0])).astype(int), no_bins=2
        )

        cp_mondrian = ConformalPredictor(clf, estimator_type="classifier", mondrian=mc)
        cp_mondrian.fit(x_train, y_train)
        cp_mondrian.calibrate(x_calib, y_calib)
        sets_custom = cp_mondrian.predict_conformal_set(x_test)
        p_values_custom = cp_mondrian.predict_p(x_test)

        cp_baseline = ConformalPredictor(
            clf, estimator_type="classifier", mondrian=False
        )
        cp_baseline.fit(x_train, y_train)
        cp_baseline.calibrate(x_calib, y_calib)
        sets_baseline = cp_baseline.predict_conformal_set(x_test)
        p_values_baseline = cp_baseline.predict_p(x_test)

        self.assertEqual(len(sets_custom), len(sets_baseline))
        self.assertEqual(len(p_values_custom), len(y_test))
        self.assertEqual(len(p_values_baseline), len(y_test))

        self._check_prediction_sets_content(sets_custom, n_classes)
        self._check_prediction_sets_content(sets_baseline, n_classes)

        self.assertTrue(np.all((p_values_custom >= 0) & (p_values_custom <= 1)))
        self.assertTrue(np.all((p_values_baseline >= 0) & (p_values_baseline <= 1)))

    def test_mondrian_conformal_regression(self) -> None:
        """Test Mondrian conformal prediction for regression."""
        data_splits = self._get_train_calib_test_splits(self.x_reg, self.y_reg)
        x_train, x_calib, x_test, y_train, y_calib, y_test = data_splits

        reg = RandomForestRegressor(random_state=42, n_estimators=5)

        mc = MondrianCategorizer()
        mc.fit(
            x_calib, f=lambda x: (x[:, 0] > np.median(x[:, 0])).astype(int), no_bins=2
        )

        cp_mondrian = ConformalPredictor(reg, estimator_type="regressor", mondrian=mc)
        cp_mondrian.fit(x_train, y_train)
        cp_mondrian.calibrate(x_calib, y_calib)
        intervals_mondrian = cp_mondrian.predict_int(x_test)

        cp_baseline = ConformalPredictor(
            reg, estimator_type="regressor", mondrian=False
        )
        cp_baseline.fit(x_train, y_train)
        cp_baseline.calibrate(x_calib, y_calib)
        intervals_baseline = cp_baseline.predict_int(x_test)

        self.assertEqual(intervals_mondrian.shape, (len(y_test), 2))
        self.assertEqual(intervals_baseline.shape, (len(y_test), 2))

        self.assertFalse(np.array_equal(intervals_mondrian, intervals_baseline))

    def test_error_handling(self) -> None:
        """Test error handling for various invalid operations."""
        x_train, x_test, y_train, y_test = train_test_split(
            self.x_clf, self.y_clf, test_size=0.3, random_state=42
        )

        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        cp = ConformalPredictor(clf, estimator_type="classifier")

        x_test_subset, _, _, _ = train_test_split(
            x_test, y_test, test_size=0.5, random_state=42
        )
        with self.assertRaises(ValueError):
            cp.predict(x_test_subset)

        with self.assertRaises(RuntimeError):
            cp.calibrate(x_test_subset, y_test[: len(x_test_subset)])

        x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
            self.x_reg, self.y_reg, test_size=0.3, random_state=42
        )
        reg = RandomForestRegressor(random_state=42, n_estimators=5)
        cp_reg = ConformalPredictor(reg, estimator_type="regressor")
        cp_reg.fit(x_train_reg, y_train_reg)

        x_test_reg_subset, _, _, _ = train_test_split(
            x_test_reg, y_test_reg, test_size=0.5, random_state=42
        )
        with self.assertRaises(NotImplementedError):
            cp_reg.predict_proba(x_test_reg_subset)

        cp.fit(x_train, y_train)
        with self.assertRaises(NotImplementedError):
            cp.predict_int(x_test_subset)


class TestCrossConformalPredictorCore(BaseConformalTestData):
    """Core functionality tests for CrossConformalPredictor."""

    def test_cross_conformal_classifier(self) -> None:
        """Test CrossConformalPredictor with a classifier."""
        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        ccp = CrossConformalPredictor(clf, estimator_type="classifier", n_folds=3)
        ccp.fit(self.x_clf, self.y_clf)

        preds = ccp.predict(self.x_clf)
        probs = ccp.predict_proba(self.x_clf)
        sets = ccp.predict_conformal_set(self.x_clf)
        p_values = ccp.predict_p(self.x_clf)

        self.assertEqual(len(preds), len(self.y_clf))
        self.assertEqual(probs.shape[0], len(self.y_clf))
        self.assertEqual(len(sets), len(self.y_clf))
        self.assertEqual(len(p_values), len(self.y_clf))

    def test_cross_conformal_regressor(self) -> None:
        """Test CrossConformalPredictor with a regressor."""
        reg = RandomForestRegressor(random_state=42, n_estimators=5)
        ccp = CrossConformalPredictor(reg, estimator_type="regressor", n_folds=3)
        ccp.fit(self.x_reg, self.y_reg)

        intervals = ccp.predict_int(self.x_reg)

        for model in ccp.models_:
            model_intervals = model.predict_int(self.x_reg)
            self.assertEqual(model_intervals.shape[0], len(self.y_reg))
            self.assertEqual(model_intervals.shape[1], 2)

        self.assertEqual(intervals.shape[0], len(self.y_reg))
        self.assertEqual(intervals.shape[1], 2)

    def test_cross_conformal_confidence_effect_regression(self) -> None:
        """Test confidence level effect in cross-conformal regression."""
        reg = RandomForestRegressor(random_state=42, n_estimators=5)
        ccp = CrossConformalPredictor(reg, estimator_type="regressor", n_folds=3)
        ccp.fit(self.x_reg, self.y_reg)
        intervals_90 = ccp.predict_int(self.x_reg, confidence=0.90)
        intervals_95 = ccp.predict_int(self.x_reg, confidence=0.95)
        width_90 = float(np.mean(intervals_90[:, 1] - intervals_90[:, 0]))
        width_95 = float(np.mean(intervals_95[:, 1] - intervals_95[:, 0]))
        self.assertLess(width_90, width_95)

    def test_cross_conformal_confidence_effect_classification(self) -> None:
        """Test confidence level effect in cross-conformal classification."""
        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        ccp = CrossConformalPredictor(clf, estimator_type="classifier", n_folds=3)
        ccp.fit(self.x_clf, self.y_clf)
        sets_90 = ccp.predict_conformal_set(self.x_clf, confidence=0.90)
        sets_95 = ccp.predict_conformal_set(self.x_clf, confidence=0.95)
        size_90 = float(np.mean([len(s) for s in sets_90]))
        size_95 = float(np.mean([len(s) for s in sets_95]))

        self.assertLessEqual(size_90, size_95)

    def test_cross_conformal_mondrian_classification(self) -> None:
        """Test Mondrian vs baseline CrossConformalPredictor for classification."""
        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        mc_clf = MondrianCategorizer()
        mc_clf.fit(
            self.x_clf,
            f=lambda x: (x[:, 0] > np.median(x[:, 0])).astype(int),
            no_bins=2,
        )
        ccp_clf = CrossConformalPredictor(
            clf,
            estimator_type="classifier",
            n_folds=3,
            mondrian=mc_clf,
            random_state=42,
        )
        ccp_clf.fit(self.x_clf, self.y_clf)
        x_test_subset, _, _, _ = train_test_split(
            self.x_clf, self.y_clf, test_size=0.1, random_state=42
        )
        sets_mondrian = ccp_clf.predict_conformal_set(x_test_subset)
        p_values_mondrian = ccp_clf.predict_p(x_test_subset)
        ccp_clf_baseline = CrossConformalPredictor(
            clf, estimator_type="classifier", n_folds=3, mondrian=False, random_state=42
        )
        ccp_clf_baseline.fit(self.x_clf, self.y_clf)
        sets_baseline = ccp_clf_baseline.predict_conformal_set(x_test_subset)
        self.assertEqual(len(sets_mondrian), len(sets_baseline))
        self.assertEqual(len(p_values_mondrian), len(x_test_subset))

    def test_cross_conformal_mondrian_regression(self) -> None:
        """Test Mondrian-style binning vs baseline CrossConformalPredictor for regression."""
        reg = RandomForestRegressor(random_state=42, n_estimators=5)

        ccp_reg = CrossConformalPredictor(
            reg, estimator_type="regressor", n_folds=3, binning=3, random_state=42
        )
        ccp_reg.fit(self.x_reg, self.y_reg)

        x_test_subset, _, _, _ = train_test_split(
            self.x_reg, self.y_reg, test_size=0.1, random_state=42
        )
        intervals_binned = ccp_reg.predict_int(x_test_subset)

        ccp_reg_baseline = CrossConformalPredictor(
            reg, estimator_type="regressor", n_folds=3, binning=None, random_state=42
        )
        ccp_reg_baseline.fit(self.x_reg, self.y_reg)
        intervals_baseline_reg = ccp_reg_baseline.predict_int(x_test_subset)

        self.assertEqual(intervals_binned.shape, (len(x_test_subset), 2))
        self.assertEqual(intervals_baseline_reg.shape, (len(x_test_subset), 2))


class TestConformalPipelineIntegration(BaseConformalTestData):
    """Pipeline integration tests for conformal prediction."""

    def test_pipeline_wrapped_by_conformal_classifier(self) -> None:  # pylint: disable=too-many-locals
        """Test a MolPipeline wrapped by ConformalPredictor."""
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
                        FilterReinserter.from_error_filter(error_filter, np.nan)
                    ),
                ),
            ]
        )

        conformal_pipeline = ConformalPredictor(
            base_pipeline, estimator_type="classifier"
        )

        train_smiles, temp_smiles, y_train, y_temp = train_test_split(
            self.smiles_clf, self.y_clf, test_size=0.4, random_state=42
        )
        calib_smiles, test_smiles, y_calib, y_test = train_test_split(
            temp_smiles, y_temp, test_size=0.5, random_state=42
        )

        conformal_pipeline.fit(train_smiles, y_train)
        conformal_pipeline.calibrate(calib_smiles, y_calib)

        preds = conformal_pipeline.predict(test_smiles)
        sets = conformal_pipeline.predict_conformal_set(test_smiles)
        p_values = conformal_pipeline.predict_p(test_smiles)

        self.assertEqual(len(preds), len(y_test))
        self.assertEqual(len(sets), len(y_test))
        self.assertEqual(len(p_values), len(y_test))

        invalid_smiles = ["C1CC", "X", ""]
        test_smiles_with_invalid = test_smiles + invalid_smiles
        predicted_with_invalid = conformal_pipeline.predict(test_smiles_with_invalid)

        self.assertEqual(len(predicted_with_invalid), len(test_smiles_with_invalid))

    def test_pipeline_wrapped_by_conformal_regressor(self) -> None:  # pylint: disable=too-many-locals
        """Test a MolPipeline wrapped by ConformalPredictor for regression."""
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
                        FilterReinserter.from_error_filter(error_filter, np.nan)
                    ),
                ),
            ]
        )

        conformal_pipeline = ConformalPredictor(
            base_pipeline, estimator_type="regressor"
        )

        train_smiles, temp_smiles, y_train, y_temp = train_test_split(
            self.smiles_reg, self.y_reg, test_size=0.4, random_state=42
        )
        calib_smiles, test_smiles, y_calib, y_test = train_test_split(
            temp_smiles, y_temp, test_size=0.5, random_state=42
        )
        conformal_pipeline.fit(train_smiles, y_train)
        conformal_pipeline.calibrate(calib_smiles, y_calib)
        preds = conformal_pipeline.predict(test_smiles)
        intervals = conformal_pipeline.predict_int(test_smiles)
        self.assertEqual(len(preds), len(y_test))
        self.assertEqual(intervals.shape, (len(y_test), 2))

    def test_cross_conformal_wrapping_pipeline(self) -> None:
        """Test CrossConformalPredictor wrapping a complete pipeline."""
        base_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                (
                    "morgan",
                    MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense"),
                ),
                ("classifier", RandomForestClassifier(n_estimators=5, random_state=42)),
            ]
        )

        cross_conformal = CrossConformalPredictor(
            base_pipeline, estimator_type="classifier", n_folds=2, random_state=42
        )

        train_smiles, test_smiles, train_labels, _test_labels = train_test_split(
            self.smiles_clf, self.y_clf, test_size=0.3, random_state=42
        )

        cross_conformal.fit(train_smiles, train_labels)
        preds = cross_conformal.predict(test_smiles)
        sets = cross_conformal.predict_conformal_set(test_smiles)

        self.assertEqual(len(preds), len(test_smiles))
        self.assertEqual(len(sets), len(test_smiles))

    def test_conformal_in_pipeline_like_ml_experiments(self) -> None:
        """Test conformal predictor as part of pipeline."""
        conformal_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                (
                    "mol2morgan",
                    MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense"),
                ),
                (
                    "conformal_classifier",
                    ConformalPredictor(
                        RandomForestClassifier(n_estimators=5, random_state=42),
                        estimator_type="classifier",
                    ),
                ),
            ]
        )

        train_smiles, calib_smiles, y_train, y_calib = train_test_split(
            self.smiles_clf, self.y_clf, test_size=0.2, random_state=42
        )

        x_train = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                (
                    "mol2morgan",
                    MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense"),
                ),
            ]
        ).fit_transform(train_smiles)

        conformal_pipeline.fit(train_smiles, y_train)

        conformal_pipeline.named_steps["conformal_classifier"].calibrate(
            x_train, y_train
        )

        preds = conformal_pipeline.predict(calib_smiles)

        self.assertEqual(len(preds), len(y_calib))

    def test_cross_conformal_in_pipeline_like_ml_experiments(self) -> None:
        """Test CrossConformalPredictor as part of pipeline."""
        cross_conformal_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                (
                    "mol2morgan",
                    MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense"),
                ),
                (
                    "cross_conformal_regressor",
                    CrossConformalPredictor(
                        RandomForestRegressor(n_estimators=5, random_state=42),
                        estimator_type="regressor",
                        n_folds=2,
                        random_state=42,
                    ),
                ),
            ]
        )
        train_smiles, test_smiles, train_values, _test_values = train_test_split(
            self.smiles_reg, self.y_reg, test_size=0.25, random_state=42
        )
        cross_conformal_pipeline.fit(train_smiles, train_values)
        preds = cross_conformal_pipeline.predict(test_smiles)

        intervals = cross_conformal_pipeline.named_steps[
            "cross_conformal_regressor"
        ].predict_int(
            Pipeline(
                [
                    ("smi2mol", SmilesToMol()),
                    (
                        "mol2morgan",
                        MolToMorganFP(
                            radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense"
                        ),
                    ),
                ]
            ).fit_transform(test_smiles)
        )

        self.assertEqual(len(preds), len(test_smiles))
        self.assertEqual(intervals.shape, (len(test_smiles), 2))


class TestConformalSerialization(BaseConformalTestData):
    """Serialization tests for conformal prediction (JSON and joblib)."""

    def test_json_serialization_conformal_predictor(self) -> None:
        """Test JSON serialization of ConformalPredictor."""
        x_train, x_calib, x_test, y_train, y_calib, y_test = (
            self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        )

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        cp = ConformalPredictor(clf, estimator_type="classifier")
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)

        test_preds = cp.predict(x_test)
        json_str = recursive_to_json(cp)
        loaded_cp = recursive_from_json(json_str)
        self.assertIsInstance(loaded_cp, ConformalPredictor)
        self.assertEqual(loaded_cp.estimator_type, "classifier")
        self.assertEqual(loaded_cp.confidence_level, 0.9)

        self.assertEqual(len(test_preds), len(y_test))

    def test_json_serialization_cross_conformal_predictor(self) -> None:
        """Test JSON serialization of CrossConformalPredictor.

        Note: JSON serialization preserves configuration only, not fitted state.
        Use joblib for fitted model persistence.
        """
        clf = RandomForestClassifier(n_estimators=5, random_state=42)

        ccp = CrossConformalPredictor(
            estimator=clf,
            estimator_type="classifier",
            n_folds=2,
            random_state=42,
            confidence_level=0.8,
        )
        ccp.fit(self.x_clf, self.y_clf)
        json_str = recursive_to_json(ccp)
        loaded_ccp = recursive_from_json(json_str)

        self.assertIsInstance(loaded_ccp, CrossConformalPredictor)
        self.assertEqual(loaded_ccp.estimator_type, "classifier")
        self.assertEqual(loaded_ccp.n_folds, 2)
        self.assertEqual(loaded_ccp.confidence_level, 0.8)
        self.assertEqual(loaded_ccp.random_state, 42)

        self.assertEqual(loaded_ccp.difficulty_estimator, None)
        self.assertEqual(loaded_ccp.n_jobs, 1)

        self.assertIsInstance(loaded_ccp.estimator, RandomForestClassifier)
        self.assertEqual(loaded_ccp.estimator.n_estimators, 5)
        self.assertEqual(loaded_ccp.estimator.random_state, 42)

    def test_json_serialization_cross_conformal_predictor_regression(self) -> None:
        """Test JSON serialization of CrossConformalPredictor for regression."""
        reg = RandomForestRegressor(n_estimators=5, random_state=42)

        ccp = CrossConformalPredictor(
            estimator=reg,
            estimator_type="regressor",
            n_folds=2,
            random_state=42,
            confidence_level=0.9,
        )

        ccp.fit(self.x_reg, self.y_reg)
        json_str = recursive_to_json(ccp)
        loaded_ccp = recursive_from_json(json_str)
        self.assertIsInstance(loaded_ccp, CrossConformalPredictor)
        self.assertEqual(loaded_ccp.estimator_type, "regressor")
        self.assertEqual(loaded_ccp.n_folds, 2)
        self.assertEqual(loaded_ccp.confidence_level, 0.9)
        self.assertEqual(loaded_ccp.random_state, 42)
        self.assertEqual(loaded_ccp.difficulty_estimator, None)
        self.assertEqual(loaded_ccp.n_jobs, 1)
        self.assertIsInstance(loaded_ccp.estimator, RandomForestRegressor)
        self.assertEqual(loaded_ccp.estimator.n_estimators, 5)
        self.assertEqual(loaded_ccp.estimator.random_state, 42)

    def test_json_serialization_pipeline_wrapped_by_conformal(self) -> None:
        """Test JSON serialization of Pipeline wrapped by ConformalPredictor."""
        base_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                (
                    "morgan",
                    MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense"),
                ),
                ("classifier", RandomForestClassifier(n_estimators=5, random_state=42)),
            ]
        )

        conformal_wrapper = ConformalPredictor(
            base_pipeline, estimator_type="classifier"
        )

        train_smiles, temp_smiles, train_labels, temp_labels = train_test_split(
            self.smiles_clf, self.y_clf, test_size=0.4, random_state=42
        )
        calib_smiles, test_smiles, calib_labels, _test_labels = train_test_split(
            temp_smiles, temp_labels, test_size=0.5, random_state=42
        )

        conformal_wrapper.fit(train_smiles, train_labels)
        conformal_wrapper.calibrate(calib_smiles, calib_labels)
        test_preds = conformal_wrapper.predict(test_smiles)
        json_str = recursive_to_json(conformal_wrapper)
        loaded_wrapper = recursive_from_json(json_str)
        self.assertIsInstance(loaded_wrapper, ConformalPredictor)
        self.assertEqual(len(test_preds), len(test_smiles))

    def test_json_serialization_conformal_in_pipeline(self) -> None:
        """Test JSON serialization of Pipeline containing ConformalPredictor."""
        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                (
                    "morgan",
                    MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense"),
                ),
                (
                    "conformal",
                    ConformalPredictor(
                        RandomForestClassifier(n_estimators=5, random_state=42),
                        estimator_type="classifier",
                    ),
                ),
            ]
        )

        train_smiles, temp_smiles, train_labels, temp_labels = train_test_split(
            self.smiles_clf, self.y_clf, test_size=0.4, random_state=42
        )
        calib_smiles, test_smiles, calib_labels, _test_labels = train_test_split(
            temp_smiles, temp_labels, test_size=0.5, random_state=42
        )

        pipeline.fit(train_smiles, train_labels)

        calib_fps = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                (
                    "morgan",
                    MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense"),
                ),
            ]
        ).fit_transform(calib_smiles)
        pipeline.named_steps["conformal"].calibrate(calib_fps, calib_labels)

        test_preds = pipeline.predict(test_smiles)
        json_str = recursive_to_json(pipeline)
        loaded_pipeline = recursive_from_json(json_str)
        self.assertIsInstance(loaded_pipeline, Pipeline)
        self.assertEqual(len(test_preds), len(test_smiles))

    def test_joblib_serialization_conformal_predictor(self) -> None:  # pylint: disable=too-many-locals
        """Test joblib serialization of ConformalPredictor."""
        x_train, x_calib, x_test, y_train, y_calib, _y_test = (
            self._get_train_calib_test_splits(self.x_clf, self.y_clf)
        )
        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        cp = ConformalPredictor(clf, estimator_type="classifier")
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)
        original_preds = cp.predict(x_test)
        original_sets = cp.predict_conformal_set(x_test)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "conformal_model.joblib"
            joblib.dump(cp, model_path)
            loaded_cp = joblib.load(model_path)
            self.assertIsInstance(loaded_cp, ConformalPredictor)
            loaded_preds = loaded_cp.predict(x_test)
            loaded_sets = loaded_cp.predict_conformal_set(x_test)
            self.assertTrue(np.array_equal(original_preds, loaded_preds))
            self.assertEqual(len(original_sets), len(loaded_sets))
            for orig_set, loaded_set in zip(original_sets, loaded_sets):
                orig_set_items = set(orig_set.tolist())
                loaded_set_items = set(loaded_set.tolist())
                if len(orig_set_items) > 0 or len(loaded_set_items) > 0:
                    all_items = orig_set_items.union(loaded_set_items)
                    self.assertTrue(
                        all(isinstance(item, (int, np.integer)) for item in all_items)
                    )
                    self.assertTrue(all(0 <= item <= 1 for item in all_items))

    def test_joblib_serialization_cross_conformal_predictor(self) -> None:
        """Test joblib serialization of CrossConformalPredictor."""
        x_train, x_test, y_train, _y_test = train_test_split(
            self.x_reg, self.y_reg, test_size=0.3, random_state=42
        )
        reg = RandomForestRegressor(n_estimators=5, random_state=42)
        ccp = CrossConformalPredictor(
            reg, estimator_type="regressor", n_folds=2, random_state=42
        )
        ccp.fit(x_train, y_train)
        original_preds = ccp.predict(x_test)
        original_intervals = ccp.predict_int(x_test)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "cross_conformal_model.joblib"
            joblib.dump(ccp, model_path)
            loaded_ccp = joblib.load(model_path)
            self.assertIsInstance(loaded_ccp, CrossConformalPredictor)
            loaded_preds = loaded_ccp.predict(x_test)
            loaded_intervals = loaded_ccp.predict_int(x_test)
            self.assertTrue(np.array_equal(original_preds, loaded_preds))
            self.assertTrue(np.allclose(original_intervals, loaded_intervals))

    def test_joblib_serialization_pipeline_wrapped_by_conformal(self) -> None:  # pylint: disable=too-many-locals
        """Test joblib serialization of Pipeline wrapped by ConformalPredictor."""
        base_pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                (
                    "morgan",
                    MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense"),
                ),
                ("classifier", RandomForestClassifier(n_estimators=5, random_state=42)),
            ]
        )

        conformal_wrapper = ConformalPredictor(
            base_pipeline, estimator_type="classifier"
        )

        train_smiles, temp_smiles, train_labels, temp_labels = train_test_split(
            self.smiles_clf, self.y_clf, test_size=0.4, random_state=42
        )
        calib_smiles, test_smiles, calib_labels, _test_labels = train_test_split(
            temp_smiles, temp_labels, test_size=0.5, random_state=42
        )

        conformal_wrapper.fit(train_smiles, train_labels)
        conformal_wrapper.calibrate(calib_smiles, calib_labels)
        original_preds = conformal_wrapper.predict(test_smiles)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "wrapped_pipeline.joblib"
            joblib.dump(conformal_wrapper, model_path)
            loaded_wrapper = joblib.load(model_path)
            loaded_preds = loaded_wrapper.predict(test_smiles)
            self.assertTrue(np.array_equal(original_preds, loaded_preds))

    def test_joblib_serialization_conformal_in_pipeline(self) -> None:  # pylint: disable=too-many-locals
        """Test joblib serialization of Pipeline containing ConformalPredictor."""
        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                (
                    "morgan",
                    MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense"),
                ),
                (
                    "conformal",
                    ConformalPredictor(
                        RandomForestClassifier(n_estimators=5, random_state=42),
                        estimator_type="classifier",
                    ),
                ),
            ]
        )

        train_smiles, temp_smiles, train_labels, temp_labels = train_test_split(
            self.smiles_clf, self.y_clf, test_size=0.4, random_state=42
        )
        calib_smiles, test_smiles, calib_labels, _test_labels = train_test_split(
            temp_smiles, temp_labels, test_size=0.5, random_state=42
        )

        pipeline.fit(train_smiles, train_labels)

        calib_fps = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                (
                    "morgan",
                    MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, return_as="dense"),
                ),
            ]
        ).fit_transform(calib_smiles)
        pipeline.named_steps["conformal"].calibrate(calib_fps, calib_labels)
        original_preds = pipeline.predict(test_smiles)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "pipeline_with_conformal.joblib"
            joblib.dump(pipeline, model_path)
            loaded_pipeline = joblib.load(model_path)
            loaded_preds = loaded_pipeline.predict(test_smiles)
            self.assertTrue(np.array_equal(original_preds, loaded_preds))


if __name__ == "__main__":
    unittest.main()
