"""Unit tests for conformal prediction wrappers."""

import unittest
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from crepes.extras import MondrianCategorizer, hinge, margin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from molpipeline.any2mol import SmilesToMol
from molpipeline.experimental.uncertainty.conformal import (
    ConformalPredictor,
    CrossConformalPredictor,
)
from molpipeline.mol2any import MolToMorganFP

# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "test_data"

# Constants for fingerprints
FP_RADIUS = 2
FP_SIZE = 1024


class TestConformalCV(unittest.TestCase):
    """Unit tests for ConformalPredictor and CrossConformalPredictor wrappers."""

    # Class attributes for test data
    x_clf: npt.NDArray[Any]
    y_clf: npt.NDArray[Any]
    x_reg: npt.NDArray[Any]
    y_reg: npt.NDArray[Any]

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test data once for all tests."""
        # Load data
        bbbp_df = pd.read_csv(TEST_DATA_DIR / "molecule_net_bbbp.tsv.gz",
                              sep="\t", compression="gzip")
        logd_df = pd.read_csv(TEST_DATA_DIR / "molecule_net_logd.tsv.gz",
                              sep="\t", compression="gzip")

        # Set up pipeline stages separately to handle invalid molecules
        smi2mol = SmilesToMol(n_jobs=1)
        morgan = MolToMorganFP(radius=FP_RADIUS, n_bits=FP_SIZE, n_jobs=1)

        # Process classification data
        bbbp_clean = bbbp_df.dropna(subset=["smiles", "p_np"])
        smiles_list = bbbp_clean["smiles"].tolist()
        labels_list = bbbp_clean["p_np"].tolist()

        # Convert SMILES to molecules first, filter out invalid ones
        molecules = smi2mol.fit_transform(smiles_list)
        valid_clf_data = []

        for mol, label in zip(molecules, labels_list, strict=False):
            # Skip InvalidInstance objects
            if mol is None or hasattr(mol, "_fields"):  # InvalidInstance is a NamedTuple
                continue
            # Generate fingerprint for valid molecule
            try:
                fp = morgan.transform([mol])[0]  # type: ignore[list-item]
                if fp is not None and hasattr(fp, "toarray"):
                    valid_clf_data.append((fp.toarray().flatten(), label))
            except (AttributeError, TypeError):
                # Skip molecules that can't be processed
                continue

        if not valid_clf_data:
            raise ValueError("No valid classification data found")

        cls.x_clf, cls.y_clf = map(np.array, zip(*valid_clf_data, strict=False))

        # Process regression data
        logd_clean = logd_df.dropna(subset=["smiles", "exp"])
        smiles_list_reg = logd_clean["smiles"].tolist()
        labels_list_reg = logd_clean["exp"].tolist()

        # Convert SMILES to molecules first, filter out invalid ones
        molecules_reg = smi2mol.transform(smiles_list_reg)
        valid_reg_data = []

        for mol, label in zip(molecules_reg, labels_list_reg, strict=False):
            # Skip InvalidInstance objects
            if mol is None or hasattr(mol, "_fields"):  # InvalidInstance is a NamedTuple
                continue
            # Generate fingerprint for valid molecule - ensure mol is valid
            try:
                fp = morgan.transform([mol])[0]  # type: ignore[list-item]
                if fp is not None and hasattr(fp, "toarray"):
                    valid_reg_data.append((fp.toarray().flatten(), label))
            except (AttributeError, TypeError):
                # Skip molecules that can't be processed
                continue

        if not valid_reg_data:
            raise ValueError("No valid regression data found")

        cls.x_reg, cls.y_reg = map(np.array, zip(*valid_reg_data, strict=False))

    def test_conformal_prediction_classifier(self) -> None:
        """Test ConformalPredictor with a classifier."""
        x_train, x_calib, y_train, y_calib = train_test_split(
            self.x_clf,
            self.y_clf,
            test_size=0.2,
            random_state=42,
        )
        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        cp = ConformalPredictor(clf, estimator_type="classifier")
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)
        preds = cp.predict(x_calib)
        probs = cp.predict_proba(x_calib)
        sets = cp.predict_conformal_set(x_calib)
        p_values = cp.predict_p(x_calib)

        self.assertEqual(len(preds), len(y_calib))
        self.assertEqual(probs.shape[0], len(y_calib))
        self.assertEqual(len(sets), len(y_calib))
        self.assertEqual(len(p_values), len(y_calib))

    def test_conformal_prediction_regressor(self) -> None:
        """Test ConformalPredictor with a regressor."""
        x_train, x_calib, y_train, y_calib = train_test_split(
            self.x_reg,
            self.y_reg,
            test_size=0.2,
            random_state=42,
        )
        reg = RandomForestRegressor(random_state=42, n_estimators=5)
        cp = ConformalPredictor(reg, estimator_type="regressor")
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)
        intervals = cp.predict_int(x_calib)

        self.assertEqual(intervals.shape[0], len(y_calib))
        self.assertEqual(intervals.shape[1], 2)

    def test_confidence_level_effect_regression(self) -> None:
        """Test that increasing confidence level increases interval width."""
        x_train, x_calib, y_train, y_calib = train_test_split(
            self.x_reg,
            self.y_reg,
            test_size=0.2,
            random_state=42,
        )
        reg = RandomForestRegressor(random_state=42, n_estimators=5)
        cp = ConformalPredictor(reg, estimator_type="regressor")
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)

        # Test different confidence levels
        intervals_90 = cp.predict_int(x_calib, confidence=0.90)
        intervals_95 = cp.predict_int(x_calib, confidence=0.95)
        intervals_99 = cp.predict_int(x_calib, confidence=0.99)

        # Calculate average interval widths
        width_90 = float(np.mean(intervals_90[:, 1] - intervals_90[:, 0]))
        width_95 = float(np.mean(intervals_95[:, 1] - intervals_95[:, 0]))
        width_99 = float(np.mean(intervals_99[:, 1] - intervals_99[:, 0]))

        # Higher confidence should lead to wider intervals
        self.assertLess(width_90, width_95)
        self.assertLess(width_95, width_99)

    def test_confidence_level_effect_classification(self) -> None:
        """Test that lower confidence level increases prediction set size."""
        x_train, x_calib, y_train, y_calib = train_test_split(
            self.x_clf,
            self.y_clf,
            test_size=0.2,
            random_state=42,
        )
        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        cp = ConformalPredictor(clf, estimator_type="classifier")
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)

        # Test different confidence levels
        sets_90 = cp.predict_conformal_set(x_calib, confidence=0.90)
        sets_95 = cp.predict_conformal_set(x_calib, confidence=0.95)
        sets_99 = cp.predict_conformal_set(x_calib, confidence=0.99)

        # Calculate average prediction set sizes
        size_90 = float(np.mean([len(s) for s in sets_90]))
        size_95 = float(np.mean([len(s) for s in sets_95]))
        size_99 = float(np.mean([len(s) for s in sets_99]))
        # Higher confidence should lead to larger prediction sets
        self.assertLessEqual(size_90, size_95)
        self.assertLessEqual(size_95, size_99)

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

        # Each model should produce intervals for all samples
        for model in ccp.models_:
            model_intervals = model.predict_int(self.x_reg)
            self.assertEqual(model_intervals.shape[0], len(self.y_reg))
            self.assertEqual(model_intervals.shape[1], 2)

        # Aggregated intervals should have correct shape
        self.assertEqual(intervals.shape[0], len(self.y_reg))
        self.assertEqual(intervals.shape[1], 2)

    def test_cross_conformal_confidence_effect_regression(self) -> None:
        """Test confidence level effect in cross-conformal regression."""
        reg = RandomForestRegressor(random_state=42, n_estimators=5)
        ccp = CrossConformalPredictor(reg, estimator_type="regressor", n_folds=3)
        ccp.fit(self.x_reg, self.y_reg)

        # Test different confidence levels
        intervals_90 = ccp.predict_int(self.x_reg, confidence=0.90)
        intervals_95 = ccp.predict_int(self.x_reg, confidence=0.95)
        intervals_99 = ccp.predict_int(self.x_reg, confidence=0.99)

        # Calculate average interval widths
        width_90 = float(np.mean(intervals_90[:, 1] - intervals_90[:, 0]))
        width_95 = float(np.mean(intervals_95[:, 1] - intervals_95[:, 0]))
        width_99 = float(np.mean(intervals_99[:, 1] - intervals_99[:, 0]))

        # Higher confidence should lead to wider intervals
        self.assertLess(width_90, width_95)
        self.assertLess(width_95, width_99)

    def test_cross_conformal_confidence_effect_classification(self) -> None:
        """Test confidence level effect in cross-conformal classification."""
        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        ccp = CrossConformalPredictor(clf, estimator_type="classifier", n_folds=3)
        ccp.fit(self.x_clf, self.y_clf)

        # Test different confidence levels
        sets_90 = ccp.predict_conformal_set(self.x_clf, confidence=0.90)
        sets_95 = ccp.predict_conformal_set(self.x_clf, confidence=0.95)
        sets_99 = ccp.predict_conformal_set(self.x_clf, confidence=0.99)

        # Calculate average prediction set sizes
        size_90 = float(np.mean([len(s) for s in sets_90]))
        size_95 = float(np.mean([len(s) for s in sets_95]))
        size_99 = float(np.mean([len(s) for s in sets_99]))

        # Higher confidence should lead to larger prediction sets
        self.assertLessEqual(size_90, size_95)
        self.assertLessEqual(size_95, size_99)

    def test_auto_detection(self) -> None:
        """Test automatic estimator type detection."""
        # Test classifier auto-detection
        clf = RandomForestClassifier(random_state=42)
        cp_clf = ConformalPredictor(clf, estimator_type="auto")
        self.assertEqual(cp_clf.estimator_type, "classifier")

        # Test regressor auto-detection
        reg = RandomForestRegressor(random_state=42)
        cp_reg = ConformalPredictor(reg, estimator_type="auto")
        self.assertEqual(cp_reg.estimator_type, "regressor")

    def test_nonconformity_functions(self) -> None:
        """Test nonconformity functions for classification."""
        x_train, x_calib, y_train, y_calib = train_test_split(
            self.x_clf, self.y_clf, test_size=0.2, random_state=42,
        )

        clf = RandomForestClassifier(random_state=42, n_estimators=5)

        # Test with hinge nonconformity
        cp_hinge = ConformalPredictor(clf, estimator_type="classifier",
                                      nonconformity=hinge)
        cp_hinge.fit(x_train, y_train)
        cp_hinge.calibrate(x_calib, y_calib)
        sets_hinge = cp_hinge.predict_conformal_set(x_calib)
        p_values_hinge = cp_hinge.predict_p(x_calib)

        # Test with margin nonconformity
        cp_margin = ConformalPredictor(clf, estimator_type="classifier",
                                       nonconformity=margin)
        cp_margin.fit(x_train, y_train)
        cp_margin.calibrate(x_calib, y_calib)
        sets_margin = cp_margin.predict_conformal_set(x_calib)
        p_values_margin = cp_margin.predict_p(x_calib)

        # Verify outputs have correct shapes
        self.assertEqual(len(sets_hinge), len(y_calib))
        self.assertEqual(len(sets_margin), len(y_calib))
        self.assertEqual(len(p_values_hinge), len(y_calib))
        self.assertEqual(len(p_values_margin), len(y_calib))

        # Different nonconformity functions should give different results
        self.assertNotEqual(sets_hinge, sets_margin)

    def test_mondrian_conformal_classification(self) -> None:
        """Test Mondrian conformal prediction for classification."""
        x_train, x_calib, y_train, y_calib = train_test_split(
            self.x_clf, self.y_clf, test_size=0.2, random_state=42,
        )

        clf = RandomForestClassifier(random_state=42, n_estimators=5)

        # Test with custom MondrianCategorizer (skip mondrian=True for now)
        mc = MondrianCategorizer()
        # Simple categorizer based on first feature
        mc.fit(x_calib,
               f=lambda x: (x[:, 0] > np.median(x[:, 0])).astype(int),
               no_bins=2)

        cp_mondrian_custom = ConformalPredictor(clf, estimator_type="classifier",
                                                mondrian=mc)
        cp_mondrian_custom.fit(x_train, y_train)
        cp_mondrian_custom.calibrate(x_calib, y_calib)
        sets_custom = cp_mondrian_custom.predict_conformal_set(x_calib)
        p_values_custom = cp_mondrian_custom.predict_p(x_calib)

        # Test without Mondrian (baseline)
        cp_baseline = ConformalPredictor(clf, estimator_type="classifier",
                                         mondrian=False)
        cp_baseline.fit(x_train, y_train)
        cp_baseline.calibrate(x_calib, y_calib)
        sets_baseline = cp_baseline.predict_conformal_set(x_calib)

        # Verify outputs have correct shapes
        self.assertEqual(len(sets_custom), len(sets_baseline))
        self.assertEqual(len(p_values_custom), len(y_calib))

        # Verify that prediction sets contain valid class indices
        for pred_set in sets_custom:
            self.assertIsInstance(pred_set, list)
            for class_idx in pred_set:
                self.assertIsInstance(class_idx, (int, np.integer))
                self.assertGreaterEqual(class_idx, 0)

        self.assertTrue(np.all(p_values_custom >= 0))
        self.assertTrue(np.all(p_values_custom <= 1))

    def test_mondrian_conformal_regression(self) -> None:
        """Test Mondrian conformal prediction for regression."""
        x_train, x_calib, y_train, y_calib = train_test_split(
            self.x_reg, self.y_reg, test_size=0.2, random_state=42,
        )

        reg = RandomForestRegressor(random_state=42, n_estimators=5)

        # Test with custom MondrianCategorizer for regression
        mc = MondrianCategorizer()
        # Categorize based on median of first feature
        mc.fit(x_calib,
               f=lambda x: (x[:, 0] > np.median(x[:, 0])).astype(int),
               no_bins=2)

        cp_mondrian = ConformalPredictor(reg, estimator_type="regressor",
                                         mondrian=mc)
        cp_mondrian.fit(x_train, y_train)
        cp_mondrian.calibrate(x_calib, y_calib)
        intervals_mondrian = cp_mondrian.predict_int(x_calib)

        # Test without Mondrian (baseline)
        cp_baseline = ConformalPredictor(reg, estimator_type="regressor",
                                         mondrian=False)
        cp_baseline.fit(x_train, y_train)
        cp_baseline.calibrate(x_calib, y_calib)
        intervals_baseline = cp_baseline.predict_int(x_calib)

        # Verify outputs have correct shapes
        self.assertEqual(intervals_mondrian.shape, (len(y_calib), 2))
        self.assertEqual(intervals_baseline.shape, (len(y_calib), 2))

        # Mondrian should give different results than baseline
        self.assertFalse(np.array_equal(intervals_mondrian, intervals_baseline))

    def test_cross_conformal_mondrian_both_classes(self) -> None:
        """Test Mondrian with CrossConformalPredictors."""
        # Test classification with custom MondrianCategorizer
        clf = RandomForestClassifier(random_state=42, n_estimators=5)

        # Create a simple Mondrian categorizer for classification
        mc_clf = MondrianCategorizer()
        mc_clf.fit(self.x_clf,
                   f=lambda x: (x[:, 0] > np.median(x[:, 0])).astype(int),
                   no_bins=2)

        ccp_clf = CrossConformalPredictor(clf, estimator_type="classifier",
                                          n_folds=3, mondrian=mc_clf, random_state=42)
        ccp_clf.fit(self.x_clf, self.y_clf)
        sets_mondrian = ccp_clf.predict_conformal_set(self.x_clf[:10])
        p_values_mondrian = ccp_clf.predict_p(self.x_clf[:10])

        # Test without Mondrian for comparison
        ccp_clf_baseline = CrossConformalPredictor(clf, estimator_type="classifier",
                                                   n_folds=3, mondrian=False,
                                                   random_state=42)
        ccp_clf_baseline.fit(self.x_clf, self.y_clf)
        sets_baseline = ccp_clf_baseline.predict_conformal_set(self.x_clf[:10])

        # Verify shapes
        self.assertEqual(len(sets_mondrian), len(sets_baseline))
        self.assertEqual(len(p_values_mondrian), 10)

        # Test regression with binning (Mondrian-style for regression)
        reg = RandomForestRegressor(random_state=42, n_estimators=5)
        ccp_reg = CrossConformalPredictor(reg, estimator_type="regressor",
                                          n_folds=3, binning=3, random_state=42)
        ccp_reg.fit(self.x_reg, self.y_reg)
        intervals_binned = ccp_reg.predict_int(self.x_reg[:10])

        # Test without binning for comparison
        ccp_reg_baseline = CrossConformalPredictor(reg, estimator_type="regressor",
                                                   n_folds=3, binning=None,
                                                   random_state=42)
        ccp_reg_baseline.fit(self.x_reg, self.y_reg)
        intervals_baseline_reg = ccp_reg_baseline.predict_int(self.x_reg[:10])

        # Verify shapes
        self.assertEqual(intervals_binned.shape, (10, 2))
        self.assertEqual(intervals_baseline_reg.shape, (10, 2))

    def test_error_handling(self) -> None:
        """Test error handling for various invalid operations."""
        clf = RandomForestClassifier(random_state=42, n_estimators=5)
        cp = ConformalPredictor(clf, estimator_type="classifier")

        # Test prediction before fitting
        with self.assertRaises(ValueError):
            cp.predict(self.x_clf[:5])

        # Test calibration before fitting
        with self.assertRaises(RuntimeError):
            cp.calibrate(self.x_clf[:10], self.y_clf[:10])

        # Test predict_proba on regressor
        reg = RandomForestRegressor(random_state=42, n_estimators=5)
        cp_reg = ConformalPredictor(reg, estimator_type="regressor")
        cp_reg.fit(self.x_reg[:50], self.y_reg[:50])

        with self.assertRaises(NotImplementedError):
            cp_reg.predict_proba(self.x_reg[:5])

        # Test predict_int on classifier
        cp.fit(self.x_clf[:50], self.y_clf[:50])
        with self.assertRaises(NotImplementedError):
            cp.predict_int(self.x_clf[:5])


if __name__ == "__main__":
    unittest.main()
