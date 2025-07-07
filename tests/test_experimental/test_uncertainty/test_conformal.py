"""Unit tests for conformal prediction wrappers."""

import unittest

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from molpipeline.experimental.uncertainty.conformal import (
    CrossConformalCV,
    UnifiedConformalCV,
)


class TestConformalCV(unittest.TestCase):
    """Unit tests for UnifiedConformalCV and CrossConformalCV wrappers."""

    def test_unified_conformal_classifier(self) -> None:
        """Test UnifiedConformalCV with a classifier."""
        x, y = make_classification(n_samples=100, n_features=10, random_state=42)
        x_train, x_calib, y_train, y_calib = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
        )
        clf = RandomForestClassifier(random_state=42)
        cp = UnifiedConformalCV(clf, estimator_type="classifier")
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)
        preds = cp.predict(x_calib)
        probs = cp.predict_proba(x_calib)
        sets = cp.predict_conformal_set(x_calib)
        self.assertEqual(len(preds), len(y_calib))
        self.assertEqual(probs.shape[0], len(y_calib))
        self.assertEqual(len(sets), len(y_calib))

    def test_unified_conformal_regressor(self) -> None:
        """Test UnifiedConformalCV with a regressor."""
        x, y, _ = make_regression(
            n_samples=100, n_features=10, random_state=42, coef=True,
        )
        x_train, x_calib, y_train, y_calib = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
        )
        reg = RandomForestRegressor(random_state=42)
        cp = UnifiedConformalCV(reg, estimator_type="regressor")
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)
        intervals = cp.predict_int(x_calib)
        self.assertEqual(intervals.shape[0], len(y_calib))
        self.assertEqual(intervals.shape[1], 2)

    def test_cross_conformal_classifier(self) -> None:
        """Test CrossConformalCV with a classifier."""
        x, y = make_classification(n_samples=100, n_features=10, random_state=42)
        clf = RandomForestClassifier(random_state=42)
        ccp = CrossConformalCV(clf, estimator_type="classifier", n_folds=3)
        ccp.fit(x, y)
        preds = ccp.predict(x)
        probs = ccp.predict_proba(x)
        sets = ccp.predict_conformal_set(x)
        self.assertEqual(len(preds), len(y))
        self.assertEqual(probs.shape[0], len(y))
        self.assertEqual(len(sets), len(y))

    def test_cross_conformal_regressor(self) -> None:
        """Test CrossConformalCV with a regressor."""
        x, y, _ = make_regression(
            n_samples=100, n_features=10, random_state=42, coef=True,
        )
        reg = RandomForestRegressor(random_state=42)
        ccp = CrossConformalCV(reg, estimator_type="regressor", n_folds=3)
        ccp.fit(x, y)
        # Each model should produce intervals for all samples
        for model in ccp.models_:
            intervals = model.predict_int(x)
            self.assertEqual(intervals.shape[0], len(y))
            self.assertEqual(intervals.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
