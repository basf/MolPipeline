import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from molpipeline.experimental.uncertainty.conformal import UnifiedConformalCV, CrossConformalCV

class TestConformalCV(unittest.TestCase):
    def test_unified_conformal_classifier(self):
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(random_state=42)
        cp = UnifiedConformalCV(clf, estimator_type="classifier")
        cp.fit(X_train, y_train)
        cp.calibrate(X_calib, y_calib)
        preds = cp.predict(X_calib)
        probs = cp.predict_proba(X_calib)
        sets = cp.predict_conformal_set(X_calib)
        self.assertEqual(len(preds), len(y_calib))
        self.assertEqual(probs.shape[0], len(y_calib))
        self.assertEqual(len(sets), len(y_calib))

    def test_unified_conformal_regressor(self):
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.2, random_state=42)
        reg = RandomForestRegressor(random_state=42)
        cp = UnifiedConformalCV(reg, estimator_type="regressor")
        cp.fit(X_train, y_train)
        cp.calibrate(X_calib, y_calib)
        intervals = cp.predict_int(X_calib)
        self.assertEqual(intervals.shape[0], len(y_calib))
        self.assertEqual(intervals.shape[1], 2)

    def test_cross_conformal_classifier(self):
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        clf = RandomForestClassifier(random_state=42)
        ccp = CrossConformalCV(clf, estimator_type="classifier", n_folds=3)
        ccp.fit(X, y)
        preds = ccp.predict(X)
        probs = ccp.predict_proba(X)
        sets = ccp.predict_conformal_set(X)
        self.assertEqual(len(preds), len(y))
        self.assertEqual(probs.shape[0], len(y))
        self.assertEqual(len(sets), len(y))

    def test_cross_conformal_regressor(self):
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        reg = RandomForestRegressor(random_state=42)
        ccp = CrossConformalCV(reg, estimator_type="regressor", n_folds=3)
        ccp.fit(X, y)
        # Each model should produce intervals for all samples
        for model in ccp.models_:
            intervals = model.predict_int(X)
            self.assertEqual(intervals.shape[0], len(y))
            self.assertEqual(intervals.shape[1], 2)

if __name__ == "__main__":
    unittest.main()