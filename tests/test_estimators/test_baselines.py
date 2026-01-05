"""Test baseline estimators."""

import unittest

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from molpipeline.estimators.baselines import (
    get_rf_classifier_baseline,
    get_rf_regressor_baseline,
)

SEED = 42


class TestBaselines(unittest.TestCase):
    """Test baseline estimators."""

    def setUp(self) -> None:
        """Set up any necessary components before each test."""
        self.test_smiles = ["CCO", "CCN", "CCC", "CCCl", "CCBr"]
        self.test_y = [0, 1, 0, 1, 0]

    def test_get_rf_classifier_baseline(self) -> None:
        """Test get_rf_classifier_baseline function.

        Raises
        ------
        AssertionError
            If the model is not a RandomForestClassifier.

        """
        pipeline = get_rf_classifier_baseline(n_jobs=1, random_state=SEED)
        self.assertEqual(len(pipeline.steps), 4)
        self.assertEqual(pipeline.steps[-1][0], "model")
        rf = pipeline.steps[-1][1]
        if not isinstance(rf, RandomForestClassifier):
            raise AssertionError("Model is not a RandomForestClassifier")
        self.assertEqual(rf.n_estimators, 500)
        self.assertEqual(rf.random_state, 42)
        self.assertEqual(rf.n_jobs, 1)
        self.assertEqual(rf.max_features, "log2")

        # test fit and predict
        pipeline.fit(self.test_smiles, self.test_y)
        predictions = pipeline.predict(self.test_smiles)
        proba = pipeline.predict_proba(self.test_smiles)
        self.assertEqual(len(predictions), len(self.test_smiles))
        self.assertEqual(proba.shape, (len(self.test_smiles), 2))

    def test_get_rf_regressor_baseline(self) -> None:
        """Test get_rf_regressor_baseline function.

        Raises
        ------
        AssertionError
            If the model is not a RandomForestRegressor.

        """
        pipeline = get_rf_regressor_baseline(n_jobs=1, random_state=SEED)
        self.assertEqual(len(pipeline.steps), 4)
        self.assertEqual(pipeline.steps[-1][0], "model")
        rf = pipeline.steps[-1][1]
        if not isinstance(rf, RandomForestRegressor):
            raise AssertionError("Model is not a RandomForestRegressor")
        self.assertEqual(rf.n_estimators, 500)
        self.assertEqual(rf.random_state, 42)
        self.assertEqual(rf.n_jobs, 1)
        self.assertEqual(rf.max_features, "log2")

        # test fit and predict
        pipeline.fit(self.test_smiles, self.test_y)
        predictions = pipeline.predict(self.test_smiles)
        self.assertEqual(len(predictions), len(self.test_smiles))
