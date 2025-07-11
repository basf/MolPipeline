"""Unit tests for conformal prediction wrappers using real datasets."""

import unittest
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from molpipeline.experimental.uncertainty.conformal import (
    CrossConformalCV,
    UnifiedConformalCV,
)
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import MolToMorganFP


class TestConformalCVWithRealData(unittest.TestCase):
    """Unit tests for UnifiedConformalCV and CrossConformalCV using real datasets."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test by loading the datasets."""
        # Paths to the datasets
        logd_path = "tests/test_data/molecule_net_logd.tsv.gz"
        bbbp_path = "tests/test_data/molecule_net_bbbp.tsv.gz"

        # Load the datasets directly from the .gz files
        cls.logd_data = pd.read_csv(logd_path, compression="gzip", sep="\t", nrows=100)
        cls.bbbp_data = pd.read_csv(bbbp_path, compression="gzip", sep="\t", nrows=100)

        # Initialize the pipeline
        smi2mol = SmilesToMol()
        mol2morgan = MolToMorganFP(radius=2, n_bits=2048)
        cls.pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("morgan", mol2morgan),
            ]
        )

    def featurize_smiles(self, smiles: pd.Series, labels: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Featurize SMILES strings into Morgan fingerprints and filter corresponding labels."""
        # Validate SMILES strings
        valid_smiles = []
        valid_labels = []
        for smi, label in zip(smiles, labels):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)
                valid_labels.append(label)
            else:
                print(f"Warning: Invalid SMILES string skipped: {smi}")

        # Transform valid SMILES to fingerprints
        try:
            matrix = self.pipeline.fit_transform(valid_smiles)
            return matrix.toarray(), np.array(valid_labels)  # Convert sparse matrix to dense array
        except Exception as e:
            print(f"Error during featurization: {e}")
            raise

    def test_unified_conformal_regressor_logd(self) -> None:
        """Test UnifiedConformalCV with a regressor on the logd dataset."""
        x, y = self.featurize_smiles(self.logd_data["smiles"], self.logd_data["exp"])

        # Split into train and calibration sets
        x_train, x_calib, y_train, y_calib = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        # Initialize and test the UnifiedConformalCV regressor
        reg = RandomForestRegressor(n_estimators=5, random_state=42)
        cp = UnifiedConformalCV(reg, estimator_type="auto")
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)

        # Prediction intervals
        intervals = cp.predict_int(x_calib)

        # Assertions
        self.assertEqual(intervals.shape[0], len(y_calib))
        self.assertEqual(intervals.shape[1], 2)  # Lower and upper bounds
        self.assertTrue(np.all(intervals[:, 0] <= intervals[:, 1]))  # Valid intervals

    def test_unified_conformal_classifier_bbbp(self) -> None:
        """Test UnifiedConformalCV with a classifier on the bbbp dataset."""
        x, y = self.featurize_smiles(self.bbbp_data["smiles"], self.bbbp_data["p_np"])

        # Split into train and calibration sets
        x_train, x_calib, y_train, y_calib = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        # Initialize and test the UnifiedConformalCV classifier
        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        cp = UnifiedConformalCV(clf, estimator_type="auto")
        cp.fit(x_train, y_train)
        cp.calibrate(x_calib, y_calib)

        # Predictions
        preds = cp.predict(x_calib)
        probs = cp.predict_proba(x_calib)
        sets = cp.predict_conformal_set(x_calib)

        # Assertions
        self.assertEqual(len(preds), len(y_calib))
        self.assertEqual(probs.shape[0], len(y_calib))
        self.assertEqual(len(sets), len(y_calib))
        self.assertTrue(all(len(s) > 0 for s in sets))  # Ensure non-empty sets

    def test_cross_conformal_regressor_logd(self) -> None:
        """Test CrossConformalCV with a regressor on the logd dataset."""
        x, y = self.featurize_smiles(self.logd_data["smiles"], self.logd_data["exp"])

        # Initialize and test the CrossConformalCV regressor
        reg = RandomForestRegressor(n_estimators=5, random_state=42)
        ccp = CrossConformalCV(reg, estimator_type="auto", n_folds=3)
        ccp.fit(x, y)

        # Prediction intervals
        intervals = ccp.predict_int(x)

        # Assertions
        self.assertEqual(intervals.shape[0], len(y))
        self.assertEqual(intervals.shape[1], 2)  # Lower and upper bounds
        self.assertTrue(np.all(intervals[:, 0] <= intervals[:, 1]))  # Valid intervals

    def test_cross_conformal_classifier_bbbp(self) -> None:
        """Test CrossConformalCV with a classifier on the bbbp dataset."""
        x, y = self.featurize_smiles(self.bbbp_data["smiles"], self.bbbp_data["p_np"])

        # Initialize and test the CrossConformalCV classifier
        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        ccp = CrossConformalCV(clf, estimator_type="auto", n_folds=3)
        ccp.fit(x, y)

        # Predictions
        preds = ccp.predict(x)
        probs = ccp.predict_proba(x)
        sets = ccp.predict_conformal_set(x)

        # Assertions
        self.assertEqual(len(preds), len(y))
        self.assertEqual(probs.shape[0], len(y))
        self.assertEqual(len(sets), len(y))
        self.assertTrue(all(len(s) > 0 for s in sets))  # Ensure non-empty sets


if __name__ == "__main__":
    unittest.main()