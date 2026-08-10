"""Test MolToFeaturesFromFile pipeline element."""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem

from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.mol2any.mol2data_from_csv import MolToDataFromCSV


class TestMolToFeaturesFromCSV(unittest.TestCase):
    """Test the MolToFeaturesFromFile pipeline element."""

    def setUp(self):
        """Set up test data and molecules."""
        # Create test molecules with known identifiers
        self.mols = [
            Chem.MolFromSmiles("CCO"),  # ethanol
            Chem.MolFromSmiles("CC(=O)O"),  # acetic acid
            Chem.MolFromSmiles("c1ccccc1"),  # benzene
            Chem.MolFromSmiles("CCCCCC"),  # hexane - not in test data
        ]

        # Create CSV files with test features
        self.temp_dir = tempfile.TemporaryDirectory()

        # SMILES data
        self.smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
        self.features_df = pd.DataFrame(
            {
                "smiles": self.smiles_list,
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
                "feature3": [7.0, 8.0, 9.0],
            }
        )
        self.feature_file_path = Path(self.temp_dir.name) / "features.csv"
        self.features_df.to_csv(self.feature_file_path, index=False)

        # InChI data
        self.inchis = [Chem.MolToInchi(mol) for mol in self.mols[:3]]
        self.features_df_inchi = pd.DataFrame(
            {
                "inchi": self.inchis,
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        self.inchi_file_path = os.path.join(self.temp_dir.name, "features_inchi.csv")
        self.features_df_inchi.to_csv(self.inchi_file_path, index=False)

        # InChIKey data
        self.inchikeys = [Chem.MolToInchiKey(mol) for mol in self.mols[:3]]
        self.features_df_inchikey = pd.DataFrame(
            {
                "inchikey": self.inchikeys,
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        self.inchikey_file_path = os.path.join(
            self.temp_dir.name, "features_inchikey.csv"
        )
        self.features_df_inchikey.to_csv(self.inchikey_file_path, index=False)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_basic_functionality(self):
        """Test that features are correctly returned for molecules."""
        mol2feat = MolToDataFromCSV(
            feature_file_path=self.feature_file_path,
            identifier_column="smiles",
            feature_columns=["feature1", "feature2", "feature3"],
        )

        results = mol2feat.transform(self.mols[:3])

        self.assertEqual(len(results), 3)
        np.testing.assert_array_equal(results[0], np.array([1.0, 4.0, 7.0]))
        np.testing.assert_array_equal(results[1], np.array([2.0, 5.0, 8.0]))
        np.testing.assert_array_equal(results[2], np.array([3.0, 6.0, 9.0]))

    def test_missing_molecule_invalid_instance(self):
        """Test handling of missing molecules with invalid_instance strategy."""
        mol2feat = MolToDataFromCSV(
            feature_file_path=self.feature_file_path,
            identifier_column="smiles",
            feature_columns=["feature1", "feature2", "feature3"],
            missing_value_strategy="invalid_instance",
        )

        results = mol2feat.transform(self.mols)  # Include hexane (not in data)

        self.assertEqual(len(results), 4)
        self.assertIsInstance(results[3], InvalidInstance)

    def test_missing_molecule_nan(self):
        """Test handling of missing molecules with nan strategy."""
        mol2feat = MolToDataFromCSV(
            feature_file_path=self.feature_file_path,
            identifier_column="smiles",
            feature_columns=["feature1", "feature2", "feature3"],
            missing_value_strategy="nan",
        )

        results = mol2feat.transform(self.mols)  # Include hexane (not in data)

        self.assertEqual(len(results), 4)
        self.assertTrue(np.isnan(results[3]).all())

    def test_inchi_identifier(self):
        """Test using InChI as the identifier."""
        mol2feat = MolToDataFromCSV(
            feature_file_path=self.inchi_file_path,
            identifier_column="inchi",
            feature_columns=["feature1", "feature2"],
            id_type="inchi",
        )

        results = mol2feat.transform(self.mols[:3])

        self.assertEqual(len(results), 3)
        np.testing.assert_array_equal(results[0], np.array([1.0, 4.0]))

    def test_inchikey_identifier(self):
        """Test using InChIKey as the identifier."""
        mol2feat = MolToDataFromCSV(
            feature_file_path=self.inchikey_file_path,
            identifier_column="inchikey",
            feature_columns=["feature1", "feature2"],
            id_type="inchikey",
        )

        results = mol2feat.transform(self.mols[:3])

        self.assertEqual(len(results), 3)
        np.testing.assert_array_equal(results[0], np.array([1.0, 4.0]))

    def test_empty_feature_columns(self):
        """Test that an empty feature_columns list raises ValueError."""
        with self.assertRaises(ValueError) as context:
            MolToDataFromCSV(
                feature_file_path=self.feature_file_path,
                identifier_column="smiles",
                feature_columns=[],
            )
        self.assertTrue(
            str(context.exception).startswith("Empty feature_columns is not allowed")
        )

    def test_nonexistent_file(self):
        """Test that a nonexistent feature file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            MolToDataFromCSV(
                feature_file_path="nonexistent_file.csv",
                identifier_column="smiles",
                feature_columns=["feature1"],
            )

    def test_missing_columns(self):
        """Test that missing columns in the feature file raise ValueError."""
        with self.assertRaises(ValueError) as context:
            MolToDataFromCSV(
                feature_file_path=self.feature_file_path,
                identifier_column="smiles",
                feature_columns=["feature1", "nonexistent_feature"],
            )
        self.assertTrue(str(context.exception).startswith("Error reading feature file"))

    def test_get_params(self):
        """Test that get_params returns the correct parameters."""
        mol2feat = MolToDataFromCSV(
            feature_file_path=self.feature_file_path,
            identifier_column="smiles",
            feature_columns=["feature1", "feature2"],
            id_type="smiles",
            missing_value_strategy="invalid_instance",
            name="TestElement",
            n_jobs=2,
        )

        params = mol2feat.get_params()

        self.assertEqual(params["feature_file_path"], Path(self.feature_file_path))
        self.assertEqual(params["identifier_column"], "smiles")
        self.assertEqual(params["feature_columns"], ["feature1", "feature2"])
        self.assertEqual(params["id_type"], "smiles")
        self.assertEqual(params["missing_value_strategy"], "invalid_instance")
        self.assertEqual(params["name"], "TestElement")
        self.assertEqual(params["n_jobs"], 2)

    def test_set_params(self):
        """Test that set_params correctly sets parameters."""
        mol2feat = MolToDataFromCSV(
            feature_file_path=self.feature_file_path,
            identifier_column="smiles",
            feature_columns=["feature1"],
            name="OriginalName",
        )

        mol2feat.set_params(name="NewName", n_jobs=4)

        params = mol2feat.get_params()
        self.assertEqual(params["name"], "NewName")
        self.assertEqual(params["n_jobs"], 4)


if __name__ == "__main__":
    unittest.main()
