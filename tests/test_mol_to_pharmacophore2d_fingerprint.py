"""Test for MolToPharmacophore2DFP."""

import tempfile
import unittest

import joblib
import numpy as np
from rdkit import Chem
from rdkit.DataStructs import ExplicitBitVect, IntSparseIntVect
from scipy import sparse

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any.mol2pharmacophor2d_fingerprint import MolToPharmacophore2DFP
from molpipeline.utils.json_operations import recursive_to_json, recursive_from_json
from tests.utils.fingerprints import fingerprints_to_numpy


class TestMolToPharmacophore2DFP(unittest.TestCase):
    """Test MolToPharmacophore2DFP."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.test_molecules = [
            Chem.MolFromSmiles("CCO"),  # ethanol
            Chem.MolFromSmiles("CC(=O)O"),  # acetic acid
            Chem.MolFromSmiles("c1ccccc1"),  # benzene
            Chem.MolFromSmiles("CCN"),  # ethylamine
        ]

    def test_init_default_parameters(self) -> None:
        """Test initialization with default parameters."""
        fp_element = MolToPharmacophore2DFP()

        self.assertEqual(fp_element.min_point_count, 2)
        self.assertEqual(fp_element.max_point_count, 3)
        self.assertTrue(fp_element.triangular_pruning)
        self.assertEqual(fp_element.distance_bins, [(1, 2), (2, 5), (5, 8)])
        self.assertFalse(fp_element.counted)  # Default should be False

    def test_init_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        custom_bins = [(0, 3), (3, 6), (6, 10)]
        fp_element = MolToPharmacophore2DFP(
            min_point_count=3,
            max_point_count=4,
            triangular_pruning=False,
            distance_bins=custom_bins,
            counted=True,
            return_as="dense",
        )

        self.assertEqual(fp_element.min_point_count, 3)
        self.assertEqual(fp_element.max_point_count, 4)
        self.assertFalse(fp_element.triangular_pruning)
        self.assertEqual(fp_element.distance_bins, custom_bins)
        self.assertTrue(fp_element.counted)

    def test_init_validation_errors(self) -> None:
        """Test validation errors during initialization."""
        # Test min_point_count < 2
        with self.assertRaises(ValueError):
            MolToPharmacophore2DFP(min_point_count=1)

        # Test max_point_count < min_point_count
        with self.assertRaises(ValueError):
            MolToPharmacophore2DFP(min_point_count=3, max_point_count=2)

        # Test invalid feature factory path
        with self.assertRaises(ValueError):
            MolToPharmacophore2DFP(feature_definition="bad definition")

    def test_feature_names(self) -> None:
        """Test feature names generation."""
        fp_element = MolToPharmacophore2DFP()
        feature_names = fp_element.feature_names

        self.assertEqual(len(feature_names), fp_element.n_bits)
        self.assertTrue(all(name.startswith("pharm2d_") for name in feature_names))

    def test_get_params(self) -> None:
        """Test parameter retrieval."""
        fp_element = MolToPharmacophore2DFP()
        params = fp_element.get_params()

        expected_keys = {
            "feature_definition",
            "min_point_count",
            "max_point_count",
            "triangular_pruning",
            "distance_bins",
            "return_as",
            "name",
            "n_jobs",
            "uuid",
            "counted",
        }
        self.assertTrue(expected_keys.issubset(set(params.keys())))

        # Test deep copy
        params_deep = fp_element.get_params(deep=True)
        params_shallow = fp_element.get_params(deep=False)

        # Modify the shallow copy and ensure deep copy is unaffected
        params_shallow["distance_bins"].append((10, 15))
        self.assertNotEqual(
            len(params_deep["distance_bins"]), len(params_shallow["distance_bins"])
        )

    def test_set_params(self) -> None:
        """Test parameter setting."""
        fp_element = MolToPharmacophore2DFP()
        new_bins = [(0, 4), (4, 8)]
        new_element = fp_element.set_params(
            min_point_count=3,
            max_point_count=4,
            triangular_pruning=False,
            distance_bins=new_bins,
            counted=True,
        )

        self.assertIs(new_element, fp_element)  # Should return self
        self.assertEqual(fp_element.min_point_count, 3)
        self.assertEqual(fp_element.max_point_count, 4)
        self.assertFalse(fp_element.triangular_pruning)
        self.assertEqual(fp_element.distance_bins, new_bins)
        self.assertEqual(fp_element.counted, True)

    def test_set_params_validation(self) -> None:
        """Test parameter setting validation."""
        # Test invalid min_point_count
        with self.assertRaises(ValueError):
            MolToPharmacophore2DFP().set_params(min_point_count=1)

        # Test invalid max_point_count
        with self.assertRaises(ValueError):
            MolToPharmacophore2DFP().set_params(min_point_count=4, max_point_count=2)

        # Test invalid feature factory
        with self.assertRaises(ValueError):
            MolToPharmacophore2DFP().set_params(feature_definition="bad definition")

    def test_joblib_serialization(self) -> None:
        """Test serialization with joblib."""
        fp_element = MolToPharmacophore2DFP(
            counted=True,
            return_as="dense",
            triangular_pruning=False,
            distance_bins=[(0, 3), (3, 7)],
        )

        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_file:
            joblib.dump(fp_element, tmp_file.name)
            loaded_element = joblib.load(tmp_file.name)

        # Test that parameters are preserved
        self.assertEqual(fp_element.get_params(), loaded_element.get_params())

        # Test that functionality is preserved
        original_fp = fp_element.transform(self.test_molecules)
        loaded_fp = loaded_element.transform(self.test_molecules)

        self.assertTrue(np.array_equal(original_fp, loaded_fp))

    def test_json_serialization(self) -> None:
        """Test JSON serialization of element."""
        fp_element = MolToPharmacophore2DFP(
            min_point_count=3,
            max_point_count=3,
            triangular_pruning=False,
            distance_bins=[(0, 3), (3, 7)],
            counted=True,
            return_as="dense",
        )

        json_dict = recursive_to_json(fp_element)
        fp_element_loaded = recursive_from_json(json_dict)

        # test get_params returns the same parameters
        self.assertEqual(fp_element.get_params(), fp_element_loaded.get_params())

        # test generated fingerprints are equal
        fp = fp_element.transform(self.test_molecules)
        fp_loaded = fp_element_loaded.transform(self.test_molecules)
        self.assertIsInstance(fp, np.ndarray)
        self.assertIsInstance(fp_loaded, np.ndarray)
        self.assertTrue(np.array_equal(fp, fp_loaded))

    def test_custom_feature_definition_with_valid_file(self) -> None:
        """Test using a custom feature factory file."""
        # Create a temporary minimal feature definition file
        minimal_fdef = """
AtomType Donor [N,O;H1,H2]
DefineFeature TestDonor [{Donor}]
  Family Donor
  Weights 1
EndFeature

AtomType Acceptor [N,O]
DefineFeature TestAcceptor [{Acceptor}]
  Family Acceptor
  Weights 1
EndFeature
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fdef", delete=False
        ) as tmp_file:
            tmp_file.write(minimal_fdef)
            tmp_file.flush()

            # Should work with custom feature factory
            fp_element = MolToPharmacophore2DFP(feature_definition=tmp_file.name)
            fingerprint = fp_element.transform(self.test_molecules)

            self.assertIsInstance(fingerprint, sparse.csr_matrix)
            self.assertTrue(fingerprint.nnz >= 1)  # May be 0 for simple features


class TestMolToPharmacophore2DFPFingerprintCalculation(unittest.TestCase):
    """Test fingerprint calculation with MolToPharmacophore2DFP."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.test_molecules = [
            Chem.MolFromSmiles("CCO"),  # ethanol
            Chem.MolFromSmiles("CC(=O)O"),  # acetic acid
            Chem.MolFromSmiles("c1ccccc1"),  # benzene
            Chem.MolFromSmiles("CCN"),  # ethylamine
        ]

    def test_fingerprint_generation_sparse_binary(self) -> None:
        """Test fingerprint generation with sparse output."""
        fp_element = MolToPharmacophore2DFP(return_as="sparse", counted=False)
        fingerprints = fp_element.transform(self.test_molecules)

        self.assertIsInstance(fingerprints, sparse.csr_matrix)
        self.assertEqual(fingerprints.shape[0], len(self.test_molecules))
        self.assertEqual(fingerprints.shape[1], fp_element.n_bits)
        self.assertTrue(
            np.array_equal(np.unique(np.asarray(fingerprints.todense())), [0, 1])
        )

    def test_fingerprint_generation_sparse_counted(self) -> None:
        """Test fingerprint generation with sparse output."""
        fp_element = MolToPharmacophore2DFP(return_as="sparse", counted=True)
        fingerprints = fp_element.transform(self.test_molecules)

        self.assertIsInstance(fingerprints, sparse.csr_matrix)
        self.assertEqual(fingerprints.shape[0], len(self.test_molecules))
        self.assertEqual(fingerprints.shape[1], fp_element.n_bits)
        self.assertTrue(fingerprints.nnz > 0)  # Should have some non-zero elements
        self.assertTrue(max(fingerprints.data) > 1)  # Counted should have counts > 1

    def test_fingerprint_generation_dense_binary(self) -> None:
        """Test fingerprint generation with dense output."""
        fp_element = MolToPharmacophore2DFP(return_as="dense", counted=False)
        fingerprints = fp_element.transform(self.test_molecules)

        self.assertIsInstance(fingerprints, np.ndarray)
        self.assertEqual(fingerprints.shape[0], len(self.test_molecules))
        self.assertEqual(fingerprints.shape[1], fp_element.n_bits)
        self.assertTrue(np.array_equal(np.unique(fingerprints), [0, 1]))

    def test_fingerprint_generation_counted(self) -> None:
        """Test fingerprint generation with dense output."""
        fp_element = MolToPharmacophore2DFP(return_as="dense", counted=True)
        fingerprints = fp_element.transform(self.test_molecules)

        self.assertIsInstance(fingerprints, np.ndarray)
        self.assertEqual(fingerprints.shape[0], len(self.test_molecules))
        self.assertEqual(fingerprints.shape[1], fp_element.n_bits)
        self.assertGreater(np.max(fingerprints), 1)  # Should have counts >= 1

    def test_fingerprint_generation_rdkit_binary(self) -> None:
        """Test fingerprint generation with "rdkit_explicit" output."""
        fp_element = MolToPharmacophore2DFP(return_as="rdkit", counted=False)
        fingerprints = fp_element.transform(self.test_molecules)

        self.assertIsInstance(fingerprints, list)
        self.assertEqual(len(fingerprints), len(self.test_molecules))

        # Check that each element is an ExplicitBitVect
        for fp in fingerprints:
            self.assertTrue(isinstance(fp, ExplicitBitVect))
            self.assertEqual(fp.GetNumBits(), fp_element.n_bits)

    def test_fingerprint_generation_rdkit_counted(self) -> None:
        """Test fingerprint generation with ExplicitBitVect output."""
        fp_element = MolToPharmacophore2DFP(return_as="rdkit", counted=True)
        fingerprints = fp_element.transform(self.test_molecules)

        self.assertIsInstance(fingerprints, list)
        self.assertEqual(len(fingerprints), len(self.test_molecules))

        # Check that each element is an IntSparseIntVect
        for fp in fingerprints:
            self.assertTrue(isinstance(fp, IntSparseIntVect))
            self.assertEqual(fp.GetLength(), fp_element.n_bits)

    def test_pretransform_single_binary(self) -> None:
        """Test molecule pretransformation for all return_as variants with binary."""
        mol = self.test_molecules[0]

        # Test sparse format
        fp_element_sparse = MolToPharmacophore2DFP(return_as="sparse", counted=False)
        result_sparse = fp_element_sparse.pretransform_single(mol)

        # Test dense format
        fp_element_dense = MolToPharmacophore2DFP(return_as="dense", counted=False)
        result_dense = fp_element_dense.pretransform_single(mol)

        # Test rdkit format
        fp_element_rdkit = MolToPharmacophore2DFP(return_as="rdkit", counted=False)
        result_rdkit = fp_element_rdkit.pretransform_single(mol)

        # Verify consistency across formats
        # Convert all to same format for comparison
        sparse_as_dense = np.zeros(fp_element_sparse.n_bits)
        for idx in result_sparse:
            sparse_as_dense[idx] = 1
        explicit_as_dense = fingerprints_to_numpy([result_rdkit])[0]
        self.assertTrue(np.array_equal(result_dense, sparse_as_dense))
        self.assertTrue(np.array_equal(result_dense, explicit_as_dense))

    def test_pretransform_single_counted(self) -> None:
        """Test molecule pretransformation for all return_as variants with counted."""
        mol = self.test_molecules[0]

        # Test sparse format (default)
        fp_element_sparse = MolToPharmacophore2DFP(return_as="sparse", counted=True)
        result_sparse = fp_element_sparse.pretransform_single(mol)

        # Test dense format
        fp_element_dense = MolToPharmacophore2DFP(return_as="dense", counted=True)
        result_dense = fp_element_dense.pretransform_single(mol)

        # Test explicit_bit_vect format
        fp_element_rdkit = MolToPharmacophore2DFP(return_as="rdkit", counted=True)
        result_rdkit = fp_element_rdkit.pretransform_single(mol)

        # Verify consistency across formats
        # Convert all to same format for comparison
        sparse_as_dense = np.zeros(fp_element_sparse.n_bits)
        for idx, count in result_sparse.items():
            sparse_as_dense[idx] = count
        explicit_as_dense = fingerprints_to_numpy([result_rdkit])[0]
        self.assertTrue(np.array_equal(result_dense, sparse_as_dense))
        self.assertTrue(np.array_equal(result_dense, explicit_as_dense))


class TestMolToPharmacophore2DFPFingerprintCalculationPipeline(unittest.TestCase):
    """Test fingerprint calculation with MolToPharmacophore2DFP in a pipeline."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.test_smiles = [
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "c1ccccc1",  # benzene
            "CCN",  # ethylamine
        ]

    def test_pipeline_fingerprint_generation(self) -> None:
        """Test fingerprint generation in a pipeline."""
        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                ("mol_fp", MolToPharmacophore2DFP()),
            ],
        )

        fingerprints = pipeline.transform(self.test_smiles)

        self.assertIsInstance(fingerprints, sparse.csr_matrix)
        self.assertEqual(fingerprints.shape[0], len(self.test_smiles))
        self.assertEqual(fingerprints.shape[1], pipeline.named_steps["mol_fp"].n_bits)
        self.assertGreater(fingerprints.nnz, 0)


if __name__ == "__main__":
    unittest.main()
