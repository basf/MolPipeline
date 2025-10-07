"""Test for MolToPharmacophore2DFP."""

import tempfile
import unittest

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint
from rdkit.DataStructs import ExplicitBitVect, IntSparseIntVect
from scipy import sparse
from sklearn import clone

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any.mol2pharmacophore2d_fingerprint import MolToPharmacophore2DFP
from molpipeline.utils.json_operations import recursive_from_json, recursive_to_json
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
        # Create a minimal feature definition
        self.minimal_fdef = """
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

    def test_init_default_parameters(self) -> None:
        """Test initialization with default parameters."""
        fp_element = MolToPharmacophore2DFP()

        self.assertEqual(fp_element.min_point_count, 2)
        self.assertEqual(fp_element.max_point_count, 3)
        self.assertTrue(fp_element.triangular_pruning)
        self.assertEqual(fp_element.distance_bins, Gobbi_Pharm2D.defaultBins)
        self.assertFalse(fp_element.counted)  # Default should be False

    def test_init_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        custom_bins = [(0.0, 3.0), (3.0, 6.0), (6.0, 10.0)]
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

        # Test invalid distance_bins
        with self.assertRaises(ValueError):
            MolToPharmacophore2DFP(distance_bins=[(0, 1), (1, 0)])

    def test_feature_names(self) -> None:
        """Test feature names generation."""
        fp_element = MolToPharmacophore2DFP()
        feature_names = fp_element.feature_names

        self.assertEqual(len(feature_names), fp_element.n_bits)
        self.assertTrue(all(name.startswith("pharm2d_") for name in feature_names))

    def test_get_params(self) -> None:
        """Test get_params method."""
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
            len(params_deep["distance_bins"]),
            len(params_shallow["distance_bins"]),
        )

    def test_set_params(self) -> None:
        """Test set_params method."""
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
        """Test serialization with joblib.

        Raises
        ------
        AssertionError
            If the prior test does not raise an error. Primarily for type checking.

        """
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

        self.assertIsInstance(original_fp, np.ndarray)
        self.assertIsInstance(loaded_fp, np.ndarray)
        if not isinstance(original_fp, np.ndarray):
            raise AssertionError("Expected original_fp to be a numpy array.")
        if not isinstance(loaded_fp, np.ndarray):
            raise AssertionError("Expected loaded_fp to be a numpy array.")
        self.assertTrue(np.array_equal(original_fp, loaded_fp))

    def test_json_serialization(self) -> None:
        """Test JSON serialization of element.

        Raises
        ------
        AssertionError
            If the prior test does not raise an error. Primarily for type checking.

        """
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
        if not isinstance(fp, np.ndarray):
            raise AssertionError("Expected fp to be a numpy array.")
        if not isinstance(fp_loaded, np.ndarray):
            raise AssertionError("Expected fp_loaded to be a numpy array.")
        self.assertTrue(np.array_equal(fp, fp_loaded))

    def test_custom_feature_definition(self) -> None:
        """Test using a custom feature definition."""
        fp_element = MolToPharmacophore2DFP(feature_definition=self.minimal_fdef)
        fingerprints1 = fp_element.transform(self.test_molecules)
        self.assertIsInstance(fingerprints1, sparse.csr_matrix)
        self.assertTrue(fingerprints1.nnz >= 1)  # May be 0 for simple features

    def test_from_file(self) -> None:
        """Test using a feature definition from a file."""
        fp_element1 = MolToPharmacophore2DFP(feature_definition=self.minimal_fdef)
        fingerprints1 = fp_element1.transform(self.test_molecules)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".fdef",
            encoding="utf-8",
        ) as tmp_file:
            tmp_file.write(self.minimal_fdef)
            tmp_file.flush()

            # Should work with custom feature factory
            fp_element2 = MolToPharmacophore2DFP.from_file(tmp_file.name)
        fingerprints2 = fp_element2.transform(self.test_molecules)

        self.assertIsInstance(fingerprints2, sparse.csr_matrix)
        self.assertTrue(fingerprints2.nnz >= 1)  # May be 0 for simple features

        # assert that both fingerprints are equal
        self.assertTrue(
            np.array_equal(fingerprints1.toarray(), fingerprints2.toarray()),
        )

    def test_default_configuration_corresponds_to_gobbi(self) -> None:
        """Test default config corresponds to Gobbi pharmacophore fingerprint.

        Raises
        ------
        AssertionError
            If the generated fingerprint does not match the RDKit Gobbi_Pharm2D factory.

        """
        fp_element = MolToPharmacophore2DFP(
            return_as="dense",
        )
        gobbi_fps = fp_element.transform(self.test_molecules)

        # compare to RDKit's Gobbi_Pharm2D fingerprint factory
        gobbi_fp_rdkit_list = [
            Gen2DFingerprint(m, Gobbi_Pharm2D.factory) for m in self.test_molecules
        ]
        gobbi_fp_rdkit = fingerprints_to_numpy(gobbi_fp_rdkit_list)
        if not isinstance(gobbi_fps, np.ndarray):
            raise AssertionError("Expected gobbi_fps to be a numpy array.")
        self.assertTrue(np.array_equal(gobbi_fps, gobbi_fp_rdkit))

    def test_from_preconfiguration_gobbi(self) -> None:
        """Test from_preconfiguration works with 'gobbi'.

        Raises
        ------
        AssertionError
            If the generated fingerprint does not match the RDKit Gobbi_Pharm2D factory.

        """
        fp_element = MolToPharmacophore2DFP.from_preconfiguration(
            "gobbi",
            return_as="dense",
        )

        self.assertEqual(fp_element.min_point_count, 2)
        self.assertEqual(fp_element.max_point_count, 3)
        self.assertTrue(fp_element.triangular_pruning)
        self.assertEqual(fp_element.distance_bins, Gobbi_Pharm2D.defaultBins)

        fps = fp_element.transform(self.test_molecules)

        # compare to RDKit's Gobbi_Pharm2D fingerprint factory
        gobbi_fp_rdkit_list = [
            Gen2DFingerprint(m, Gobbi_Pharm2D.factory) for m in self.test_molecules
        ]
        gobbi_fp_rdkit = fingerprints_to_numpy(gobbi_fp_rdkit_list)
        if not isinstance(fps, np.ndarray):
            raise AssertionError("Expected fps to be a numpy array.")
        self.assertTrue(np.array_equal(fps, gobbi_fp_rdkit))

    def test_from_preconfiguration_base(self) -> None:
        """Test from_preconfiguration works with 'base'."""
        fp_element = MolToPharmacophore2DFP.from_preconfiguration(
            "base",
            return_as="dense",
        )

        self.assertEqual(fp_element.min_point_count, 2)
        self.assertEqual(fp_element.max_point_count, 3)
        self.assertTrue(fp_element.triangular_pruning)
        self.assertEqual(
            fp_element.distance_bins,
            [  # pylint: disable=R0801
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 100),
            ],
        )

        fps = fp_element.transform(self.test_molecules)
        self.assertIsInstance(fps, np.ndarray)
        self.assertEqual(fps.shape[0], len(self.test_molecules))

    def test_from_preconfiguration_unknown_name(self) -> None:
        """Test preconfigured fingerprint with an unknown name."""
        with self.assertRaises(ValueError):
            MolToPharmacophore2DFP.from_preconfiguration("unknown_fingerprint")  # type: ignore[arg-type]

    def test_sklearn_clone(self) -> None:
        """Test sklearn clone functionality."""
        fp_element = MolToPharmacophore2DFP(
            min_point_count=2,
            max_point_count=3,
            triangular_pruning=True,
            counted=False,
        )
        cloned_fp_element = clone(fp_element)

        self.assertEqual(fp_element.get_params(), cloned_fp_element.get_params())


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
            np.array_equal(np.unique(np.asarray(fingerprints.todense())), [0, 1]),
        )

    def test_fingerprint_generation_sparse_counted(self) -> None:
        """Test fingerprint generation with sparse output."""
        fp_element = MolToPharmacophore2DFP(return_as="sparse", counted=True)
        fingerprints = fp_element.transform(self.test_molecules)

        self.assertIsInstance(fingerprints, sparse.csr_matrix)
        self.assertEqual(fingerprints.shape[0], len(self.test_molecules))
        self.assertEqual(fingerprints.shape[1], fp_element.n_bits)
        self.assertTrue(fingerprints.nnz > 0)  # Should have some non-zero elements

    def test_fingerprint_generation_dense_binary(self) -> None:
        """Test fingerprint generation with dense output.

        Raises
        ------
        AssertionError
            If the prior does not raise an error. Primarily for type checking.

        """
        fp_element = MolToPharmacophore2DFP(return_as="dense", counted=False)
        fingerprints = fp_element.transform(self.test_molecules)

        self.assertIsInstance(fingerprints, np.ndarray)
        if not isinstance(fingerprints, np.ndarray):
            raise AssertionError("Expected fingerprints to be a numpy array.")
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

    def test_fingerprint_generation_rdkit_binary(self) -> None:
        """Test fingerprint generation with "rdkit_explicit" output.

        Raises
        ------
        AssertionError
            If the prior does not raise an error. Primarily for type checking.

        """
        fp_element = MolToPharmacophore2DFP(return_as="rdkit", counted=False)
        fingerprints = fp_element.transform(self.test_molecules)

        self.assertIsInstance(fingerprints, list)
        self.assertEqual(len(fingerprints), len(self.test_molecules))

        # Check that each element is an ExplicitBitVect
        for fp in fingerprints:
            self.assertIsInstance(fp, ExplicitBitVect)
            if not isinstance(fp, ExplicitBitVect):
                raise AssertionError("Expected fp to be an ExplicitBitVect.")
            self.assertEqual(fp.GetNumBits(), fp_element.n_bits)

    def test_fingerprint_generation_rdkit_counted(self) -> None:
        """Test fingerprint generation with ExplicitBitVect output.

        Raises
        ------
        AssertionError
            If the prior does not raise an error. Primarily for type checking.

        """
        fp_element = MolToPharmacophore2DFP(return_as="rdkit", counted=True)
        fingerprints = fp_element.transform(self.test_molecules)

        self.assertIsInstance(fingerprints, list)
        if not isinstance(fingerprints, list):
            raise AssertionError("Expected fingerprints to be a list.")
        self.assertEqual(len(fingerprints), len(self.test_molecules))

        # Check that each element is an IntSparseIntVect
        for fp in fingerprints:
            self.assertIsInstance(fp, IntSparseIntVect)
            self.assertEqual(fp.GetLength(), fp_element.n_bits)

    def test_pretransform_single_binary(self) -> None:
        """Test molecule pretransformation for all return_as variants with binary.

        Raises
        ------
        AssertionError
            If the prior test does not raise an error. Primarily for type checking.

        """
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
        self.assertIsInstance(result_rdkit, ExplicitBitVect)
        if not isinstance(result_rdkit, ExplicitBitVect):
            raise AssertionError("Expected result_rdkit to be an ExplicitBitVect.")

        # Verify consistency across formats
        # Convert all to same format for comparison
        sparse_as_dense = np.zeros(fp_element_sparse.n_bits)
        for idx in result_sparse:
            sparse_as_dense[idx] = 1
        rdkit_as_dense = fingerprints_to_numpy([result_rdkit])[0]
        if not isinstance(result_dense, np.ndarray):
            raise AssertionError("Expected result_dense to be a numpy array.")
        self.assertTrue(np.array_equal(result_dense, sparse_as_dense))
        self.assertTrue(np.array_equal(result_dense, rdkit_as_dense))

    def test_pretransform_single_counted(self) -> None:
        """Test molecule pretransformation for all return_as variants with counted.

        Raises
        ------
        AssertionError
            If the results are not consistent across formats.

        """
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
        self.assertIsInstance(result_rdkit, IntSparseIntVect)
        if not isinstance(result_rdkit, IntSparseIntVect):
            raise AssertionError("Expected result_rdkit to be an IntSparseIntVect.")

        # Verify consistency across formats
        # Convert all to same format for comparison
        sparse_as_dense = np.zeros(fp_element_sparse.n_bits)
        if not isinstance(result_sparse, dict):
            raise AssertionError("Expected result_sparse to be a dict.")
        for idx, count in result_sparse.items():
            sparse_as_dense[idx] = count
        explicit_as_dense = fingerprints_to_numpy([result_rdkit])[0]
        if not isinstance(result_dense, np.ndarray):
            raise AssertionError("Expected result_dense to be a numpy array.")
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
