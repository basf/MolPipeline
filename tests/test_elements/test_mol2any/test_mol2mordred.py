"""Test generation of Mordred descriptors."""

import unittest

import numpy as np
from rdkit.Chem import MolFromSmiles
from sklearn import clone

from molpipeline import Pipeline
from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any.mol2mordred import (
    DEFAULT_DESCRIPTORS,
    MolToMordred,
)


class TestMolToMordred(unittest.TestCase):
    """Unittest for MolToMordred, which calculates Mordred descriptors."""

    def test_default_construction(self) -> None:
        """Test default construction uses all Mordred descriptors."""
        mol2mordred = MolToMordred(standardizer=None)
        self.assertEqual(mol2mordred.descriptor_list, DEFAULT_DESCRIPTORS)
        self.assertEqual(mol2mordred.n_features, len(DEFAULT_DESCRIPTORS))

    def test_custom_descriptor_list(self) -> None:
        """Test construction with a custom descriptor list."""
        descriptors = ["MW", "nAtom", "nHeavyAtom"]
        mol2mordred = MolToMordred(descriptor_list=descriptors, standardizer=None)
        self.assertEqual(mol2mordred.descriptor_list, descriptors)
        self.assertEqual(mol2mordred.n_features, 3)

    def test_invalid_descriptor_list(self) -> None:
        """Test that invalid descriptor names raise ValueError."""
        with self.assertRaises(ValueError):
            MolToMordred(descriptor_list=["NOT_A_REAL_DESCRIPTOR"])
        with self.assertRaises(ValueError):
            MolToMordred(descriptor_list=[])

    def test_pretransform_single(self) -> None:
        """Test descriptor calculation for a single molecule."""
        descriptors = ["MW", "nAtom", "nHeavyAtom"]
        mol2mordred = MolToMordred(descriptor_list=descriptors, standardizer=None)
        mol = MolFromSmiles("CCO")  # ethanol
        result = mol2mordred.pretransform_single(mol)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 3)
        # MW of ethanol ~46.04, 9 atoms (with H), 3 heavy atoms
        self.assertTrue(np.isclose(result[0], 46.042))
        self.assertEqual(result[1], 9)
        self.assertEqual(result[2], 3)

    def test_pipeline_transform(self) -> None:
        """Test Mordred descriptors in a pipeline context."""
        descriptors = ["MW", "nHeavyAtom"]
        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                (
                    "mordred",
                    MolToMordred(descriptor_list=descriptors, standardizer=None),
                ),
            ],
        )
        smiles = ["CCO", "c1ccccc1"]
        result = pipeline.fit_transform(smiles)
        self.assertEqual(result.shape, (2, 2))
        # ethanol: MW ~46.07, 3 heavy atoms
        self.assertAlmostEqual(result[0, 1], 3)
        # benzene: 6 heavy atoms
        self.assertAlmostEqual(result[1, 1], 6)

    def test_get_set_params(self) -> None:
        """Test get_params and set_params round-trip.

        Raises
        ------
        AssertionError
            If pretransform_single does not return a np.ndarray.

        """
        descriptors = ["MW", "nAtom"]
        mol2mordred = MolToMordred(descriptor_list=descriptors, standardizer=None)
        params = mol2mordred.get_params(deep=True)
        self.assertEqual(params["descriptor_list"], descriptors)

        # Deep copy: modifying returned params should not affect original
        params["descriptor_list"].append("nHeavyAtom")
        self.assertEqual(mol2mordred.descriptor_list, descriptors)
        mol2mordred2 = MolToMordred(**params)

        mol = MolFromSmiles("CCO")
        result = mol2mordred.pretransform_single(mol)
        result2 = mol2mordred2.pretransform_single(mol)
        if not isinstance(result, np.ndarray):
            raise AssertionError(f"Expected np.ndarray, got {type(result)}")
        if not isinstance(result2, np.ndarray):
            raise AssertionError(f"Expected np.ndarray, got {type(result2)}")
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result2), 3)
        self.assertTrue(np.array_equal(result, result2[:2]))

    def test_descriptor_list_getter_returns_copy(self) -> None:
        """Test that the getter returns a copy, not the internal list."""
        descriptors = ["MW", "nAtom"]
        mol2mordred = MolToMordred(descriptor_list=descriptors, standardizer=None)
        returned = mol2mordred.descriptor_list
        returned.append("nHeavyAtom")
        self.assertEqual(mol2mordred.descriptor_list, descriptors)

    def test_sklearn_clone(self) -> None:
        """Test that the element can be cloned with sklearn's clone.

        Raises
        ------
        AssertionError
            If sklearn's clone does not produce a MolToMordred instance.
        AssertionError
            If pretransform_single does not return a np.ndarray.

        """
        descriptors = ["MW", "nAtom", "nHeavyAtom"]
        original = MolToMordred(descriptor_list=descriptors, standardizer=None)
        cloned = clone(original)
        self.assertIsNot(original, cloned)
        if not isinstance(cloned, MolToMordred):
            raise AssertionError(f"Expected MolToMordred, got {type(cloned)}")
        self.assertEqual(original.descriptor_list, cloned.descriptor_list)

        mol = MolFromSmiles("CCO")
        result_original = original.pretransform_single(mol)
        result_cloned = cloned.pretransform_single(mol)
        if not isinstance(result_original, np.ndarray):
            raise AssertionError(f"Expected np.ndarray, got {type(result_original)}")
        if not isinstance(result_cloned, np.ndarray):
            raise AssertionError(f"Expected np.ndarray, got {type(result_cloned)}")
        self.assertTrue(np.array_equal(result_original, result_cloned))

    def test_return_with_errors(self) -> None:
        """Test that return_with_errors controls NaN handling.

        Raises
        ------
        AssertionError
            If pretransform_single does not return a np.ndarray in lenient mode.

        """
        # There are a range of descriptors usually failing in the DEFAULT_DESCRIPTORS
        descriptors = DEFAULT_DESCRIPTORS
        mol = MolFromSmiles("CCO")  # single hydrogen

        mol2mordred_strict = MolToMordred(
            descriptor_list=descriptors,
            return_with_errors=False,
            standardizer=None,
            log_exceptions=False,
        )
        result_strict = mol2mordred_strict.pretransform_single(mol)
        self.assertIsInstance(result_strict, InvalidInstance)

        mol2mordred_lenient = MolToMordred(
            descriptor_list=descriptors,
            return_with_errors=True,
            standardizer=None,
            log_exceptions=False,
        )
        result_lenient = mol2mordred_lenient.pretransform_single(mol)
        if not isinstance(result_lenient, np.ndarray):
            # explicit instance checking and assertion for type checker
            raise AssertionError("Expected result to be a numpy array.")
        self.assertIsInstance(result_lenient, np.ndarray)
        self.assertEqual(result_lenient.shape, (len(descriptors),))
        # should have some NaNs for failed descriptors
        self.assertGreater(np.isnan(result_lenient).sum(), 0)


if __name__ == "__main__":
    unittest.main()
