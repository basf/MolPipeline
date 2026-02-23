"""Tests for the MolToElementCount pipeline element."""

import unittest

import numpy as np
from rdkit.Chem import MolFromSmiles

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import MolToElementCount


class TestMolToElementCount(unittest.TestCase):
    """Unittest for MolToElementCount."""

    def test_default_construction(self) -> None:
        """Test default construction with all 118 elements."""
        element_count = MolToElementCount(standardizer=None)
        self.assertEqual(element_count.n_features, 118)
        self.assertEqual(len(element_count.feature_names), 118)
        self.assertEqual(element_count.feature_names[0], "Count_H")
        self.assertEqual(element_count.feature_names[5], "Count_C")

    def test_custom_element_list(self) -> None:
        """Test construction with a custom element list."""
        element_count = MolToElementCount(element_list=[1, 6, 7, 8], standardizer=None)
        self.assertEqual(element_count.n_features, 4)
        self.assertEqual(
            element_count.feature_names,
            ["Count_H", "Count_C", "Count_N", "Count_O"],
        )

    def test_invalid_element_list(self) -> None:
        """Test that invalid element lists raise ValueError."""
        with self.assertRaises(ValueError):
            MolToElementCount(element_list=[0, 1])
        with self.assertRaises(ValueError):
            MolToElementCount(element_list=[119])
        with self.assertRaises(ValueError):
            MolToElementCount(element_list=[-1])

    def test_pretransform_single(self) -> None:
        """Test element counting for a single molecule.

        Raises
        ------
        AssertionError
            If pretransform_single does not return an np.ndarray.

        """
        element_count = MolToElementCount(element_list=[6, 8], standardizer=None)
        mol = MolFromSmiles("CCO")  # ethanol: 2 C, 1 O (implicit H not counted)
        result = element_count.pretransform_single(mol)
        if not isinstance(result, np.ndarray):
            raise AssertionError(f"Expected np.ndarray, got {type(result)}")
        self.assertTrue(np.array_equal(result, [2.0, 1.0]))
        self.assertEqual(result.dtype, np.float64)

        element_count = MolToElementCount(element_list=[8], standardizer=None)
        mol = MolFromSmiles("CCO")  # ethanol: 2 C, 1 O (implicit H not counted)
        result = element_count.pretransform_single(mol)
        if not isinstance(result, np.ndarray):
            raise AssertionError(f"Expected np.ndarray, got {type(result)}")
        self.assertTrue(np.array_equal(result, [1.0]))
        self.assertEqual(result.dtype, np.float64)

    def test_pipeline_transform(self) -> None:
        """Test element counting in a pipeline context."""
        pipeline = Pipeline(
            [
                ("smi2mol", SmilesToMol()),
                (
                    "element_count",
                    MolToElementCount(
                        element_list=[1, 6, 7, 8, 16, 35],
                        standardizer=None,
                    ),
                ),
            ],
        )
        smiles = ["CCO", "CCN", "c1ccccc1", "CS(=O)C", "CBr", "c1cc(Br)ccc1S"]
        result = pipeline.fit_transform(smiles)
        self.assertEqual(result.shape, (6, 6))
        expected = np.array(
            [
                #    H   C   N   O   S  Br
                [0.0, 2.0, 0.0, 1.0, 0.0, 0.0],  # CCO: 2C, 1O
                [0.0, 2.0, 1.0, 0.0, 0.0, 0.0],  # CCN: 2C, 1N
                [0.0, 6.0, 0.0, 0.0, 0.0, 0.0],  # benzene: 6C
                [0.0, 2.0, 0.0, 1.0, 1.0, 0.0],  # DMSO: 2C, 1O, 1S
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # CBr: 1C, 1Br
                [0.0, 6.0, 0.0, 0.0, 1.0, 1.0],  # 4-bromothiophenol: 6C, 1S, 1Br
            ],
        )
        self.assertTrue(np.array_equal(result, expected))

    def test_get_set_params(self) -> None:
        """Test get_params and set_params round-trip.

        Raises
        ------
        AssertionError
            If pretransform_single does not return an np.ndarray.

        """
        element_count = MolToElementCount(element_list=[1, 6], standardizer=None)
        params = element_count.get_params(deep=True)
        self.assertEqual(params["element_list"], [1, 6])
        # Deep copy: modifying returned params should not affect original
        params["element_list"].append(7)
        self.assertEqual(element_count.element_list, [1, 6])
        count1 = element_count.pretransform_single(MolFromSmiles("CCO"))
        if not isinstance(count1, np.ndarray):
            raise AssertionError(f"Expected np.ndarray, got {type(count1)}")
        self.assertTrue(np.array_equal(count1, [0.0, 2.0]))

        # set_params should update element_list and feature_names
        element_count.set_params(element_list=[8, 16])
        self.assertEqual(element_count.element_list, [8, 16])
        self.assertEqual(element_count.feature_names, ["Count_O", "Count_S"])
        count2 = element_count.pretransform_single(MolFromSmiles("CCO"))
        if not isinstance(count2, np.ndarray):
            raise AssertionError(f"Expected np.ndarray, got {type(count2)}")
        self.assertTrue(np.array_equal(count2, [1.0, 0.0]))  # 1 O, 0 S

    def test_recreate_from_params(self) -> None:
        """Test that the element can be recreated from its params.

        Raises
        ------
        AssertionError
            If pretransform_single does not return an np.ndarray.

        """
        original = MolToElementCount(element_list=[6, 7, 8], standardizer=None)
        counts_original = original.pretransform_single(MolFromSmiles("CCO"))
        if not isinstance(counts_original, np.ndarray):
            raise AssertionError(f"Expected np.ndarray, got {type(counts_original)}")
        recreated = MolToElementCount(**original.get_params())
        self.assertEqual(original.element_list, recreated.element_list)
        self.assertEqual(original.feature_names, recreated.feature_names)
        counts_recreated = recreated.pretransform_single(MolFromSmiles("CCO"))
        if not isinstance(counts_recreated, np.ndarray):
            raise AssertionError(f"Expected np.ndarray, got {type(counts_recreated)}")
        self.assertTrue(np.array_equal(counts_original, counts_recreated))

    def test_element_list_setter(self) -> None:
        """Test that the element_list setter validates and updates feature names."""
        element_count = MolToElementCount(element_list=[1], standardizer=None)
        self.assertEqual(element_count.feature_names, ["Count_H"])
        self.assertEqual(element_count.n_features, 1)
        element_count.element_list = [6, 7]
        self.assertEqual(element_count.feature_names, ["Count_C", "Count_N"])
        self.assertEqual(element_count.n_features, 2)

        with self.assertRaises(ValueError):
            element_count.element_list = [0]

    def test_element_list_getter_returns_copy(self) -> None:
        """Test that the getter returns a copy, not the internal list."""
        element_count = MolToElementCount(element_list=[1, 6], standardizer=None)
        returned = element_count.element_list
        returned.append(99)
        self.assertEqual(element_count.element_list, [1, 6])


if __name__ == "__main__":
    unittest.main()
