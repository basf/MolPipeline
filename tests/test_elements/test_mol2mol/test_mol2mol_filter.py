"""Unittest for MolFilters functionality.

MolFilters flag Molecules as invalid based on the criteria defined in the filter.

"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from molpipeline import ErrorFilter, FilterReinserter, Pipeline
from molpipeline.abstract_pipeline_elements.core import InvalidInstance
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import MolToSmiles
from molpipeline.mol2mol import (
    ComplexFilter,
    ElementFilter,
    InorganicsFilter,
    MixtureFilter,
    RDKitDescriptorsFilter,
    SmartsFilter,
    SmilesFilter,
)
from molpipeline.utils.comparison import compare_recursive
from molpipeline.utils.json_operations import recursive_from_json, recursive_to_json

if TYPE_CHECKING:
    from molpipeline.utils.molpipeline_types import FloatCountRange, IntOrIntCountRange

# pylint: disable=duplicate-code  # test case molecules are allowed to be duplicated
SMILES_ANTIMONY = "[SbH6+3]"
SMILES_BENZENE = "c1ccccc1"
SMILES_CHLOROBENZENE = "Clc1ccccc1"
SMILES_CL_BR = "NC(Cl)(Br)C(=O)O"
SMILES_METAL_AU = "OC[C@H]1OC([S][Au])[C@H](O)[C@@H](O)[C@@H]1O"

SMILES_LIST = [
    SMILES_ANTIMONY,
    SMILES_BENZENE,
    SMILES_CHLOROBENZENE,
    SMILES_METAL_AU,
    SMILES_CL_BR,
]


class ElementFilterTest(unittest.TestCase):
    """Unittest for Elementiflter."""

    def test_element_filter(self) -> None:
        """Test if molecules are filtered correctly by allowed chemical elements."""
        default_atoms_dict = {
            1: (0, None),
            5: (0, None),
            6: (0, None),
            7: (0, None),
            8: (0, None),
            9: (0, None),
            14: (0, None),
            15: (0, None),
            16: (0, None),
            17: (0, None),
            34: (0, None),
            35: (0, None),
            53: (0, None),
        }

        element_filter = ElementFilter()
        self.assertEqual(element_filter.allowed_element_numbers, default_atoms_dict)
        pipeline = Pipeline(
            [
                ("Smiles2Mol", SmilesToMol()),
                ("ElementFilter", element_filter),
                ("Mol2Smiles", MolToSmiles()),
                ("ErrorFilter", ErrorFilter()),
            ],
        )

        test_params_list_with_results = [
            {
                "params": {},
                "result": [SMILES_BENZENE, SMILES_CHLOROBENZENE, SMILES_CL_BR],
            },
            {
                "params": {
                    "ElementFilter__allowed_element_numbers": {
                        6: 6,
                        1: (5, 6),
                        17: (0, 1),
                    },
                },
                "result": [SMILES_BENZENE, SMILES_CHLOROBENZENE],
            },
            {"params": {"ElementFilter__add_hydrogens": False}, "result": []},
        ]

        for test_params in test_params_list_with_results:
            pipeline.set_params(**test_params["params"])
            filtered_smiles = pipeline.fit_transform(SMILES_LIST)
            self.assertEqual(filtered_smiles, test_params["result"])

    def test_json_roundtrip(self) -> None:
        """Test if ElementFilter can be serialized and deserialized.

        Notes
        -----
        It is important to save the ElementFilter as a JSON file and then load it back.
        This is because json.dumps() sets the keys of the dictionary to strings.

        """
        element_filter = ElementFilter()
        json_object = recursive_to_json(element_filter)
        with tempfile.TemporaryDirectory() as temp_folder:
            temp_file_path = Path(temp_folder) / "test.json"
            with temp_file_path.open("w", encoding="UTF-8") as out_file:
                json.dump(json_object, out_file)
            with temp_file_path.open(encoding="UTF-8") as in_file:
                loaded_json_object = json.load(in_file)
        recreated_element_filter = recursive_from_json(loaded_json_object)

        original_params = element_filter.get_params()
        recreated_params = recreated_element_filter.get_params()
        self.assertEqual(original_params.keys(), recreated_params.keys())
        for param_name, original_value in original_params.items():
            with self.subTest(param_name=param_name):
                self.assertTrue(
                    compare_recursive(original_value, recreated_params[param_name]),
                    f"Original: {original_value}, "
                    f"Recreated: {recreated_params[param_name]}",
                )


class ComplexFilterTest(unittest.TestCase):
    """Unittest for ComplexFilter."""

    @staticmethod
    def _create_pipeline() -> Pipeline:
        """Create a pipeline with a complex filter.

        Returns
        -------
        Pipeline
            Pipeline with a complex filter.

        """
        element_filter_1 = ElementFilter({6: 6, 1: 6})
        element_filter_2 = ElementFilter({6: 6, 1: 5, 17: 1})

        multi_element_filter = ComplexFilter(
            (
                ("element_filter_1", element_filter_1),
                ("element_filter_2", element_filter_2),
            ),
        )

        return Pipeline(
            [
                ("Smiles2Mol", SmilesToMol()),
                ("MultiElementFilter", multi_element_filter),
                ("Mol2Smiles", MolToSmiles()),
                ("ErrorFilter", ErrorFilter()),
            ],
        )

    def test_complex_filter(self) -> None:
        """Test if molecules are filtered correctly by allowed chemical elements."""
        pipeline = ComplexFilterTest._create_pipeline()

        test_params_list_with_results = [
            {
                "params": {},
                "result": [SMILES_BENZENE, SMILES_CHLOROBENZENE],
            },
            {
                "params": {"MultiElementFilter__mode": "all"},
                "result": [],
            },
            {
                "params": {
                    "MultiElementFilter__mode": "any",
                    "MultiElementFilter__pipeline_filter_elements__element_filter_1__add_hydrogens": False,  # noqa: E501
                },
                "result": [SMILES_CHLOROBENZENE],
            },
        ]

        for test_params in test_params_list_with_results:
            pipeline.set_params(**test_params["params"])
            filtered_smiles = pipeline.fit_transform(SMILES_LIST)
            self.assertEqual(filtered_smiles, test_params["result"])

    def test_json_serialization(self) -> None:
        """Test if complex filter can be serialized and deserialized."""
        pipeline = ComplexFilterTest._create_pipeline()
        json_object = recursive_to_json(pipeline)
        newpipeline = recursive_from_json(json_object)
        self.assertEqual(json_object, recursive_to_json(newpipeline))

        pipeline_result = pipeline.fit_transform(SMILES_LIST)
        newpipeline_result = newpipeline.fit_transform(SMILES_LIST)
        self.assertEqual(pipeline_result, newpipeline_result)

    def test_complex_filter_non_unique_names(self) -> None:
        """Test if molecules are filtered correctly by allowed chemical elements."""
        element_filter_1 = ElementFilter({6: 6, 1: 6})
        element_filter_2 = ElementFilter({6: 6, 1: 5, 17: 1})

        with self.assertRaises(ValueError):
            ComplexFilter(
                (("filter_1", element_filter_1), ("filter_1", element_filter_2)),
            )


class SmartsSmilesFilterTest(unittest.TestCase):
    """Unittest for SmartsFilter and SmilesFilter."""

    def test_smarts_smiles_filter(self) -> None:
        """Test if molecules are filtered correctly by allowed SMARTS patterns."""
        smarts_pats: dict[str, IntOrIntCountRange] = {
            "c": (4, None),
            "Cl": 1,
        }
        smarts_filter = SmartsFilter(smarts_pats)

        smiles_pats: dict[str, IntOrIntCountRange] = {
            "c1ccccc1": (1, None),
            "Cl": 1,
        }
        smiles_filter = SmilesFilter(smiles_pats)

        for filter_ in [smarts_filter, smiles_filter]:
            new_input_as_list = list(filter_.filter_elements.keys())
            pipeline = Pipeline(
                [
                    ("Smiles2Mol", SmilesToMol()),
                    ("SmartsFilter", filter_),
                    ("Mol2Smiles", MolToSmiles()),
                    ("ErrorFilter", ErrorFilter()),
                ],
            )

            test_params_list_with_results = [
                {
                    "params": {},
                    "result": [SMILES_BENZENE, SMILES_CHLOROBENZENE, SMILES_CL_BR],
                },
                {
                    "params": {"SmartsFilter__keep_matches": False},
                    "result": [SMILES_ANTIMONY, SMILES_METAL_AU],
                },
                {
                    "params": {
                        "SmartsFilter__mode": "all",
                        "SmartsFilter__keep_matches": True,
                    },
                    "result": [SMILES_CHLOROBENZENE],
                },
                {
                    "params": {
                        "SmartsFilter__keep_matches": True,
                        "SmartsFilter__filter_elements": ["I"],
                    },
                    "result": [],
                },
                {
                    "params": {
                        "SmartsFilter__keep_matches": False,
                        "SmartsFilter__mode": "any",
                        "SmartsFilter__filter_elements": new_input_as_list,
                    },
                    "result": [SMILES_ANTIMONY, SMILES_METAL_AU],
                },
            ]

            for test_params in test_params_list_with_results:
                pipeline.set_params(**test_params["params"])
                filtered_smiles = pipeline.fit_transform(SMILES_LIST)
                self.assertEqual(filtered_smiles, test_params["result"])

    def test_smarts_smiles_filter_wrong_pattern(self) -> None:
        """Test if molecules are filtered correctly by allowed SMARTS patterns."""
        smarts_pats: dict[str, IntOrIntCountRange] = {
            "cIOnk": (4, None),
            "cC": 1,
        }
        with self.assertRaises(ValueError):
            SmartsFilter(smarts_pats)

        smiles_pats: dict[str, IntOrIntCountRange] = {
            "cC": (1, None),
            "Cl": 1,
        }
        with self.assertRaises(ValueError):
            SmilesFilter(smiles_pats)

    def _test_smarts_filter_input_types_helper(
        self,
        smarts_input: Any,
    ) -> None:
        """Test different input types for SmartsFilter.

        Parameters
        ----------
        smarts_input : Any
            The input to the SmartsFilter, can be a list, set, Series or dict of
            patterns.

        """
        pipeline = Pipeline(
            [
                ("Smiles2Mol", SmilesToMol()),
                ("SmartsFilter", SmartsFilter(smarts_input)),
            ],
        )
        filtered_list = pipeline.transform(SMILES_LIST)
        self.assertEqual(len(filtered_list), len(SMILES_LIST))
        self.assertIsInstance(filtered_list[0], InvalidInstance)
        self.assertFalse(isinstance(filtered_list[1], InvalidInstance))
        self.assertFalse(isinstance(filtered_list[2], InvalidInstance))
        self.assertIsInstance(filtered_list[3], InvalidInstance)
        self.assertFalse(isinstance(filtered_list[4], InvalidInstance))

    def test_smarts_filter_input_types(self) -> None:
        """Test different input types for SmartsFilter."""
        self._test_smarts_filter_input_types_helper(["c", "Cl"])
        self._test_smarts_filter_input_types_helper({"c", "Cl"})
        self._test_smarts_filter_input_types_helper(pd.Series(["c", "Cl"]))
        self._test_smarts_filter_input_types_helper({"c": (1, None), "Cl": (1, None)})

    def test_smarts_filter_parallel(self) -> None:
        """Test if molecules are filtered correctly.

        This test runs the SmartsFilter in parallel.

        """
        smarts_pats: dict[str, IntOrIntCountRange] = {
            "c": (4, None),
            "Cl": 1,
            "cc": (1, None),
            "ccc": (1, None),
            "cccc": (1, None),
            "ccccc": (1, None),
            "cccccc": (1, None),
            "c1ccccc1": (1, None),
            "cCl": 1,
        }
        smarts_filter = SmartsFilter(smarts_pats, mode="all", n_jobs=2)
        pipeline = Pipeline(
            [
                ("Smiles2Mol", SmilesToMol()),
                ("SmartsFilter", smarts_filter),
                ("Mol2Smiles", MolToSmiles()),
                ("ErrorFilter", ErrorFilter()),
            ],
        )
        filtered_smiles = pipeline.fit_transform(SMILES_LIST)
        self.assertEqual(filtered_smiles, [SMILES_CHLOROBENZENE])


class RDKitDescriptorsFilterTest(unittest.TestCase):
    """Unittest for RDKitDescriptorsFilter."""

    def test_descriptor_filter(self) -> None:
        """Test if molecules are filtered correctly by allowed descriptors."""
        descriptors: dict[str, FloatCountRange] = {
            "MolWt": (None, 190),
            "NumHAcceptors": (2, 10),
        }

        descriptor_filter = RDKitDescriptorsFilter(descriptors)

        pipeline = Pipeline(
            [
                ("Smiles2Mol", SmilesToMol()),
                ("DescriptorsFilter", descriptor_filter),
                ("Mol2Smiles", MolToSmiles()),
                ("ErrorFilter", ErrorFilter()),
            ],
        )
        test_params_list_with_results = [
            {"params": {}, "result": SMILES_LIST},
            {"params": {"DescriptorsFilter__mode": "all"}, "result": [SMILES_CL_BR]},
            {
                "params": {"DescriptorsFilter__keep_matches": False},
                "result": [
                    SMILES_ANTIMONY,
                    SMILES_BENZENE,
                    SMILES_CHLOROBENZENE,
                    SMILES_METAL_AU,
                ],
            },
            {"params": {"DescriptorsFilter__mode": "any"}, "result": []},
            {
                "params": {
                    "DescriptorsFilter__keep_matches": True,
                    "DescriptorsFilter__filter_elements": {"NumHAcceptors": (2.00, 4)},
                },
                "result": [SMILES_CL_BR],
            },
            {
                "params": {
                    "DescriptorsFilter__filter_elements": {"NumHAcceptors": (1.99, 4)},
                },
                "result": [SMILES_CL_BR],
            },
            {
                "params": {
                    "DescriptorsFilter__filter_elements": {"NumHAcceptors": (2.01, 4)},
                },
                "result": [],
            },
            {
                "params": {
                    "DescriptorsFilter__filter_elements": {"NumHAcceptors": (1, 2.00)},
                },
                "result": [SMILES_CL_BR],
            },
            {
                "params": {
                    "DescriptorsFilter__filter_elements": {"NumHAcceptors": (1, 2.01)},
                },
                "result": [SMILES_CL_BR],
            },
            {
                "params": {
                    "DescriptorsFilter__filter_elements": {"NumHAcceptors": (1, 1.99)},
                },
                "result": [],
            },
        ]

        for test_params in test_params_list_with_results:
            pipeline.set_params(**test_params["params"])
            filtered_smiles = pipeline.fit_transform(SMILES_LIST)
            self.assertEqual(filtered_smiles, test_params["result"])


class MixtureFilterTest(unittest.TestCase):
    """Unittest for MixtureFilter."""

    def test_invalidate_mixtures(self) -> None:
        """Test if mixtures are correctly invalidated."""
        mol_list = ["CCC.CC.C", "c1ccccc1.[Na+].[Cl-]", "c1ccccc1"]
        expected_invalidated_mol_list = [None, None, "c1ccccc1"]

        smi2mol = SmilesToMol()
        mixture_filter = MixtureFilter()
        mol2smi = MolToSmiles()
        error_filter = ErrorFilter.from_element_list([smi2mol, mixture_filter, mol2smi])
        error_replacer = FilterReinserter.from_error_filter(error_filter, None)

        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("mixture_filter", mixture_filter),
                ("mol2smi", mol2smi),
                ("error_filter", error_filter),
                ("error_replacer", error_replacer),
            ],
        )
        mols_processed = pipeline.fit_transform(mol_list)
        self.assertEqual(expected_invalidated_mol_list, mols_processed)


class InorganicsFilterTest(unittest.TestCase):
    """Unittest for InorganicsFilter."""

    def test_inorganic_filter(self) -> None:
        """Test if molecules are filtered correctly by allowed chemical elements."""
        smiles2mol = SmilesToMol()
        inorganics_filter = InorganicsFilter()
        mol2smiles = MolToSmiles()
        error_filter = ErrorFilter.from_element_list(
            [smiles2mol, inorganics_filter, mol2smiles],
        )
        pipeline = Pipeline(
            [
                ("Smiles2Mol", smiles2mol),
                ("ElementFilter", inorganics_filter),
                ("Mol2Smiles", mol2smiles),
                ("ErrorFilter", error_filter),
            ],
        )
        filtered_smiles = pipeline.fit_transform(SMILES_LIST)
        self.assertEqual(
            filtered_smiles,
            [SMILES_BENZENE, SMILES_CHLOROBENZENE, SMILES_METAL_AU, SMILES_CL_BR],
        )

        filtered_inroganics = pipeline.fit_transform(["O=C=O", "[O+]#[C-]"])
        self.assertEqual(
            filtered_inroganics,
            [],
        )

        filtered_inroganics = pipeline.fit_transform(InorganicsFilter.CARBON_INORGANICS)
        self.assertEqual(
            filtered_inroganics,
            [],
        )


if __name__ == "__main__":
    unittest.main()
