"""Test MolFilter, which invalidate molecules based on criteria defined in the respective filter."""

import unittest
from typing import Optional, Union

from molpipeline import ErrorFilter, FilterReinserter, Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import MolToSmiles
from molpipeline.mol2mol import (
    DescriptorsFilter,
    ElementFilter,
    InorganicsFilter,
    MixtureFilter,
    SmartsFilter,
    SmilesFilter,
)

# pylint: disable=duplicate-code  # test case molecules are allowed to be duplicated
SMILES_ANTIMONY = "[SbH6+3]"
SMILES_BENZENE = "c1ccccc1"
SMILES_CHLOROBENZENE = "Clc1ccccc1"
SMILES_CL_BR = "NC(Cl)(Br)C(=O)O"
SMILES_METAL_AU = "OC[C@H]1OC(S[Au])[C@H](O)[C@@H](O)[C@@H]1O"

SMILES_LIST = [
    SMILES_ANTIMONY,
    SMILES_BENZENE,
    SMILES_CHLOROBENZENE,
    SMILES_METAL_AU,
    SMILES_CL_BR,
]


class MolFilterTest(unittest.TestCase):
    """Unittest for MolFilter, which invalidate molecules based on criteria defined in the respective filter."""

    def test_element_filter(self) -> None:
        """Test if molecules are filtered correctly by allowed chemical elements."""
        default_atoms_dict = {
            1: (None, None),
            5: (None, None),
            6: (None, None),
            7: (None, None),
            8: (None, None),
            9: (None, None),
            14: (None, None),
            15: (None, None),
            16: (None, None),
            17: (None, None),
            34: (None, None),
            35: (None, None),
            53: (None, None),
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
        filtered_smiles = pipeline.fit_transform(SMILES_LIST)
        self.assertEqual(
            filtered_smiles, [SMILES_BENZENE, SMILES_CHLOROBENZENE, SMILES_CL_BR]
        )
        pipeline.set_params(
            ElementFilter__allowed_element_numbers={6: 6, 1: (5, 6), 17: 1}
        )
        filtered_smiles_2 = pipeline.fit_transform(SMILES_LIST)
        self.assertEqual(filtered_smiles_2, [SMILES_BENZENE, SMILES_CHLOROBENZENE])

    def test_smarts_smiles_filter(self) -> None:
        """Test if molecules are filtered correctly by allowed SMARTS patterns."""
        smarts_pats: dict[str, Union[int, tuple[Optional[int], Optional[int]]]] = {
            "c": (4, None),
            "Cl": 1,
        }
        smarts_filter = SmartsFilter(smarts_pats)

        smiles_pats: dict[str, Union[int, tuple[Optional[int], Optional[int]]]] = {
            "c1ccccc1": (1, None),
            "Cl": 1,
        }
        smiles_filter = SmilesFilter(smiles_pats)

        for filter_ in [smarts_filter, smiles_filter]:
            new_input_as_list = list(filter_.patterns.keys())
            pipeline = Pipeline(
                [
                    ("Smiles2Mol", SmilesToMol()),
                    ("SmartsFilter", filter_),
                    ("Mol2Smiles", MolToSmiles()),
                    ("ErrorFilter", ErrorFilter()),
                ],
            )
            filtered_smiles = pipeline.fit_transform(SMILES_LIST)
            self.assertEqual(
                filtered_smiles, [SMILES_BENZENE, SMILES_CHLOROBENZENE, SMILES_CL_BR]
            )

            pipeline.set_params(SmartsFilter__keep=False)
            filtered_smiles_2 = pipeline.fit_transform(SMILES_LIST)
            self.assertEqual(filtered_smiles_2, [SMILES_ANTIMONY, SMILES_METAL_AU])

            pipeline.set_params(SmartsFilter__mode="all", SmartsFilter__keep=True)
            filtered_smiles_3 = pipeline.fit_transform(SMILES_LIST)
            self.assertEqual(filtered_smiles_3, [SMILES_CHLOROBENZENE])

            pipeline.set_params(SmartsFilter__keep=True, SmartsFilter__patterns=["I"])
            filtered_smiles_4 = pipeline.fit_transform(SMILES_LIST)
            self.assertEqual(filtered_smiles_4, [])

            pipeline.set_params(
                SmartsFilter__keep=False,
                SmartsFilter__mode="any",
                SmartsFilter__patterns=new_input_as_list,
            )
            filtered_smiles_5 = pipeline.fit_transform(SMILES_LIST)
            self.assertEqual(filtered_smiles_5, [SMILES_ANTIMONY, SMILES_METAL_AU])

    def test_descriptor_filter(self) -> None:
        """Test if molecules are filtered correctly by allowed descriptors."""
        descriptors: dict[str, tuple[Optional[float], Optional[float]]] = {
            "MolWt": (None, 190),
            "NumHAcceptors": (2, 10),
        }

        descriptor_filter = DescriptorsFilter(descriptors)

        pipeline = Pipeline(
            [
                ("Smiles2Mol", SmilesToMol()),
                ("DescriptorsFilter", descriptor_filter),
                ("Mol2Smiles", MolToSmiles()),
                ("ErrorFilter", ErrorFilter()),
            ],
        )
        filtered_smiles = pipeline.fit_transform(SMILES_LIST)
        self.assertEqual(filtered_smiles, SMILES_LIST)

        pipeline.set_params(DescriptorsFilter__mode="all")
        filtered_smiles_2 = pipeline.fit_transform(SMILES_LIST)
        self.assertEqual(filtered_smiles_2, [SMILES_CL_BR])

        pipeline.set_params(DescriptorsFilter__keep=False)
        filtered_smiles_3 = pipeline.fit_transform(SMILES_LIST)
        # why is this not
        self.assertEqual(
            filtered_smiles_3,
            [SMILES_ANTIMONY, SMILES_BENZENE, SMILES_CHLOROBENZENE, SMILES_METAL_AU],
        )

        pipeline.set_params(DescriptorsFilter__mode="any")
        filtered_smiles_4 = pipeline.fit_transform(SMILES_LIST)
        self.assertEqual(filtered_smiles_4, [])

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
            ]
        )
        mols_processed = pipeline.fit_transform(mol_list)
        self.assertEqual(expected_invalidated_mol_list, mols_processed)

    def test_inorganic_filter(self) -> None:
        """Test if molecules are filtered correctly by allowed chemical elements."""
        smiles2mol = SmilesToMol()
        inorganics_filter = InorganicsFilter()
        mol2smiles = MolToSmiles()
        error_filter = ErrorFilter.from_element_list(
            [smiles2mol, inorganics_filter, mol2smiles]
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
