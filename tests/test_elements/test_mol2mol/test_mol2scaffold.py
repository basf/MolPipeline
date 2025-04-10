"""Test the mol2scaffold module."""

from typing import Any
from unittest import TestCase

from molpipeline import Pipeline
from molpipeline.any2mol import AutoToMol
from molpipeline.mol2any import MolToSmiles
from molpipeline.mol2mol.scaffolds import MakeScaffoldGeneric, MurckoScaffold


class TestMurckoScaffold(TestCase):
    """Test the MurckoScaffold class."""

    def test_murcko_scaffold_generation_pipeline(self) -> None:
        """Test the scaffold generation."""
        scaffold_pipeline = Pipeline(
            steps=[
                ("smiles_to_mol", AutoToMol()),
                ("murcko_scaffold", MurckoScaffold()),
                ("scaffold_to_smiles", MolToSmiles()),
            ]
        )
        smiles_list = ["Cc1ccc(=O)[nH]c1", "O=CC1CCC(c2ccccc2)CC1", "CCC"]
        expected_scaffold_list = ["O=c1cccc[nH]1", "c1ccc(C2CCCCC2)cc1", ""]

        scaffold_list = scaffold_pipeline.transform(smiles_list)
        self.assertListEqual(expected_scaffold_list, scaffold_list)


class TestMakeScaffoldGeneric(TestCase):
    """Test the MakeScaffoldGeneric class."""

    def setUp(self) -> None:
        """Set up the pipeline and common variables."""
        self.generic_scaffold_pipeline = Pipeline(
            steps=[
                ("smiles_to_mol", AutoToMol()),
                ("murcko_scaffold", MurckoScaffold()),
                ("make_scaffold_generic", MakeScaffoldGeneric()),
                ("scaffold_to_smiles", MolToSmiles()),
            ]
        )
        self.smiles_list = ["Cc1ccc(=O)[nH]c1", "O=CC1CCC(c2ccccc2)CC1", "CCC"]

    def check_generic_scaffold(
        self, params: dict[str, Any], expected_scaffold_list: list[str]
    ) -> None:
        """Helper function to set parameters and check the results.

        Parameters
        ----------
        params: dict[str, Any]
            Parameters to set for the pipeline.
        expected_scaffold_list: list[str]
            Expected output of the pipeline.
        """
        self.generic_scaffold_pipeline.set_params(**params)
        generic_scaffold_list = self.generic_scaffold_pipeline.transform(
            self.smiles_list
        )
        self.assertListEqual(expected_scaffold_list, generic_scaffold_list)

    def test_generic_scaffold_generation_pipeline(self) -> None:
        """Test the generic scaffold generation."""
        self.check_generic_scaffold(
            params={}, expected_scaffold_list=["CC1CCCCC1", "C1CCC(C2CCCCC2)CC1", ""]
        )

        # Test the generic scaffold generation with generic atoms
        self.check_generic_scaffold(
            params={"make_scaffold_generic__generic_atoms": True},
            expected_scaffold_list=["**1*****1", "*1***(*2*****2)**1", ""],
        )

        # Test the generic scaffold generation with generic bonds
        self.check_generic_scaffold(
            params={
                "make_scaffold_generic__generic_atoms": False,
                "make_scaffold_generic__generic_bonds": True,
            },
            expected_scaffold_list=[
                "C~C1~C~C~C~C~C~1",
                "C1~C~C~C(~C2~C~C~C~C~C~2)~C~C~1",
                "",
            ],
        )

        # Test the generic scaffold generation with generic atoms and bonds
        self.check_generic_scaffold(
            params={
                "make_scaffold_generic__generic_atoms": True,
                "make_scaffold_generic__generic_bonds": True,
            },
            expected_scaffold_list=[
                "*~*1~*~*~*~*~*~1",
                "*1~*~*~*(~*2~*~*~*~*~*~2)~*~*~1",
                "",
            ],
        )
