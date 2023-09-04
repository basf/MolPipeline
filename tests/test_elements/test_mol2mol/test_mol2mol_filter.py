import unittest
from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.mol2mol.mol2mol_filter import (
    ElementFilterPipelineElement,
    MixtureFilterPipelineElement,
)
from molpipeline.pipeline_elements.mol2any.mol2smiles import MolToSmilesPipelineElement
from molpipeline.pipeline_elements.none_handling import (
    NoneFilter,
    NoneFiller,
)

SMILES_ANTIMONY = "[SbH6+3]"
SMILES_BENZENE = "c1ccccc1"
SMILES_CHLOROBENZENE = "Clc1ccccc1"
SMILES_Cl_Br = "NC(Cl)(Br)C(=O)O"
SMILES_METAL_AU = "OC[C@H]1OC(S[Au])[C@H](O)[C@@H](O)[C@@H]1O"


class MolFilterTest(unittest.TestCase):
    def test_element_filter(self) -> None:
        """Test if molecules are filtered correctly by allowed chemical elements.

        Returns
        -------
        None
        """
        smiles2mol = SmilesToMolPipelineElement()
        element_filter = ElementFilterPipelineElement(
            allowed_element_numbers=[
                1,
                5,
                6,
                7,
                8,
                9,
                14,
                15,
                16,
                17,
                34,
                35,
                53,
            ],
        )
        mol2smiles = MolToSmilesPipelineElement()
        none_filter = NoneFilter.from_element_list(
            [smiles2mol, element_filter, mol2smiles]
        )
        pipeline = Pipeline(
            [
                ("Smiles2Mol", smiles2mol),
                ("ElementFilter", element_filter),
                ("Mol2Smiles", mol2smiles),
                ("NoneFilter", none_filter),
            ],
        )
        filtered_smiles = pipeline.fit_transform(
            [
                SMILES_ANTIMONY,
                SMILES_BENZENE,
                SMILES_CHLOROBENZENE,
                SMILES_METAL_AU,
                SMILES_Cl_Br,
            ]
        )
        self.assertEqual(
            filtered_smiles, [SMILES_BENZENE, SMILES_CHLOROBENZENE, SMILES_Cl_Br]
        )

    def test_invalidate_mixtures(self) -> None:
        """Test if mixtures are correctly invalidated.

        Returns
        -------
        None
        """
        mol_list = ["CCC.CC.C", "c1ccccc1.[Na+].[Cl-]", "c1ccccc1"]
        expected_invalidated_mol_list = [None, None, "c1ccccc1"]

        smi2mol = SmilesToMolPipelineElement()
        mixture_filter = MixtureFilterPipelineElement()
        mol2smi = MolToSmilesPipelineElement()
        none_filter = NoneFilter.from_element_list(
            [smi2mol, mixture_filter, mol2smi]
        )
        none_filler = NoneFiller.from_none_filter(none_filter, None)

        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("mixture_filter", mixture_filter),
                ("mol2smi", mol2smi),
                ("none_filter", none_filter),
                ("none_filler", none_filler),
            ]
        )
        mols_processed = pipeline.fit_transform(mol_list)
        self.assertEqual(expected_invalidated_mol_list, mols_processed)


if __name__ == "__main__":
    unittest.main()
