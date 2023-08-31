import unittest
from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.mol2mol.mol2mol_filter import (
    ElementFilterPipelineElement,
)
from molpipeline.pipeline_elements.mol2any.mol2smiles import MolToSmilesPipelineElement
from molpipeline.pipeline_elements.none_handling import NoneFilter

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


if __name__ == "__main__":
    unittest.main()
