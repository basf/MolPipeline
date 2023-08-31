import unittest
from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement

from molpipeline.pipeline_elements.mol2any.mol2inchi import (
    MolToInchiPipelineElement,
    MolToInchiKeyPipelineElement,
)

SMILES_ANTIMONY = "[SbH6+3]"
SMILES_BENZENE = "c1ccccc1"
SMILES_CHLOROBENZENE = "Clc1ccccc1"
SMILES_Cl_Br = "NC(Cl)(Br)C(=O)O"
SMILES_METAL_AU = "OC[C@H]1OC(S[Au])[C@H](O)[C@@H](O)[C@@H]1O"


class TestMol2Inchi(unittest.TestCase):
    def test_to_inchi(self) -> None:
        """Test if smiles converted correctly to inchi string.

        Returns
        -------
        None
        """

        input_smiles = ["CN(C)CCOC(C1=CC=CC=C1)C1=CC=CC=C1"]
        expected_inchis = [
            "InChI=1S/C17H21NO/c1-18(2)13-14-19-17(15-9-5-3-6-10-15)16-11-7-4-8-12-16/h3-12,17H,13-14H2,1-2H3"
        ]
        pipeline = Pipeline(
            [
                ("Smiles2Mol", SmilesToMolPipelineElement()),
                ("Mol2Inchi", MolToInchiPipelineElement()),
            ]
        )
        actual_inchis = pipeline.fit_transform(input_smiles)
        self.assertEqual(expected_inchis, actual_inchis)

    def test_to_inchikey(self) -> None:
        """Test if smiles is converted correctly to inchikey string.

        Returns
        -------
        None
        """

        input_smiles = ["CN(C)CCOC(C1=CC=CC=C1)C1=CC=CC=C1"]
        expected_inchikeys = ["ZZVUWRFHKOJYTH-UHFFFAOYSA-N"]

        pipeline = Pipeline(
            [
                ("Smiles2Mol", SmilesToMolPipelineElement()),
                ("Mol2Inchi", MolToInchiKeyPipelineElement()),
            ],
        )
        actual_inchikeys = pipeline.fit_transform(input_smiles)
        self.assertEqual(expected_inchikeys, actual_inchikeys)


if __name__ == "__main__":
    unittest.main()
