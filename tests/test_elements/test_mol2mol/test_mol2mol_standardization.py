import unittest
from molpipeline.pipeline import MolPipeline
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.mol2mol.mol2mol_standardization import (
    RemoveStereoInformationPipelineElement,
)
from molpipeline.pipeline_elements.mol2any.mol2smiles import MolToSmilesPipelineElement

STEREO_MOL_LIST = ["Br[C@@H](Cl)F"]
NON_STEREO_MOL_LIST = ["FC(Cl)Br"]


class MolStandardizationTest(unittest.TestCase):
    def test_stereo_removal(self) -> None:
        """Test if stereo-information is removed correctly.

        Returns
        -------
        None
        """
        stereo_removal_pipeline = MolPipeline(
            [
                SmilesToMolPipelineElement(),
                RemoveStereoInformationPipelineElement(),
                MolToSmilesPipelineElement(),
            ]
        )
        stereo_removed_mol_list = stereo_removal_pipeline.fit_transform(STEREO_MOL_LIST)
        self.assertEqual(stereo_removed_mol_list, NON_STEREO_MOL_LIST)


if __name__ == "__main__":
    unittest.main()
