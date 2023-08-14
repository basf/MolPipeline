import unittest
from molpipeline.pipeline import MolPipeline
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.mol2mol.mol2mol_standardization import (
    RemoveStereoInformationPipelineElement,
MetalDisconnectorPipelineElement
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

    def test_metal_disconnector_does_not_lose_ringinfo(self) -> None:
        """Test metal disconnector returns valid molecules containing ring info.

        Returns
        -------
        None
        """

        # example where metal disconnection leads to inconsistent ringinfo -> Sanitization is necessary.
        smiles_uninitialized_ringinfo_after_disconnect = 'OC[C@H]1OC(S[Au])[C@H](O)[C@@H](O)[C@@H]1O'
        pipeline = MolPipeline(
            [
                SmilesToMolPipelineElement(),
                MetalDisconnectorPipelineElement(),
            ]
        )
        mols_processed = pipeline.fit_transform([smiles_uninitialized_ringinfo_after_disconnect])
        self.assertEqual(len(mols_processed), 1)
        # Without additional sanitiziting after disconnecting metals the following would fail with
        # a pre-condition assert from within RDkit.
        self.assertEqual(mols_processed[0].GetRingInfo().NumRings(), 1)

if __name__ == "__main__":
    unittest.main()
