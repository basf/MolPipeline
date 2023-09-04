import unittest
from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.mol2mol.mol2mol_standardization import (
    RemoveStereoInformationPipelineElement,
    MetalDisconnectorPipelineElement,
    DeduplicateFragmentsBySmilesElement,
    DeduplicateFragmentsByInchiElement,
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
        smi2mol = SmilesToMolPipelineElement()
        stereo_removal = RemoveStereoInformationPipelineElement()
        mol2smi = MolToSmilesPipelineElement()
        stereo_removal_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("stereo_removal", stereo_removal),
                ("mol2smi", mol2smi),
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
        smiles_uninitialized_ringinfo_after_disconnect = (
            "OC[C@H]1OC(S[Au])[C@H](O)[C@@H](O)[C@@H]1O"
        )
        smi2mol = SmilesToMolPipelineElement()
        disconnect_metals = MetalDisconnectorPipelineElement()
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("disconnect_metals", disconnect_metals),
            ]
        )
        mols_processed = pipeline.fit_transform(
            [smiles_uninitialized_ringinfo_after_disconnect]
        )
        self.assertEqual(len(mols_processed), 1)
        # Without additional sanitiziting after disconnecting metals the following would fail with
        # a pre-condition assert from within RDkit.
        self.assertEqual(mols_processed[0].GetRingInfo().NumRings(), 1)

    def test_duplicate_fragment_by_smiles_removal(self) -> None:
        """Test metal disconnector returns valid molecules containing ring info.

        Returns
        -------
        None
        """
        duplicate_fragment_smiles_lust = [
            "CC.CC.C",
            "c1ccccc1.C1=C-C=C-C=C1",
        ]
        expected_unique_fragment_smiles_list = ["C.CC", "c1ccccc1"]

        smi2mol = SmilesToMolPipelineElement()
        unique_fragments = DeduplicateFragmentsBySmilesElement()
        mol2smi = MolToSmilesPipelineElement()
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("unique_fragments", unique_fragments),
                ("mol2smi", mol2smi),
            ]
        )
        mols_processed = pipeline.fit_transform(duplicate_fragment_smiles_lust)
        self.assertEqual(expected_unique_fragment_smiles_list, mols_processed)

    def test_duplicate_fragment_by_inchi_removal(self) -> None:
        """Test metal disconnector returns valid molecules containing ring info.

        Returns
        -------
        None
        """
        duplicate_fragment_smiles_lust = [
            "CC.CC.C",
            "c1ccccc1.C1=C-C=C-C=C1",
        ]
        expected_unique_fragment_smiles_list = ["C.CC", "c1ccccc1"]

        smi2mol = SmilesToMolPipelineElement()
        unique_fragments = DeduplicateFragmentsByInchiElement()
        mol2smi = MolToSmilesPipelineElement()
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("unique_fragments", unique_fragments),
                ("mol2smi", mol2smi),
            ]
        )
        mols_processed = pipeline.fit_transform(duplicate_fragment_smiles_lust)
        self.assertEqual(expected_unique_fragment_smiles_list, mols_processed)


if __name__ == "__main__":
    unittest.main()
