"""Test elements for standardizing molecules."""

import unittest

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import MolToSmiles
from molpipeline.mol2mol import (
    ExplicitHydrogenRemover,
    FragmentDeduplicator,
    IsotopeRemover,
    LargestFragmentChooser,
    MetalDisconnector,
    SaltRemover,
    SolventRemover,
    StereoRemover,
    TautomerCanonicalizer,
    Uncharger,
)

STEREO_MOL_LIST = ["Br[C@@H](Cl)F", "Br[C@H](Cl)F.I[C@@H](Cl)F"]
NON_STEREO_MOL_LIST = ["FC(Cl)Br", "FC(Cl)Br.FC(Cl)I"]


class MolStandardizationTest(unittest.TestCase):
    """Test if the mol2mol standardization elements work as expected."""

    def test_stereo_removal(self) -> None:
        """Test if stereo-information is removed correctly.

        Returns
        -------
        None
        """
        smi2mol = SmilesToMol()
        stereo_removal = StereoRemover()
        mol2smi = MolToSmiles()
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
        smi2mol = SmilesToMol()
        disconnect_metals = MetalDisconnector()
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

    def test_duplicate_fragment_by_hash_removal(self) -> None:
        """Test if duplicate fragements are correctly removed by DeduplicateFragmentsByInchiElement.

        Returns
        -------
        None
        """
        duplicate_fragment_smiles_lust = [
            "CC.CC.C",
            "c1ccccc1.C1=C-C=C-C=C1",
            "CCC.CC.CC.C",
            "CC1=C[NH]C=N1.CC1=CN=C[NH]1",
        ]
        expected_unique_fragment_smiles_list = [
            "C.CC",
            "c1ccccc1",
            "C.CC.CCC",
            "Cc1c[nH]cn1",
        ]

        smi2mol = SmilesToMol()
        unique_fragments = FragmentDeduplicator()
        mol2smi = MolToSmiles()
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("unique_fragments", unique_fragments),
                ("mol2smi", mol2smi),
            ]
        )
        mols_processed = pipeline.fit_transform(duplicate_fragment_smiles_lust)
        self.assertEqual(expected_unique_fragment_smiles_list, mols_processed)

    def test_tautomer_canonicalization(self) -> None:
        """Test if correct tautomers are generated.

        Returns
        -------
        None
        """
        non_canonical_tautomer_list = [
            "Oc1c(cccc3)c3nc2ccncc12",
            "CN=c1nc[nH]cc1",
            "CN=c1nc[nH]cc1.Oc1c(cccc3)c3nc2ccncc12",
        ]
        canonical_tautomer_list = [
            "O=c1c2ccccc2[nH]c2ccncc12",
            "CNc1ccncn1",
            "CNc1ccncn1.O=c1c2ccccc2[nH]c2ccncc12",
        ]

        smi2mol = SmilesToMol()
        canonical_tautomer = TautomerCanonicalizer()
        mol2smi = MolToSmiles()
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("canonical_tautomer", canonical_tautomer),
                ("mol2smi", mol2smi),
            ]
        )
        mols_processed = pipeline.fit_transform(non_canonical_tautomer_list)
        self.assertEqual(canonical_tautomer_list, mols_processed)

    def test_charge_neutralization(self) -> None:
        """Test if charge neutralization works as expected.

        Returns
        -------
        None
        """
        mol_list = ["CC(=O)-[O-]", "CC(=O)-[O-].C[NH+](C)C"]
        expected_charge_neutralized_smiles_list = ["CC(=O)O", "CC(=O)O.CN(C)C"]

        smi2mol = SmilesToMol()
        charge_neutralizer = Uncharger()
        mol2smi = MolToSmiles()
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("charge_neutralizer", charge_neutralizer),
                ("mol2smi", mol2smi),
            ]
        )
        mols_processed = pipeline.fit_transform(mol_list)
        self.assertEqual(expected_charge_neutralized_smiles_list, mols_processed)

    def test_largest_fragment_chooser_element(self) -> None:
        """Test if largest fragment chooser element works as expected.

        Returns
        -------
        None
        """
        mol_list = ["CCC.CC.C", "c1ccccc1.[Na+].[Cl-]"]
        expected_largest_fragment_smiles_list = ["CCC", "c1ccccc1"]

        smi2mol = SmilesToMol()
        largest_fragment_chooser = LargestFragmentChooser()
        mol2smi = MolToSmiles()
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("largest_fragment_chooser", largest_fragment_chooser),
                ("mol2smi", mol2smi),
            ]
        )
        mols_processed = pipeline.fit_transform(mol_list)
        self.assertEqual(expected_largest_fragment_smiles_list, mols_processed)

    def test_salt_removal_pipeline_element(self) -> None:
        """Test if salt removal pipeline element works as expected.

        Returns
        -------
        None
        """

        mol_list = [
            "[Na+].ClCC(=O)[O-]",
            "[Ca+2].BrCC(=O)[O-].CCC(=O)[O-]",
        ]
        expected_smiles_list = [
            "O=C([O-])CCl",
            "CCC(=O)[O-].O=C([O-])CBr",
        ]
        smitomol = SmilesToMol()
        salt_remover = SaltRemover()
        mol2smi = MolToSmiles()
        pipeline = Pipeline(
            [
                ("smitomol", smitomol),
                ("salt_remover", salt_remover),
                ("mol2smi", mol2smi),
            ]
        )
        mols_processed = pipeline.fit_transform(mol_list)
        self.assertEqual(expected_smiles_list, mols_processed)

    def test_solvent_removal_pipeline_element(self) -> None:
        """Test if solvent removal pipeline element works as expected.

        Returns
        -------
        None
        """

        mol_list = [
            "[OH2].c1ccccc1",
            "ClCCl.c1ccccc1",
            "ClC(Cl)Cl.c1ccccc1",
            "CCOC(=O)C.c1ccccc1",
            "CO.c1ccccc1",
            "CC(C)O.c1ccccc1",
            "CC(=O)C.c1ccccc1",
            "CS(=O)C.c1ccccc1",
            "CCO.c1ccccc1",
            "[OH2].ClCCl.ClC(Cl)Cl.CCOC(=O)C.CO.CC(C)O.CC(=O)C.CS(=O)C.CCO.c1ccccc1",
        ]
        expected_largest_fragment_smiles_list = [
            "c1ccccc1",
            "c1ccccc1",
            "c1ccccc1",
            "c1ccccc1",
            "c1ccccc1",
            "c1ccccc1",
            "c1ccccc1",
            "c1ccccc1",
            "c1ccccc1",
            "c1ccccc1",
        ]

        smi2mol = SmilesToMol()
        fragment_remover = SolventRemover()
        mol2smi = MolToSmiles()
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("fragment_remover", fragment_remover),
                ("mol2smi", mol2smi),
            ]
        )
        mols_processed = pipeline.fit_transform(mol_list)
        self.assertEqual(expected_largest_fragment_smiles_list, mols_processed)

    def test_isotopeinfo_removal_pipeline_element(self) -> None:
        """Test if isotopinfo removal pipeline element works as expected.

        Returns
        -------
        None
        """

        mol_list = ["[2H]c1ccccc1", "CC[13CH2][19F]"]
        expected_largest_fragment_smiles_list = ["[H]c1ccccc1", "CCCF"]

        smi2mol = SmilesToMol()
        fragment_remover = IsotopeRemover()
        mol2smi = MolToSmiles()
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("isotope_info_remover", fragment_remover),
                ("mol2smi", mol2smi),
            ]
        )
        mols_processed = pipeline.fit_transform(mol_list)
        self.assertEqual(expected_largest_fragment_smiles_list, mols_processed)

    def test_explicit_hydrogen_removal_pipeline_element(self) -> None:
        """Test if explicit hydrogen removal pipeline element works as expected.

        Returns
        -------
        None
        """

        mol_list = ["[H]c1ccccc1", "Cc1cncn(-[H])1", "[H][H]"]
        expected_largest_fragment_smiles_list = ["c1ccccc1", "Cc1cnc[nH]1", "[H][H]"]

        smi2mol = SmilesToMol()
        fragment_remover = ExplicitHydrogenRemover()
        mol2smi = MolToSmiles()

        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("explicit_hydrogen_remover", fragment_remover),
                ("mol2smi", mol2smi),
            ]
        )
        mols_processed = pipeline.fit_transform(mol_list)
        self.assertEqual(expected_largest_fragment_smiles_list, mols_processed)


if __name__ == "__main__":
    unittest.main()
