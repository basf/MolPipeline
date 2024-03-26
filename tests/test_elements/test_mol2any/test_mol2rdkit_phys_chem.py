"""Test generation of RDKitPhysChem Descriptors."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from molpipeline import Pipeline
from molpipeline.any2mol import SmilesToMol
from molpipeline.mol2any import MolToRDKitPhysChem
from molpipeline.mol2any.mol2rdkit_phys_chem import DEFAULT_DESCRIPTORS

data_path = Path(__file__).parents[2] / "test_data" / "mol_descriptors.tsv"


class TestMol2RDKitPhyschem(unittest.TestCase):
    """Unittest for MolToRDKitPhysChem, which calculates RDKitPhysChem Descriptors."""

    def test_descriptor_list(self) -> None:
        """Test if the descriptor list is as expected.

        Returns
        -------
        None
        """
        expected_descriptors = {
            "MaxAbsEStateIndex",
            "MaxEStateIndex",
            "MinAbsEStateIndex",
            "MinEStateIndex",
            "qed",
            "HeavyAtomMolWt",
            "ExactMolWt",
            "NumValenceElectrons",
            "NumRadicalElectrons",
            "MaxPartialCharge",
            "MinPartialCharge",
            "MaxAbsPartialCharge",
            "MinAbsPartialCharge",
            "FpDensityMorgan1",
            "FpDensityMorgan2",
            "FpDensityMorgan3",
            "BCUT2D_MWHI",
            "BCUT2D_MWLOW",
            "BCUT2D_CHGHI",
            "BCUT2D_CHGLO",
            "BCUT2D_LOGPHI",
            "BCUT2D_LOGPLOW",
            "BCUT2D_MRHI",
            "BCUT2D_MRLOW",
            "AvgIpc",
            "BalabanJ",
            "BertzCT",
            "Chi0",
            "Chi0n",
            "Chi0v",
            "Chi1",
            "Chi1n",
            "Chi1v",
            "Chi2n",
            "Chi2v",
            "Chi3n",
            "Chi3v",
            "Chi4n",
            "Chi4v",
            "HallKierAlpha",
            "Kappa1",
            "Kappa2",
            "Kappa3",
            "LabuteASA",
            "PEOE_VSA1",
            "PEOE_VSA10",
            "PEOE_VSA11",
            "PEOE_VSA12",
            "PEOE_VSA13",
            "PEOE_VSA14",
            "PEOE_VSA2",
            "PEOE_VSA3",
            "PEOE_VSA4",
            "PEOE_VSA5",
            "PEOE_VSA6",
            "PEOE_VSA7",
            "PEOE_VSA8",
            "PEOE_VSA9",
            "SMR_VSA1",
            "SMR_VSA10",
            "SMR_VSA2",
            "SMR_VSA3",
            "SMR_VSA4",
            "SMR_VSA5",
            "SMR_VSA6",
            "SMR_VSA7",
            "SMR_VSA8",
            "SMR_VSA9",
            "SlogP_VSA1",
            "SlogP_VSA10",
            "SlogP_VSA11",
            "SlogP_VSA12",
            "SlogP_VSA2",
            "SlogP_VSA3",
            "SlogP_VSA4",
            "SlogP_VSA5",
            "SlogP_VSA6",
            "SlogP_VSA7",
            "SlogP_VSA8",
            "SlogP_VSA9",
            "SPS",
            "TPSA",
            "EState_VSA1",
            "EState_VSA10",
            "EState_VSA11",
            "EState_VSA2",
            "EState_VSA3",
            "EState_VSA4",
            "EState_VSA5",
            "EState_VSA6",
            "EState_VSA7",
            "EState_VSA8",
            "EState_VSA9",
            "VSA_EState1",
            "VSA_EState10",
            "VSA_EState2",
            "VSA_EState3",
            "VSA_EState4",
            "VSA_EState5",
            "VSA_EState6",
            "VSA_EState7",
            "VSA_EState8",
            "VSA_EState9",
            "FractionCSP3",
            "HeavyAtomCount",
            "NHOHCount",
            "NOCount",
            "NumAliphaticCarbocycles",
            "NumAliphaticHeterocycles",
            "NumAliphaticRings",
            "NumAromaticCarbocycles",
            "NumAromaticHeterocycles",
            "NumAromaticRings",
            "NumHAcceptors",
            "NumHDonors",
            "NumHeteroatoms",
            "NumRotatableBonds",
            "NumSaturatedCarbocycles",
            "NumSaturatedHeterocycles",
            "NumSaturatedRings",
            "RingCount",
            "MolLogP",
            "MolMR",
            "fr_Al_COO",
            "fr_Al_OH",
            "fr_Al_OH_noTert",
            "fr_ArN",
            "fr_Ar_COO",
            "fr_Ar_N",
            "fr_Ar_NH",
            "fr_Ar_OH",
            "fr_COO",
            "fr_COO2",
            "fr_C_O",
            "fr_C_O_noCOO",
            "fr_C_S",
            "fr_HOCCN",
            "fr_Imine",
            "fr_NH0",
            "fr_NH1",
            "fr_NH2",
            "fr_N_O",
            "fr_Ndealkylation1",
            "fr_Ndealkylation2",
            "fr_Nhpyrrole",
            "fr_SH",
            "fr_aldehyde",
            "fr_alkyl_carbamate",
            "fr_alkyl_halide",
            "fr_allylic_oxid",
            "fr_amide",
            "fr_amidine",
            "fr_aniline",
            "fr_aryl_methyl",
            "fr_azide",
            "fr_azo",
            "fr_barbitur",
            "fr_benzene",
            "fr_benzodiazepine",
            "fr_bicyclic",
            "fr_diazo",
            "fr_dihydropyridine",
            "fr_epoxide",
            "fr_ester",
            "fr_ether",
            "fr_furan",
            "fr_guanido",
            "fr_halogen",
            "fr_hdrzine",
            "fr_hdrzone",
            "fr_imidazole",
            "fr_imide",
            "fr_isocyan",
            "fr_isothiocyan",
            "fr_ketone",
            "fr_ketone_Topliss",
            "fr_lactam",
            "fr_lactone",
            "fr_methoxy",
            "fr_morpholine",
            "fr_nitrile",
            "fr_nitro",
            "fr_nitro_arom",
            "fr_nitro_arom_nonortho",
            "fr_nitroso",
            "fr_oxazole",
            "fr_oxime",
            "fr_para_hydroxylation",
            "fr_phenol",
            "fr_phenol_noOrthoHbond",
            "fr_phos_acid",
            "fr_phos_ester",
            "fr_piperdine",
            "fr_piperzine",
            "fr_priamide",
            "fr_prisulfonamd",
            "fr_pyridine",
            "fr_quatN",
            "fr_sulfide",
            "fr_sulfonamd",
            "fr_sulfone",
            "fr_term_acetylene",
            "fr_tetrazole",
            "fr_thiazole",
            "fr_thiocyan",
            "fr_thiophene",
            "fr_unbrch_alkane",
            "fr_urea",
        }
        self.assertSetEqual(expected_descriptors, set(DEFAULT_DESCRIPTORS))

    def test_descriptor_calculation(self) -> None:
        """Test if the calculation of RDKitPhysChem Descriptors works as expected.

        Compared to precalculated values.

        Returns
        -------
        None
        """
        expected_df = pd.read_csv(data_path, sep="\t")
        descriptor_names = expected_df.drop(columns=["smiles"]).columns.tolist()
        smi2mol = SmilesToMol()
        property_element = MolToRDKitPhysChem(
            standardizer=None, descriptor_list=descriptor_names
        )
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("property_element", property_element),
            ]
        )
        smiles = expected_df["smiles"].tolist()
        property_vector = expected_df[descriptor_names].values

        output = pipeline.fit_transform(smiles)
        difference = output - property_vector

        self.assertTrue(difference.max() < 0.0001)  # add assertion here

    def test_descriptor_normalization(self) -> None:
        """Test if the normalization of RDKitPhysChem Descriptors works as expected.

        Returns
        -------
        None
        """
        smi2mol = SmilesToMol()
        property_element = MolToRDKitPhysChem(standardizer=StandardScaler())
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("property_element", property_element),
            ]
        )
        # pylint: disable=duplicate-code  # test case molecules are allowed to be duplicated
        smiles = [
            "CC",
            "CCC",
            "CCCO",
            "CCNCO",
            "C(C)CCO",
            "CCO",
            "CCCN",
            "CCCC",
            "CCOC",
            "COO",
        ]
        output = pipeline.fit_transform(smiles)
        non_zero_descriptors = output[:, (np.abs(output).sum(axis=0) != 0)]
        self.assertTrue(
            non_zero_descriptors.mean(axis=0).max() < 0.0000001
        )  # add assertion here
        self.assertTrue(non_zero_descriptors.std(axis=0).max() < 1.0000001)
        self.assertTrue(non_zero_descriptors.std(axis=0).min() > 0.9999999)


if __name__ == "__main__":
    unittest.main()
