import os

import numpy as np
import pandas as pd
import unittest

from molpipeline.pipeline import MolPipeline
from molpipeline.pipeline_elements.mol2any.mol2rdkit_phys_chem import MolToRDKitPhysChem
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement

this_file = os.path.dirname(__file__)
data_path = os.path.join(os.path.abspath(this_file), "../../test_data/mol_descriptors.tsv")


class TestMol2RDKitPhyschem(unittest.TestCase):
    def test_descriptor_calculation(self) -> None:
        """Test if the calculation of RDKitPhysChem Descriptors works as expected.

        Compared to precalculated values.

        Returns
        -------
        None
        """
        expected_df = pd.read_csv(data_path, sep="\t")
        descriptor_names = expected_df.drop(columns=["smiles"]).columns.tolist()
        smi2mol = SmilesToMolPipelineElement()
        property_element = MolToRDKitPhysChem(
            normalize=False, descriptor_list=descriptor_names
        )
        pipeline = MolPipeline([smi2mol, property_element])
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
        smi2mol = SmilesToMolPipelineElement()
        property_element = MolToRDKitPhysChem(normalize=True)
        pipeline = MolPipeline([smi2mol, property_element])

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
