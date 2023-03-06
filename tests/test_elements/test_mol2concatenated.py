import numpy as np
import unittest

from rdkit import Chem

from molpipeline.pipeline import MolPipeline
from molpipeline.pipeline_elements.mol2any.mol2rdkit_phys_chem import MolToRDKitPhysChem
from molpipeline.pipeline_elements.mol2any.mol2concatinated_vector import (
    MolToConcatenatedVector,
)
from molpipeline.pipeline_elements.mol2any.mol2morgan_fingerprint import (
    Mol2FoldedMorganFingerprint,
)
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement


class TestConcatenatedFingerprint(unittest.TestCase):
    def test_generation(self) -> None:
        smi2mol = SmilesToMolPipelineElement()
        property_element = MolToRDKitPhysChem(normalize=True)
        morgan_element = Mol2FoldedMorganFingerprint()
        concatenated_element = MolToConcatenatedVector(
            [property_element, morgan_element]
        )
        pipeline = MolPipeline([smi2mol, concatenated_element])

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
        output2 = pipeline.transform(smiles)

        mol_list = [Chem.MolFromSmiles(smi) for smi in smiles]
        output3 = np.hstack(
            [
                property_element.transform(mol_list),
                morgan_element.transform(mol_list).toarray(),
            ]
        )
        expected_shape = (
            len(smiles),
            (property_element.n_features + morgan_element.n_bits),
        )
        self.assertTrue(output.shape == expected_shape)
        self.assertTrue(np.abs(output - output2).max() < 0.00001)
        self.assertTrue(np.abs(output - output3).max() < 0.00001)


if __name__ == "__main__":
    unittest.main()
