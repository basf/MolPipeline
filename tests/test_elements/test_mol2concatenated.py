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
        concatenated_vector_element = MolToConcatenatedVector(
            [MolToRDKitPhysChem(normalize=True), Mol2FoldedMorganFingerprint()]
        )
        smi2mol = SmilesToMolPipelineElement()
        pipeline = MolPipeline([smi2mol, concatenated_vector_element])

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
                concatenated_vector_element.component_list[0].transform(mol_list),
                concatenated_vector_element.component_list[1]
                .transform(mol_list)
                .toarray(),
            ]
        )
        pyschem_component: MolToRDKitPhysChem
        pyschem_component = concatenated_vector_element.component_list[0]  # type: ignore
        morgan_component: Mol2FoldedMorganFingerprint
        morgan_component = concatenated_vector_element.component_list[1]  # type: ignore
        expected_shape = (
            len(smiles),
            (pyschem_component.n_features + morgan_component.n_bits),
        )
        self.assertTrue(output.shape == expected_shape)
        self.assertTrue(np.abs(output - output2).max() < 0.00001)
        self.assertTrue(np.abs(output - output3).max() < 0.00001)


if __name__ == "__main__":
    unittest.main()
