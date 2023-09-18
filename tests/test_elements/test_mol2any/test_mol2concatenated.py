"""Test construction of concatenated fingerprints."""
import unittest

import numpy as np
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.mol2any.mol2rdkit_phys_chem import MolToRDKitPhysChem
from molpipeline.pipeline_elements.mol2any.mol2concatinated_vector import (
    MolToConcatenatedVector,
)
from molpipeline.pipeline_elements.mol2any.mol2morgan_fingerprint import (
    MolToFoldedMorganFingerprint,
)
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement


class TestConcatenatedFingerprint(unittest.TestCase):
    """Unittest for MolToConcatenatedVector, which calculates concatenated fingerprints."""

    def test_generation(self) -> None:
        """Test if the feature concatenation works as expected.

        Returns
        -------
        None
        """
        concat_vector_element = MolToConcatenatedVector(
            [MolToRDKitPhysChem(standardizer=StandardScaler()), MolToFoldedMorganFingerprint()]
        )
        smi2mol = SmilesToMolPipelineElement()
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("concat_vector_element", concat_vector_element),
            ]
        )

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
                concat_vector_element.element_list[0].transform(mol_list),
                concat_vector_element.element_list[1].transform(mol_list).toarray(),
            ]
        )
        pyschem_component: MolToRDKitPhysChem
        pyschem_component = concat_vector_element.element_list[0]  # type: ignore
        morgan_component: MolToFoldedMorganFingerprint
        morgan_component = concat_vector_element.element_list[1]  # type: ignore
        expected_shape = (
            len(smiles),
            (pyschem_component.n_features + morgan_component.n_bits),
        )
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(np.abs(output - output2).max() < 0.00001)
        self.assertTrue(np.abs(output - output3).max() < 0.00001)


if __name__ == "__main__":
    unittest.main()
