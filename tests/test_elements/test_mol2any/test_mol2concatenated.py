"""Test construction of concatenated fingerprints."""

from __future__ import annotations

import unittest
from typing import Any, Callable, get_args, Literal

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.mol2any.mol2concatinated_vector import (
    MolToConcatenatedVector,
)
from molpipeline.pipeline_elements.mol2any.mol2morgan_fingerprint import (
    MolToFoldedMorganFingerprint,
)
from molpipeline.pipeline_elements.mol2any.mol2rdkit_phys_chem import MolToRDKitPhysChem
from tests.utils.fingerprints import explicit_bit_vect_list_to_numpy


class TestConcatenatedFingerprint(unittest.TestCase):
    """Unittest for MolToConcatenatedVector, which calculates concatenated fingerprints."""

    def test_generation(self) -> None:
        """Test if the feature concatenation works as expected.

        Returns
        -------
        None
        """
        fingerprint_morgan_output_types: tuple[Any, ...] = get_args(
            Literal[
                "sparse",
                "dense",
                "explicit_bit_vect",
            ]
        )

        to_numpy_func_dict: dict[str, Callable[[Any], npt.NDArray[np.int_]]] = {
            "sparse": lambda x: x.toarray(),
            "dense": lambda x: x,
            "explicit_bit_vect": explicit_bit_vect_list_to_numpy,
        }

        for fp_output_type in fingerprint_morgan_output_types:

            concat_vector_element = MolToConcatenatedVector(
                [
                    (
                        "RDKitPhysChem",
                        MolToRDKitPhysChem(standardizer=StandardScaler()),
                    ),
                    (
                        "MorganFP",
                        MolToFoldedMorganFingerprint(output_datatype=fp_output_type),
                    ),
                ]
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
                    concat_vector_element.element_list[0][1].transform(mol_list),
                    to_numpy_func_dict[fp_output_type](
                        concat_vector_element.element_list[1][1].transform(mol_list)
                    ),
                ]
            )
            pyschem_component: MolToRDKitPhysChem
            pyschem_component = concat_vector_element.element_list[0][1]  # type: ignore
            morgan_component: MolToFoldedMorganFingerprint
            morgan_component = concat_vector_element.element_list[1][1]  # type: ignore
            expected_shape = (
                len(smiles),
                (pyschem_component.n_features + morgan_component.n_bits),
            )
            self.assertEqual(output.shape, expected_shape)
            self.assertTrue(np.allclose(output, output2))
            self.assertTrue(np.allclose(output, output3))


if __name__ == "__main__":
    unittest.main()
