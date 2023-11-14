"""Test functionality of the pipeline class."""
import unittest

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier

from tests.utils.fingerprints import make_sparse_fp

from molpipeline.pipeline import Pipeline
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.error_handling import ErrorFilter
from molpipeline.pipeline_elements.mol2any.mol2morgan_fingerprint import (
    MolToFoldedMorganFingerprint,
)
from molpipeline.pipeline_elements.mol2any.mol2rdkit_phys_chem import MolToRDKitPhysChem
from molpipeline.pipeline_elements.mol2any.mol2smiles import MolToSmilesPipelineElement
from molpipeline.pipeline_elements.mol2mol.mol2mol_standardization import (
    ChargeParentPipelineElement,
    MetalDisconnectorPipelineElement,
    SaltRemoverPipelineElement,
)
from molpipeline.utils.json_operations import recursive_from_json, recursive_to_json
from molpipeline.utils.matrices import are_equal


TEST_SMILES = ["CC", "CCO", "COC", "CCCCC", "CCC(-O)O", "CCCN"]
FAULTY_TEST_SMILES = ["CCCXAS", "", "O=C(O)C(F)(F)F"]
CONTAINS_OX = [0, 1, 1, 0, 1, 0]
FP_RADIUS = 2
FP_SIZE = 2048
EXPECTED_OUTPUT = make_sparse_fp(TEST_SMILES, FP_RADIUS, FP_SIZE)


class PipelineTest(unittest.TestCase):
    """Unit test for the functionality of the pipeline class."""

    def test_fit_transform_single_core(self) -> None:
        """Test if the generation of the fingerprint matrix works as expected.

        Returns
        -------
        None
        """
        # Create pipeline
        smi2mol = SmilesToMolPipelineElement()
        mol2morgan = MolToFoldedMorganFingerprint(radius=FP_RADIUS, n_bits=FP_SIZE)
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("morgan", mol2morgan),
            ]
        )

        # Run pipeline
        matrix = pipeline.fit_transform(TEST_SMILES)

        # Compare with expected output
        self.assertTrue(are_equal(EXPECTED_OUTPUT, matrix))

    def test_sklearn_pipeline(self) -> None:
        """Test if the pipeline can be used in a sklearn pipeline.

        Returns
        -------
        None
        """
        smi2mol = SmilesToMolPipelineElement()
        mol2morgan = MolToFoldedMorganFingerprint(radius=FP_RADIUS, n_bits=FP_SIZE)
        d_tree = DecisionTreeClassifier()
        s_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("morgan", mol2morgan),
                ("decision_tree", d_tree),
            ]
        )
        s_pipeline.fit(TEST_SMILES, CONTAINS_OX)
        predicted_value_array = s_pipeline.predict(TEST_SMILES)
        for pred_val, true_val in zip(predicted_value_array, CONTAINS_OX):
            self.assertEqual(pred_val, true_val)

    def test_salt_removal(self) -> None:
        """Test if salts are correctly removed from molecules.

        Returns
        -------
        None
        """
        smiles_with_salt_list = ["CCO-[Na]", "CCC(=O)[O-].[Li+]", "CCC(=O)-O-[K]"]
        smiles_without_salt_list = ["CCO", "CCC(=O)O", "CCC(=O)O"]

        smi2mol = SmilesToMolPipelineElement()
        disconnect_metal = MetalDisconnectorPipelineElement()
        salt_remover = SaltRemoverPipelineElement()
        remove_charge = ChargeParentPipelineElement()
        mol2smi = MolToSmilesPipelineElement()

        salt_remover_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("disconnect_metal", disconnect_metal),
                ("salt_remover", salt_remover),
                ("remove_charge", remove_charge),
                ("mol2smi", mol2smi),
            ]
        )
        generated_smiles = salt_remover_pipeline.transform(smiles_with_salt_list)
        for generated_smiles, smiles_without_salt in zip(
            generated_smiles, smiles_without_salt_list
        ):
            self.assertEqual(generated_smiles, smiles_without_salt)

    def test_json_generation(self) -> None:
        """Test that the json representation of a pipeline can be loaded back into a pipeline.

        Returns
        -------
        None
        """

        # Create pipeline
        smi2mol = SmilesToMolPipelineElement()
        metal_disconnector = MetalDisconnectorPipelineElement()
        salt_remover = SaltRemoverPipelineElement()
        physchem = MolToRDKitPhysChem()
        pipeline_element_list = [
            smi2mol,
            metal_disconnector,
            salt_remover,
            physchem,
        ]
        m_pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("metal_disconnector", metal_disconnector),
                ("salt_remover", salt_remover),
                ("physchem", physchem),
            ]
        )

        # Convert pipeline to json
        json_str = recursive_to_json(m_pipeline)
        # Recreate pipeline from json
        loaded_pipeline: Pipeline = recursive_from_json(json_str)
        self.assertTrue(isinstance(loaded_pipeline, Pipeline))
        # Compare pipeline elements
        for loaded_element, original_element in zip(
            loaded_pipeline.steps, pipeline_element_list
        ):
            loaded_params = loaded_element[1].get_params()
            original_params = original_element.get_params()
            for key, value in loaded_params.items():
                if isinstance(value, BaseEstimator):
                    self.assertEqual(type(value), type(original_params[key]))
                else:
                    self.assertEqual(loaded_params[key], original_params[key])

    def test_fit_transform_record_remove_nones(self) -> None:
        """Test if the generation of the fingerprint matrix works as expected.
        Returns
        -------
        None
        """
        smi2mol = SmilesToMolPipelineElement()
        salt_remover = SaltRemoverPipelineElement()
        mol2morgan = MolToFoldedMorganFingerprint(radius=FP_RADIUS, n_bits=FP_SIZE)
        remove_none = ErrorFilter.from_element_list([smi2mol, salt_remover, mol2morgan])
        # Create pipeline
        pipeline = Pipeline(
            [
                ("smi2mol", smi2mol),
                ("salt_remover", salt_remover),
                ("morgan", mol2morgan),
                ("remove_none", remove_none),
            ],
        )

        # Run pipeline
        matrix = pipeline.fit_transform(TEST_SMILES + FAULTY_TEST_SMILES)
        # Compare with expected output (Which is the same as the output without the faulty smiles)
        self.assertTrue(are_equal(EXPECTED_OUTPUT, matrix))


if __name__ == "__main__":
    unittest.main()
