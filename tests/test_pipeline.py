import unittest

from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.tree import DecisionTreeClassifier

from molpipeline.pipeline import MolPipeline

from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.mol2any.mol2morgan_fingerprint import (
    MolToFoldedMorganFingerprint,
)
from molpipeline.pipeline_elements.mol2any.mol2rdkit_phys_chem import MolToRDKitPhysChem
from molpipeline.pipeline_elements.mol2mol.mol2mol_standardization import (
    RemoveChargePipelineElement,
    MetalDisconnectorPipelineElement,
    SaltRemoverPipelineElement,
)
from molpipeline.pipeline_elements.mol2any.mol2smiles import MolToSmilesPipelineElement
from molpipeline.utils.matrices import are_equal

from tests.utils.fingerprints import make_sparse_fp


TEST_SMILES = ["CC", "CCO", "COC", "CCCCC", "CCC(-O)O", "CCCN"]
CONTAINS_OX = [0, 1, 1, 0, 1, 0]
FP_RADIUS = 2
FP_SIZE = 2048
EXPECTED_OUTPUT = make_sparse_fp(TEST_SMILES, FP_RADIUS, FP_SIZE)


class PipelineTest(unittest.TestCase):
    def test_fit_transform_single_core(self) -> None:
        """Test if the generation of the fingerprint matrix works as expected.

        Returns
        -------
        None
        """
        # Create pipeline
        pipeline = MolPipeline(
            [
                SmilesToMolPipelineElement(),
                MolToFoldedMorganFingerprint(radius=FP_RADIUS, n_bits=FP_SIZE),
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
        m_pipeline = MolPipeline(
            [
                SmilesToMolPipelineElement(),
                MolToFoldedMorganFingerprint(radius=FP_RADIUS, n_bits=FP_SIZE),
            ]
        )
        d_tree = DecisionTreeClassifier()
        s_pipeline = SkPipeline(
            [
                ("mol_pipeline", m_pipeline),
                ("decision_tree", d_tree),
            ]
        )
        s_pipeline.fit(TEST_SMILES, CONTAINS_OX)
        predicted_value_array = s_pipeline.predict(TEST_SMILES)
        for pred_val, true_val in zip(predicted_value_array, CONTAINS_OX):
            self.assertEqual(pred_val, true_val)

    def test_slicing(self) -> None:
        """Test if the pipeline can be sliced correctly.

        Returns
        -------
        None
        """
        pipeline_element_list = [
            SmilesToMolPipelineElement(),
            MetalDisconnectorPipelineElement(),
            SaltRemoverPipelineElement(),
            MolToSmilesPipelineElement(),
        ]
        m_pipeline = MolPipeline(pipeline_element_list)

        first_half = m_pipeline[:2]
        second_half = m_pipeline[2:]
        self.assertEqual(
            first_half.pipeline_elements[0].parameters,
            pipeline_element_list[0].parameters,
        )
        self.assertEqual(
            first_half.pipeline_elements[1].parameters,
            pipeline_element_list[1].parameters,
        )
        self.assertEqual(
            second_half.pipeline_elements[0].parameters,
            pipeline_element_list[2].parameters,
        )
        self.assertEqual(
            second_half.pipeline_elements[1].parameters,
            pipeline_element_list[3].parameters,
        )

        concatenated_pipeline = first_half + second_half
        for concat_element, original_element in zip(
            concatenated_pipeline.pipeline_elements, pipeline_element_list
        ):
            self.assertEqual(concat_element.parameters, original_element.parameters)

    def test_salt_removal(self) -> None:
        """Test if salts are correctly removed from molecules.

        Returns
        -------
        None
        """
        smiles_with_salt_list = ["CCO-[Na]", "CCC(=O)[O-].[Li+]", "CCC(=O)-O-[K]"]
        smiles_without_salt_list = ["CCO", "CCC(=O)O", "CCC(=O)O"]

        salt_remover_pipeline = MolPipeline(
            [
                SmilesToMolPipelineElement(),
                MetalDisconnectorPipelineElement(),
                SaltRemoverPipelineElement(),
                RemoveChargePipelineElement(),
                MolToSmilesPipelineElement(),
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
        pipeline_element_list = [
            SmilesToMolPipelineElement(),
            MetalDisconnectorPipelineElement(),
            SaltRemoverPipelineElement(),
            MolToRDKitPhysChem(),
        ]
        m_pipeline = MolPipeline(pipeline_element_list)

        # Convert pipeline to json
        json_str = m_pipeline.to_json()

        # Recreate pipeline from json
        loaded_pipeline = MolPipeline.from_json(json_str)

        # Compare pipeline elements
        for loaded_element, original_element in zip(
            loaded_pipeline.pipeline_elements, pipeline_element_list
        ):
            self.assertEqual(loaded_element.parameters, original_element.parameters)


if __name__ == "__main__":
    unittest.main()
