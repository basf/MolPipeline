import unittest
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.tree import DecisionTreeClassifier

from molpipeline.pipeline import MolPipeline

# from molpipeline.sklearn_pipeline import Pipeline as SkPipeline
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.mol2any.mol2morgan_fingerprint import (
    MolToFoldedMorganFingerprint,
)
from molpipeline.pipeline_elements.mol2mol.mol2mol_standardization import (
    RemoveChargePipelineElement,
    MetalDisconnectorPipelineElement,
    SaltRemoverPipelineElement,
)
from molpipeline.pipeline_elements.mol2any.mol2smiles import MolToSmilesPipelineElement
from molpipeline.utils.matrices import are_equal

from tests.utils.fingerprints import make_sparse_fp


TEST_SMILES = ["CC", "CCO", "COC", "CCCCC", "CCC(-O)O", "CCCN", "CCCXAS"]
CONTAINS_OX = [0, 1, 1, 0, 1, 0, 0]
FP_RADIUS = 2
FP_SIZE = 2048
EXPECTED_OUTPUT = make_sparse_fp(TEST_SMILES[:-1], FP_RADIUS, FP_SIZE)


class PipelineTest(unittest.TestCase):
    def test_fit_transform_single_core(self) -> None:
        # Create pipeline
        pipeline = MolPipeline(
            [
                SmilesToMolPipelineElement(),
                MolToFoldedMorganFingerprint(radius=FP_RADIUS, n_bits=FP_SIZE),
            ],
            handle_nones="record_remove",
        )

        # Run pipeline
        matrix = pipeline.fit_transform(TEST_SMILES)

        # Compare with expected output
        self.assertTrue(are_equal(EXPECTED_OUTPUT, matrix))


if __name__ == "__main__":
    unittest.main()
