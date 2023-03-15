import unittest

from molpipeline.pipeline import MolPipeline
from molpipeline.pipeline_elements.any2mol.smiles2mol import SmilesToMolPipelineElement
from molpipeline.pipeline_elements.mol2any.mol2smiles import MolToSmilesPipelineElement


TEST_SMILES = ["NCCCO","abc"]
EXPECTED_OUTPUT = ["NCCCO",None]

class NoneTest(unittest.TestCase):
    def test_easy_none_molpipeline(self) -> None:
        pipeline = MolPipeline([
            SmilesToMolPipelineElement(),
            MolToSmilesPipelineElement(),
            ])
        out = pipeline.transform(TEST_SMILES)
        for pred_val, true_val in zip(out, EXPECTED_OUTPUT):
            self.assertEqual(pred_val, true_val)



