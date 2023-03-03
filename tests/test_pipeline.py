import unittest
from scipy import sparse
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

from molpipeline.pipeline import MolPipeline
from molpipeline.pipeline_elements.any2mol import Smiles2Mol
from molpipeline.pipeline_elements.mol2fingerprint import Mol2FoldedMorganFingerprint
from molpipeline.utils.matrices import are_equal

TEST_SMILES = ["CC", "CCO", "COC", "CCCCC", "CCC(=O)O"]
FP_RADIUS = 2
FP_SIZE = 2048


def make_sparse_fp(smiles_list: list[str]) -> sparse.csr_matrix:
    vector_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        vector = GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_SIZE)
        vector_list.append(sparse.csr_matrix(vector))
    return sparse.vstack(vector_list)


EXPECTED_OUTPUT = make_sparse_fp(TEST_SMILES)


class PipelineTest(unittest.TestCase):
    def test_fit_transform_single_core(self) -> None:
        # Create pipeline
        pipeline = MolPipeline(
            [
                Smiles2Mol(),
                Mol2FoldedMorganFingerprint(radius=FP_RADIUS, n_bits=FP_SIZE)
            ]
        )

        # Run pipeline
        matrix, label, group = pipeline.fit_transform(TEST_SMILES)

        # Compare with expected output
        self.assertTrue(are_equal(EXPECTED_OUTPUT, matrix))


if __name__ == '__main__':
    unittest.main()
