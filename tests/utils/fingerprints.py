from scipy import sparse
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect


def make_sparse_fp(smiles_list: list[str], radius: int, n_bits: int) -> sparse.csr_matrix:
    vector_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        vector = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        vector_list.append(sparse.csr_matrix(vector))
    return sparse.vstack(vector_list)
