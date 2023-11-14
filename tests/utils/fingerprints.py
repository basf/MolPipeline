"""Functions of fingerprints for comparing output with molpipline."""
from __future__ import annotations

from rdkit import Chem

# pylint: disable=no-name-in-module
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from scipy import sparse


def make_sparse_fp(
    smiles_list: list[str], radius: int, n_bits: int
) -> sparse.csr_matrix:
    """Create a sparse Morgan fingerprint matrix from a list of SMILES.

    Used in Unittests.

    Parameters
    ----------
    smiles_list: list[str]
        SMILES representations of molecules which will be encoded as fingerprint.
    radius: int
        Radius of features.
    n_bits: int
        Obtained features will be mapped to a vector of size n_bits.

    Returns
    -------
    sparse.csr_matrix
        Feature matrix.
    """
    vector_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)  # pylint: disable=no-member
        vector = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        vector_list.append(sparse.csr_matrix(vector))
    return sparse.vstack(vector_list)
