"""Functions of fingerprints for comparing output with molpipline."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from rdkit import Chem, DataStructs

# pylint: disable=no-name-in-module
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import ExplicitBitVect
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


def explicit_bit_vect_list_to_numpy(
    explicit_bit_vect_list: list[ExplicitBitVect],
) -> npt.NDArray[np.int_]:
    """Convert explicitBitVect manually to numpy.

    It is assumed all fingerprints in the list have the same length.

    Parameters
    ----------
    explicit_bit_vect_list: list[ExplicitBitVect]
        List of fingerprints

    Returns
    -------
    npt.NDArray
        Numpy fingerprint matrix.
    """
    if len(explicit_bit_vect_list) == 0:
        return np.empty(
            (
                0,
                0,
            ),
            dtype=int,
        )
    mat = np.empty(
        (len(explicit_bit_vect_list), len(explicit_bit_vect_list[0])), dtype=int
    )
    for i, fingerprint in enumerate(explicit_bit_vect_list):
        DataStructs.ConvertToNumpyArray(fingerprint, mat[i, :])
    return mat
