"""Functions of fingerprints for comparing output with molpipline."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from rdkit import Chem

# pylint: disable=no-name-in-module
from rdkit.Chem import rdFingerprintGenerator as rdkit_fp
from rdkit.DataStructs import ExplicitBitVect, UIntSparseIntVect
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
    morgan_fp = rdkit_fp.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)  # pylint: disable=no-member
        vector = morgan_fp.GetFingerprintAsNumPy(mol)
        vector_list.append(sparse.csr_matrix(vector))
    return sparse.vstack(vector_list)


def fingerprints_to_numpy(
    fingerprints: list[ExplicitBitVect] | sparse.csr_matrix | npt.NDArray[np.int_],
) -> npt.NDArray[np.int_]:
    """Convert fingerprints in various types to numpy.

    Parameters
    ----------
    fingerprints: list[ExplicitBitVect] | sparse.csr_matrix | npt.NDArray[np.int_]
        Fingerprint matrix.

    Returns
    -------
    npt.NDArray
        Numpy fingerprint matrix.
    """
    if all(isinstance(fp, ExplicitBitVect) for fp in fingerprints):
        return np.array(fingerprints)
    if all(isinstance(fp, UIntSparseIntVect) for fp in fingerprints):
        return np.array([fp.ToList() for fp in fingerprints])
    if isinstance(fingerprints, sparse.csr_matrix):
        return fingerprints.toarray()
    if isinstance(fingerprints, np.ndarray):
        return fingerprints
    raise ValueError("Unknown fingerprint type. Can not convert to numpy")
