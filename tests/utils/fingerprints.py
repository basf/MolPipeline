"""Functions of fingerprints for comparing output with molpipline."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import (
    ExplicitBitVect,
    IntSparseIntVect,
    SparseBitVect,
    UIntSparseIntVect,
)
from scipy import sparse

if TYPE_CHECKING:
    from molpipeline.abstract_pipeline_elements.mol2any.mol2bitvector import (
        FPAssembleOutputOutputType,
    )


def make_sparse_fp(
    smiles_list: list[str],
    radius: int,
    n_bits: int,
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
    morgan_fp = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        vector = morgan_fp.GetFingerprintAsNumPy(mol)
        vector_list.append(sparse.csr_matrix(vector))
    return sparse.vstack(vector_list)


def _is_sparse_rdkit_fp_list(
    fingerprints: FPAssembleOutputOutputType,
) -> TypeGuard[list[UIntSparseIntVect] | list[IntSparseIntVect] | list[SparseBitVect]]:
    """Check if fingerprints are a list of sparse fingerprints.

    Parameters
    ----------
    fingerprints: FPAssembleOutputOutputType
        Fingerprint matrix.

    Returns
    -------
    TypeGuard
        True if fingerprints are a list of sparse fingerprints.

    """
    return (
        all(isinstance(fp, UIntSparseIntVect) for fp in fingerprints)
        or all(isinstance(fp, IntSparseIntVect) for fp in fingerprints)
        or all(isinstance(fp, SparseBitVect) for fp in fingerprints)
    )


def fingerprints_to_numpy(
    fingerprints: FPAssembleOutputOutputType,
) -> npt.NDArray[np.int_]:
    """Convert fingerprints in various types to numpy.

    Parameters
    ----------
    fingerprints: FPAssembleOutputOutputType
        Fingerprint matrix.

    Raises
    ------
    ValueError
        If the fingerprints are not in a supported format.

    Returns
    -------
    npt.NDArray
        Numpy fingerprint matrix.

    """
    if all(isinstance(fp, ExplicitBitVect) for fp in fingerprints):
        return np.array(fingerprints)
    if _is_sparse_rdkit_fp_list(fingerprints):
        return np.array([fp.ToList() for fp in fingerprints])
    if isinstance(fingerprints, sparse.csr_matrix):
        return fingerprints.toarray()
    if isinstance(fingerprints, np.ndarray):
        return fingerprints
    raise ValueError("Unknown fingerprint type. Can not convert to numpy")
