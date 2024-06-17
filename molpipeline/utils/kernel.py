"""Contains functions for molecular similarity."""

from typing import Union

import numpy as np
import numpy.typing as npt
from scipy import sparse


def tanimoto_similarity_sparse(
    matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix
) -> npt.NDArray[np.float64]:
    """Calculate a matrix of tanimoto similarities between feature matrix a and b.

    Parameters
    ----------
    matrix_a: sparse.csr_matrix
        Feature matrix A.
    matrix_b: sparse.csr_matrix
        Feature matrix B.

    Returns
    -------
    npt.NDArray[np.float64]
        Matrix of similarity values between instances of A (rows/first dim) , and instances of B (columns/second dim).
    """
    intersection = matrix_a.dot(matrix_b.transpose()).toarray()
    norm_1 = np.array(matrix_a.multiply(matrix_a).sum(axis=1))
    norm_2 = np.array(matrix_b.multiply(matrix_b).sum(axis=1))
    union = norm_1 + norm_2.T - intersection
    # avoid division by zero https://stackoverflow.com/a/37977222
    return np.divide(
        intersection,
        union,
        out=np.zeros(intersection.shape, dtype=float),
        where=union != 0,
    )


def tanimoto_distance_sparse(
    matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix
) -> npt.NDArray[np.float64]:
    """Calculate a matrix of tanimoto distance between feature matrix a and b.

    Tanimoto distance is defined as 1-similarity.

    Parameters
    ----------
    matrix_a: sparse.csr_matrix
        Feature matrix A.
    matrix_b: sparse.csr_matrix
        Feature matrix B.

    Returns
    -------
    npt.NDArray[np.float64]
        Matrix of similarity values between instances of A (rows/first dim) , and instances of B (columns/second dim).
    """
    return 1 - tanimoto_similarity_sparse(matrix_a, matrix_b)


def self_tanimoto_similarity(
    matrix_a: Union[sparse.csr_matrix, npt.NDArray[np.int_]]
) -> npt.NDArray[np.float64]:
    """Calculate a matrix of tanimoto similarity between feature matrix a and itself.

    Parameters
    ----------
    matrix_a: Union[sparse.csr_matrix, npt.NDArray[np.int_]]
        Feature matrix.

    Returns
    -------
    npt.NDArray[np.float64]
        Square matrix of similarity values between all instances in the matrix.
    """
    if isinstance(matrix_a, np.ndarray):
        sparse_matrix = sparse.csr_matrix(matrix_a)
    elif isinstance(matrix_a, sparse.csr_matrix):
        sparse_matrix = matrix_a
    else:
        raise TypeError(f"Unsupported type: {type(matrix_a)}")
    return tanimoto_similarity_sparse(sparse_matrix, sparse_matrix)


def self_tanimoto_distance(
    matrix_a: Union[sparse.csr_matrix, npt.NDArray[np.int_]]
) -> npt.NDArray[np.float64]:
    """Calculate a matrix of tanimoto distance between feature matrix a and itself.

    Parameters
    ----------
    matrix_a: Union[sparse.csr_matrix, npt.NDArray[np.int_]]
        Feature matrix.

    Returns
    -------
    npt.NDArray[np.float64]
        Square matrix of similarity values between all instances in the matrix.
    """
    return 1 - self_tanimoto_similarity(matrix_a)
