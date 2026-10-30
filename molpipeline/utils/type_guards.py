"""Type guards for various types used in the package."""

from typing import Any

from typing_extensions import TypeIs

from molpipeline.utils.molpipeline_types import SparseMatrix


def sparse_type_guard(
    matrix: Any,
) -> TypeIs[SparseMatrix]:
    """Type guard to check if a matrix is a scipy sparse matrix.

    Parameters
    ----------
    matrix : Any
        The matrix to check.

    Returns
    -------
    TypeGuard[scipy.sparse.spmatrix]:
        True if the matrix is a scipy sparse matrix, False otherwise.

    """
    return isinstance(matrix, SparseMatrix)
