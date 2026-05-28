"""Type guards for various types used in the package."""

from typing import Any

import scipy

from typing_extensions import TypeIs


def sparse_type_guard(
    matrix: Any,
) -> TypeIs[scipy.sparse.spmatrix]:
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
    return scipy.sparse.issparse(matrix)
