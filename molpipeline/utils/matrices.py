"""Functions to handle sparse matrices."""

from __future__ import annotations

from typing import Iterable

from scipy import sparse


def sparse_from_index_value_dicts(
    row_index_lists: Iterable[dict[int, int]], n_columns: int
) -> sparse.csr_matrix:
    """Create a sparse matrix from list of dicts.

    Each dict represents one row.
    Keys in dictionary correspond to colum index, values represent values of column.

    Parameters
    ----------
    row_index_lists: Iterable[dict[int, int]]
        Iterable of dicts of which each holds column positions and values.
    n_columns: int
        Total number of columns

    Returns
    -------
    sparse.csr_matrix
        Has shape (len(row_index_lists), n_columns).
    """
    data: list[int] = []
    row_positions: list[int] = []
    col_positions: list[int] = []
    row_idx = -1
    for row_idx, row_dict in enumerate(row_index_lists):
        data.extend(row_dict.values())
        col_positions.extend(row_dict.keys())
        row_positions.extend([row_idx] * len(row_dict))
    if row_idx == -1:
        return sparse.csr_matrix((0, n_columns))

    return sparse.csr_matrix(
        (data, (row_positions, col_positions)), shape=(row_idx + 1, n_columns)
    )


def are_equal(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix) -> bool:
    """Compare if any element is not equal, as this is more efficient.

    Parameters
    ----------
    matrix_a: sparse.csr_matrix
        Matrix A to compare.
    matrix_b: sparse.csr_matrix
        Matrix B to compare.

    Returns
    -------
    bool
        Whether the matrices are equal or not.
    """
    is_unequal_matrix = matrix_a != matrix_b
    number_unequal_elements = int(is_unequal_matrix.nnz)
    return number_unequal_elements == 0
