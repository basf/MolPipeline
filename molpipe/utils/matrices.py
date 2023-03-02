from typing import Iterable
from scipy import sparse


def sparse_from_index_value_dicts(
        row_index_lists: Iterable[dict[int, int]], n_columns: int) -> sparse.csr_matrix:
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
    row_idx = 0
    for row_idx, row_dict in enumerate(row_index_lists):
        data.extend(row_dict.values())
        col_positions.extend(row_dict.keys())
        row_positions.extend([row_idx] * len(row_dict))
    return sparse.csr_matrix((data, (row_positions, col_positions)), shape=(row_idx+1, n_columns))
