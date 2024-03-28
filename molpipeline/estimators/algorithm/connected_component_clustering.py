"""Connected component clustering algorithm."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from molpipeline.estimators.algorithm.union_find import UnionFindNode
from molpipeline.utils.kernel import tanimoto_similarity_sparse


def calc_chunk_size_from_memory_requirement(
    nof_rows: int, nof_cols: int, itemsize: int, memory_cutoff: float
) -> int:
    """Calculate the chunk size from the memory requirement.

    Parameters
    ----------
    nof_rows: int
        Number of rows of the matrix.
    nof_cols: int
        Number of columns of the matrix.
    itemsize: int
        Itemsize of the matrix.
    memory_cutoff: float
        Memory cutoff in GB.

    Returns
    -------
    int
        Chunk size in number of rows.
    """
    memory_cutoff_byte: float = memory_cutoff * 1024**3
    # get memory requirement of dense matrix
    row_memory: int = nof_cols * itemsize
    # get allowed rows
    allowed_rows = max(int(memory_cutoff_byte / row_memory), 1)
    allowed_rows = min(allowed_rows, nof_rows)
    return allowed_rows


def connected_components_iterative_algorithm(
    feature_mat: csr_matrix, similarity_threshold: float, chunk_size: int = 5000
) -> tuple[int, npt.NDArray[np.int32]]:
    """Compute connected component clustering iteratively.

    This algorithm is suited for large data sets since the complete similarity matrix is not stored in memory at once.

    Parameters
    ----------
    feature_mat: csr_matrix
        Feature matrix from which to calculate row-wise similarities.
    similarity_threshold: float
        Similarity threshold used to determine edges in the graph representation.
    chunk_size: int
        Number of rows for which similarities are determined at one iteration of the algorithm.

    Returns
    -------
    tuple[int, np.ndarray[int]]
        Number of clusters and cluster labels.
    """
    nof_samples = feature_mat.shape[0]
    uf_nodes = [UnionFindNode() for _ in range(nof_samples)]

    for i in range(0, nof_samples, chunk_size):
        mat_chunk = feature_mat[i : i + chunk_size, :]

        similarity_mat_chunk = tanimoto_similarity_sparse(mat_chunk, feature_mat)

        indices = np.transpose(
            np.asarray(similarity_mat_chunk >= similarity_threshold).nonzero()
        )

        for i_idx, j_idx in indices:
            if i + i_idx >= j_idx:
                continue
            uf_nodes[j_idx].union(uf_nodes[i + i_idx])

    return UnionFindNode.get_connected_components(uf_nodes)
