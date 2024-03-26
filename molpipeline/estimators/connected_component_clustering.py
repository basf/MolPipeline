"""Connected component clustering estimator."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClusterMixin, _fit_context
from sklearn.utils._param_validation import Interval

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from molpipeline.estimators.algorithm.connected_component_clustering import (
    calc_chunk_size_from_memory_requirement,
    connected_components_iterative_algorithm,
)
from molpipeline.utils.kernel import tanimoto_similarity_sparse


class ConnectedComponentClustering(ClusterMixin, BaseEstimator):
    """Connected component clustering estimator."""

    _parameter_constraints: dict[str, Any] = {
        "distance_threshold": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        distance_threshold: float,
        *,
        max_memory_usage: float = 4.0,
    ) -> None:
        """Initialize connected component clustering estimator.

        Parameters
        ----------
        distance_threshold : float
            Distance threshold for connected component clustering.
        max_memory_usage : float, optional
            Maximum memory usage in GB, by default 4.0 GB
        """
        self.distance_threshold: float = distance_threshold
        self.max_memory_usage: float = max_memory_usage
        self.n_clusters_: int | None = None
        self.labels_: npt.NDArray[np.int32] | None = None

    # pylint: disable=C0103,W0613
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: npt.NDArray[np.float64] | csr_matrix,
        y: npt.NDArray[np.float64] | None = None,
    ) -> Self:
        """Fit connected component clustering estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Self
            Fitted estimator.
        """
        X = self._validate_data(X, ensure_min_samples=2, accept_sparse=True)
        return self._fit(X)

    # pylint: disable=C0103,W0613
    def _fit(self, X: npt.NDArray[np.float64] | csr_matrix) -> Self:
        """Fit connected component clustering estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        Self
            Fitted estimator.
        """
        # convert tanimoto distance to similarity
        similarity_threshold: float = 1 - self.distance_threshold

        # get row chunk size based on 2D dense distance matrix that will be generated
        row_chunk_size = calc_chunk_size_from_memory_requirement(
            X.shape[0] * 2
            + 2,  # the self_tanimoto_distance needs two matrices of X.shape and two additional rows.
            X.shape[0],
            np.dtype("float64").itemsize,
            self.max_memory_usage,
        )

        if row_chunk_size >= X.shape[0]:
            similarity_matrix = tanimoto_similarity_sparse(X, X)
            adjacency_matrix = (similarity_matrix >= similarity_threshold).astype(
                np.int8
            )
            self.n_clusters_, self.labels_ = sparse.csgraph.connected_components(
                adjacency_matrix, directed=False, return_labels=True
            )
        else:
            self.n_clusters_, self.labels_ = connected_components_iterative_algorithm(
                X, similarity_threshold, row_chunk_size
            )
        return self

    def fit_predict(
        self,
        X: npt.NDArray[np.float64] | csr_matrix,  # pylint: disable=C0103
        y: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray[np.int32]:
        """Fit and predict connected component clustering estimator.

        Parameters
        ----------
        X: npt.NDArray[np.float64] | csr_matrix
            Feature matrix of shape  (n_samples, n_features).
        y: Ignored
            Not used, present for API consistency by convention.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        np.ndarray[int]
            Cluster labels.
        """
        # pylint: disable=W0246
        return super().fit_predict(X, y, **kwargs)
