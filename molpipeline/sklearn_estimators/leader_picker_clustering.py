"""LeaderPicker-based clustering estimator."""

from __future__ import annotations

from itertools import compress
from numbers import Real

import numpy as np
import numpy.typing as npt
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect
from rdkit.SimDivFilters import rdSimDivPickers
from sklearn.base import BaseEstimator, ClusterMixin, _fit_context
from sklearn.utils._param_validation import Interval

try:
    from typing import Any, Self
except ImportError:
    from typing_extensions import Self


class LeaderPickerClustering(ClusterMixin, BaseEstimator):
    """LeaderPicker clustering estimator (a sphere exclusion clustering algorithm)."""

    # we use sklearn's input validation to check constraints
    _parameter_constraints: dict[str, Any] = {
        "distance_threshold": [Interval(Real, 0, 1.0, closed="left")],
    }

    def __init__(
        self,
        distance_threshold: float,
    ) -> None:
        """Initialize LeaderPicker clustering estimator.

        Parameters
        ----------
        distance_threshold : float
            Minimum distance between cluster centroids.
        """
        self.distance_threshold: float = distance_threshold
        self.n_clusters_: int | None = None
        self.labels_: npt.NDArray[np.int32] | None = None
        # centroid indices
        self.centroids_: npt.NDArray[np.int32] | None = None

    # pylint: disable=C0103,W0613
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: list[ExplicitBitVect],
        y: npt.NDArray[np.float64] | None = None,
    ) -> Self:
        """Fit leader picker clustering estimator.

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
        return self._fit(X)

    @staticmethod
    def _assign_points_to_clusters_based_on_centroid(
        picks: list[int], fps: list[ExplicitBitVect]
    ) -> tuple[int, npt.NDArray[np.int32]]:
        """Assign points to clusters based on centroid.

        Based on https://rdkit.blogspot.com/2020/11/sphere-exclusion-clustering-with-rdkit.html

        Parameters
        ----------
        picks : list[int]
            Indices of selected cluster centroids to which the remaining data will be assigned.
        fps : list[ExplicitBitVect]
            Fingerprints of the whole data sets.

        Returns
        -------
        tuple[int, np.ndarray[int]]
            Number of clusters and cluster labels.
        """
        labels: npt.NDArray[np.int32] = np.full(len(fps), -1, dtype=np.int32)
        max_similarities = np.full(len(fps), -np.inf, dtype=np.float64)

        for i, pick_idx in enumerate(picks):
            similarities = DataStructs.BulkTanimotoSimilarity(fps[pick_idx], fps)
            max_mask = similarities > max_similarities
            labels[max_mask] = i
            max_similarities[max_mask] = list(compress(similarities, max_mask))

        return np.unique(labels).shape[0], labels

    # pylint: disable=C0103,W0613
    def _fit(self, X: list[ExplicitBitVect]) -> Self:
        """Fit leader picker clustering estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        Self
            Fitted estimator.
        """
        lp = rdSimDivPickers.LeaderPicker()

        # Select centroids. This part is in C++ and fast
        picks = lp.LazyBitVectorPick(
            objects=X,
            poolSize=len(X),
            threshold=self.distance_threshold,
            numThreads=1,  # according to rdkit docu this parameter is not used
            # seed=self.random_state if self.random_state is not None else -1,
        )

        # Assign points to clusters based on centroid
        (
            self.n_clusters_,
            self.labels_,
        ) = self._assign_points_to_clusters_based_on_centroid(picks, X)

        self.centroids_ = np.array(picks)
        return self

    def fit_predict(
        self,
        X: list[ExplicitBitVect],  # pylint: disable=C0103
        y: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> npt.NDArray[np.int32]:
        """Fit and predict leader picker clustering estimator.

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
