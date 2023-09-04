"""Module for merging clusters to optimise the number of each category in the final meta clusters."""
from __future__ import annotations
from typing import Any, Optional

import numpy as np
import numpy.typing as npt


class ClusterMerging:
    """Merge clusters to optimise the number of each category in the final meta clusters.

    Attributes
    ----------
    n_clusters: int
        Number of meta clusters.
    """

    n_clusters: int

    def __init__(self, n_clusters: int = 5) -> None:
        """Initialize ClusterMerging.

        Parameters
        ----------
        n_clusters: int
            Number of meta clusters.

        Returns
        -------
        None
        """
        self.n_clusters = n_clusters

    def fit(
        self,
        X: npt.NDArray[np.int_],
        y: npt.NDArray[np.int_],
    ) -> None:
        """Fit the model with X, which is a cluster assignment.

        Does nothing, but calls fit_predict anyway.

        Parameters
        ----------
        X: npt.NDArray[np.int_]
            Identifiers of assigned clusters.
        y: Optional[npt.NDArray[np.int_]]
            Categories (or label) of each sample. Used for stratification.

        Returns
        -------
        None
        """
        self.fit_predict(X, y)

    def fit_predict(
        self,
        X: npt.NDArray[np.int_],
        y: Optional[npt.NDArray[np.int_]] = None,
    ) -> npt.NDArray[np.int_]:
        """Predict the best meta clusters for X, which is a cluster assignment.

        If y is given, clusters are merged to optmise the number of each category in the final meta clusters.

        Parameters
        ----------
        X: npt.NDArray[np.int_]
            Identifiers of assigned clusters.
        y: Optional[npt.NDArray[np.int_]]
            Categories (or label) of each sample. Used for stratification.

        Returns
        -------
        npt.NDArray[np.int_]
            Assignment of meta clusters.
        """
        if y is None:
            y = np.zeros(X.shape[0], dtype=np.int_)

        # Determine all unique categories and their counts
        unique_categories, category_counts = np.unique(y, return_counts=True)

        # Initialise dictionaries
        cluster_dict = {}  # cluster_id: category_counts
        cluster_magnitude = {}  # cluster_id: magnitude, aka norm of category_counts
        # Determine the category counts and magnitude for each cluster
        for cluster_id in np.unique(X):
            cluster_members = np.where(X == cluster_id)[0]
            member_categories = y[cluster_members]

            # Count the number of each category in the cluster
            cluster_category_counts = []
            for category in unique_categories:
                category_members = int(sum(member_categories == category))
                cluster_category_counts.append(category_members)
            cluster_dict[cluster_id] = np.array(cluster_category_counts, dtype=np.int_)

            # The magnitude of the cluster is the norm of the category counts
            magnitude = float(np.linalg.norm(cluster_category_counts))
            cluster_magnitude[cluster_id] = magnitude

        # Sort clusters by magnitude, so that the largest clusters are assigned first
        cluster_order = sorted(
            cluster_magnitude.keys(),
            key=lambda c_id: cluster_magnitude[c_id],
            reverse=True,
        )
        optimal_meta_cluster_pop = category_counts / self.n_clusters
        meta_cluster_population = np.zeros(
            (self.n_clusters, len(unique_categories)), dtype=np.int_
        )
        meta_cluster_vector = np.full_like(y, -1, dtype=np.int_)
        for cluster_id in cluster_order:
            cluster_vec = cluster_dict[cluster_id]
            meta_cluster_delta = meta_cluster_population - optimal_meta_cluster_pop
            meta_cluster_distances = np.linalg.norm(meta_cluster_delta, axis=1)

            virtual_m_cluster_pos = meta_cluster_population + cluster_vec
            virtual_m_cluster_delta = virtual_m_cluster_pos - optimal_meta_cluster_pop
            virtual_m_cluster_dist = np.linalg.norm(virtual_m_cluster_delta, axis=1)

            gain = meta_cluster_distances - virtual_m_cluster_dist
            best_meta_cluster = np.argmax(gain)
            meta_cluster_population[best_meta_cluster] += cluster_vec
            meta_cluster_vector[X == cluster_id] = best_meta_cluster
        return meta_cluster_vector

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: bool
            Whether to recursively get the parameters from the estimators.

        Returns
        -------
        dict
            Parameters of this estimator.
        """
        return {"n_clusters": self.n_clusters}

    def set_params(self, **parameters: Any) -> None:
        """Set the parameters of this estimator.

        Parameters
        ----------
        parameters: dict
            Parameters to set in the estimator.

        Returns
        -------
        None
        """
        n_clusters = parameters.get("n_clusters", self.n_clusters)
        self.n_clusters = n_clusters