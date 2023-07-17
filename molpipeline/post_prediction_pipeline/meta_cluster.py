from __future__ import annotations
from typing import Optional

import numpy as np
import numpy.typing as npt


class ClusterMerging:
    def __init__(self, n_clusters: int = 5) -> None:
        self.n_clusters = n_clusters

    def fit(
        self,
        X: npt.NDArray[np.int_],
        y: npt.NDArray[np.int_],
    ) -> None:
        self.fit_predict(X, y)

    def fit_predict(
        self,
        X: npt.NDArray[np.int_],
        y: Optional[npt.NDArray[np.int_]] = None,
    ) -> npt.NDArray[np.int_]:

        if y is None:
            y = np.zeros(X.shape[0], dtype=np.int_)

        unique_categories, category_counts = np.unique(y, return_counts=True)
        cluster_dict = {}
        cluster_magnitude = {}
        for cluster_id in np.unique(X):
            cluster_members = np.where(X == cluster_id)[0]
            member_categories = y[cluster_members]
            cluster_catgory_counts = []
            for category in unique_categories:
                category_members = sum(member_categories == category)
                cluster_catgory_counts.append(category_members)
            cluster_dict[cluster_id] = np.array(cluster_catgory_counts)
            cluster_magnitude[cluster_id] = float(
                np.linalg.norm(cluster_catgory_counts)
            )

        cluster_order = sorted(
            cluster_magnitude.keys(),
            key=lambda c_id: cluster_magnitude[c_id],
            reverse=True,
        )
        optimal_meta_cluster_pop = category_counts / self.n_clusters

        meta_cluster_population = np.zeros((self.n_clusters, len(unique_categories)))
        meta_cluster_vector = np.full_like(y, np.nan)
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
