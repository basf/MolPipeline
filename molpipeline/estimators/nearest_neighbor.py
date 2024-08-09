"""Nearest neighbor algorithm."""

from __future__ import annotations

from typing import Any, Callable, Literal, Sequence, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


import numpy as np
import numpy.typing as npt
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from molpipeline.utils.value_checks import get_length

__all__ = ["NamedNearestNeighbors"]

Algorithm = Literal["auto", "ball_tree", "kd_tree", "brute"]

SklearnNativeMetrics = Literal[
    "cityblock",
    "cosine",
    "euclidean",
    "haversine",
    "jaccard",
    "l1",
    "l2",
    "manhattan",
    "minkowski",
    "nan_euclidean",
    "precomputed",
]

AllMetrics = Union[
    SklearnNativeMetrics,
    Callable[[Any, Any], float | npt.NDArray[np.float64] | Sequence[float]],
]


class NamedNearestNeighbors(NearestNeighbors):  # pylint: disable=too-many-ancestors
    """NearestNeighbors with a name attribute."""

    learned_names_: npt.NDArray[Any] | None

    def __init__(
        self,
        n_neighbors: int = 5,
        radius: float = 1.0,
        algorithm: Algorithm = "auto",
        leaf_size: int = 30,
        metric: AllMetrics = "minkowski",
        p: int = 2,
        metric_params: dict[str, Any] | None = None,
        n_jobs: int | None = None,
    ):
        """Initialize the nearest neighbor algorithm.

        Parameters
        ----------
        n_neighbors : int, optional (default = 5)
            The number of neighbors to get.
        radius : float, optional (default = 1.0)
            Range of parameter space to use by default for radius_neighbors queries.
        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional (default = 'auto')
            Algorithm used to compute the nearest neighbors.
        leaf_size : int, optional (default = 30)
            Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query,
            as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
        metric : Union[str, Callable], optional (default = 'minkowski')
            The distance metric to use for the tree.
            The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.
        p : int, optional (default = 2)
            Power parameter for the Minkowski metric.
        metric_params : dict, optional (default = None)
            Additional keyword arguments for the metric function.
        n_jobs : int, optional (default = None)
            The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context.
            -1 means using all processors.
        """
        super().__init__(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.learned_names_ = None

    # pylint: disable=arguments-differ, signature-differs
    def fit(
        self,
        X: (
            npt.NDArray[Any] | sparse.csr_matrix | Sequence[Any]
        ),  # pylint: disable=invalid-name
        y: Sequence[Any],  # pylint: disable=invalid-name
    ) -> Self:
        """Fit the model using X as training data.

        Parameters
        ----------
        X : npt.NDArray[Any] | sparse.csr_matrix | Sequence[Any]
            Training data.
        y : Sequence[Any]
            Target values. Here values are used as returned nearest neighbors.
            Must have the same length as X.
            Will be stored as the learned_names_ attribute as npt.NDArray[Any].

        Returns
        -------
        Self
            The instance itself.

        Raises
        ------
        ValueError
            If the input arrays have different lengths or do not have a shape nor len attribute.
        """
        # Check if X and y have the same length
        n_x = get_length(X)  # Allowing for any sequence type
        n_y = get_length(y)
        if n_x != n_y:
            raise ValueError("X and y must have the same length.")

        self.learned_names_ = np.array(y)
        super().fit(X)
        return self

    # pylint: disable=invalid-name
    def predict(
        self,
        X: npt.NDArray[Any] | sparse.csr_matrix | Sequence[Any],
        return_distance: bool = False,
        n_neighbors: int | None = None,
    ) -> npt.NDArray[Any]:
        """Find the k-neighbors of a point.

        Parameters
        ----------
        X : npt.NDArray[Any] | sparse.csr_matrix
            The new data to query.
        return_distance : bool, optional (default = False)
            If True, return the distances to the neighbors of each point.
            Default: False
        n_neighbors : int, optional (default = None)
            Number of neighbors to get. If None, the value set at initialization is used.

        Returns
        -------
        tuple[npt.NDArray[Any], npt.NDArray[np.float64]] | npt.NDArray[Any]
            The indices of the nearest points in the population matrix and the distances to the points.
        """
        if self.learned_names_ is None:
            raise ValueError("The model has not been fitted yet.")
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if return_distance:
            distances, indices = super().kneighbors(
                X, n_neighbors=n_neighbors, return_distance=True
            )
            # stack in such a way that the shape is (n_input, n_neighbors, 2)
            # shape 2 as the neighbor idx and distance are returned
            r_arr = np.stack([self.learned_names_[indices], distances], axis=2)
            return r_arr

        indices = super().kneighbors(X, n_neighbors=n_neighbors, return_distance=False)
        return self.learned_names_[indices]

    def fit_predict(
        self,
        X: npt.NDArray[Any] | sparse.csr_matrix,  # pylint: disable=invalid-name
        y: Sequence[Any],
        return_distance: bool = False,
        n_neighbors: int | None = None,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[np.float64]] | npt.NDArray[Any]:
        """Find the k-neighbors of a point.

        Parameters
        ----------
        X : npt.NDArray[Any] | sparse.csr_matrix
            The new data to query.
        y : Sequence[Any]
            Target values. Here values are used as returned nearest neighbors.
            Must have the same length as X.
        return_distance : bool, optional (default = False)
            If True, return the distances to the neighbors of each point.
            Default: False
        n_neighbors : int, optional (default = None)
            Number of neighbors to get. If None, the value set at initialization is used.

        Returns
        -------
        Tuple[array, array]
            The indices of the nearest points in the population matrix and the distances to the points.
        """
        self.fit(X, y)
        return self.predict(X, return_distance=return_distance, n_neighbors=n_neighbors)
