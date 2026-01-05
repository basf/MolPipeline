"""Nearest neighbor algorithm."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

from molpipeline.kernel.tanimoto_functions import tanimoto_similarity_sparse
from molpipeline.utils.multi_proc import check_available_cores
from molpipeline.utils.value_checks import get_length

if TYPE_CHECKING:
    from scipy import sparse
    from scipy.sparse import csr_matrix

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

AllMetrics = (
    SklearnNativeMetrics
    | Callable[[Any, Any], float | npt.NDArray[np.float64] | Sequence[float]]
)


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
        n_neighbors : int, default=5
            The number of neighbors to get.
        radius : float, default=1.0
            Range of parameter space to use by default for radius_neighbors queries.
        algorithm : Literal['auto', 'ball_tree', 'kd_tree', 'brute'], default='auto'
            Algorithm used to compute the nearest neighbors.
        leaf_size : int, default=30
            Leaf size passed to BallTree or KDTree. This can affect the speed of the
            construction and query, as well as the memory required to store the tree.
            The optimal value depends on the nature of the problem.
        metric : str | Callable, default = 'minkowski'
            The distance metric to use for the tree.
            The default metric is minkowski, and with p=2 is equivalent to the standard
            Euclidean metric.
        p : int, default=2
            Power parameter for the Minkowski metric.
        metric_params : dict, optional
            Additional keyword arguments for the metric function.
        n_jobs : int, optional
            The number of parallel jobs to run for neighbors search.
            None means 1, unless in a joblib.parallel_backend context.
            A value of -1 means using all processors.

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
        X: (npt.NDArray[Any] | sparse.csr_matrix | Sequence[Any]),  # pylint: disable=invalid-name # noqa: N803
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
            If the input arrays have different lengths or do not have a shape nor len
            attribute.

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
        X: npt.NDArray[Any] | sparse.csr_matrix | Sequence[Any],  # noqa: N803
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
            Number of neighbors to get. If None, the value set at initialization is
            used.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Returns
        -------
        tuple[npt.NDArray[Any], npt.NDArray[np.float64]] | npt.NDArray[Any]
            The indices of the nearest points in the population matrix and the distances
            to the points.

        """
        if self.learned_names_ is None:
            raise ValueError("The model has not been fitted yet.")
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if return_distance:
            distances, indices = super().kneighbors(
                X,
                n_neighbors=n_neighbors,
                return_distance=True,
            )
            # stack in such a way that the shape is (n_input, n_neighbors, 2)
            # shape 2 as the neighbor idx and distance are returned
            return np.stack([self.learned_names_[indices], distances], axis=2)

        indices = super().kneighbors(X, n_neighbors=n_neighbors, return_distance=False)
        return self.learned_names_[indices]

    def fit_predict(
        self,
        X: (npt.NDArray[Any] | sparse.csr_matrix),  # pylint: disable=invalid-name # noqa: N803
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
            Number of neighbors to get. If None, the value set at initialization is
            used.

        Returns
        -------
        Tuple[array, array]
            The indices of the nearest points in the population matrix and the distances
            to the points.

        """
        self.fit(X, y)
        return self.predict(X, return_distance=return_distance, n_neighbors=n_neighbors)


class TanimotoKNN(BaseEstimator):  # pylint: disable=too-few-public-methods
    """k-nearest neighbors (KNN) between data sets using Tanimoto similarity.

    This class uses the Tanimoto similarity to find the k-nearest neighbors of a query
    set in a target set. The full similarity matrix is computed and reduced to the
    k-nearest neighbors. A dot-product based algorithm is used, which is faster than
    using the RDKit native Tanimoto function.

    For handling larger datasets, the computation can be batched to reduce memory usage.
    In addition, the batches can be processed in parallel using joblib.

    Important note: This estimator is not safe to be used in a pipeline with error
                    handling. The retuned indices might not correspond to the correct
                    input molecules.
    """

    target_indices_mapping_: npt.NDArray[np.int64] | None

    def __init__(
        self,
        *,
        k: int | None,
        batch_size: int = 1000,
        n_jobs: int = 1,
    ):
        """Initialize TanimotoKNN.

        Parameters
        ----------
        k: int | None
            Number of nearest neighbors to find. If None, all neighbors are returned.
        batch_size: int, default=1000
            Size of the batches for parallel processing.
        n_jobs: int, default=1
            Number of parallel jobs to run for neighbors search.

        """
        self.target_fingerprints: csr_matrix | None = None
        self.k = k
        self.batch_size = batch_size
        self.n_jobs = check_available_cores(n_jobs)

    def fit(
        self,
        X: sparse.csr_matrix,  # pylint: disable=invalid-name # noqa: N803
        y: Sequence[Any] | None = None,  # pylint: disable=invalid-name
    ) -> Self:
        """Fit the estimator using X as target fingerprint data set.

        Parameters
        ----------
        X : sparse.csr_matrix
            The target fingerprint data set. By calling `predict`, searches are
            performed against this target data set.
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
            If the input arrays have different lengths or do not have a shape nor len
            attribute.

        """
        if y is None:
            y = list(range(X.shape[0]))
        if X.shape[0] != get_length(y):
            raise ValueError("X and y must have the same length.")

        if self.k is None:
            # set k to the number of target fingerprints if k is None
            self.k = X.shape[0]

        self.target_indices_mapping_ = np.array(y)
        self.target_fingerprints = X
        return self

    @staticmethod
    def _reduce_k_equals_1(
        similarity_matrix: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Reduce similarity matrix to k=1 nearest neighbors.

        Uses argmax to find the index of the nearest neighbor in the target
        fingerprints. This function has therefore O(n) time complexity.

        Parameters
        ----------
        similarity_matrix: npt.NDArray[np.float64]
            Similarity matrix of Tanimoto scores between query and target fingerprints.

        Returns
        -------
        npt.NDArray[np.int64]
            Indices of the query's nearest neighbors in the target fingerprints.

        """
        topk_indices = np.argmax(similarity_matrix, axis=1)
        topk_similarities = np.take_along_axis(
            similarity_matrix,
            topk_indices.reshape(-1, 1),
            axis=1,
        ).squeeze()
        return topk_indices, topk_similarities

    def _reduce_k_greater_1_less_n(
        self,
        similarity_matrix: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Reduce similarity matrix to k>1 and k<n nearest neighbors.

        Uses argpartition to find the k-nearest neighbors in the target fingerprints,
        which uses a linear partial sort algorithm. The top k hits must be sorted
        afterward to get the k-nearest neighbors in descending order. This function has
        therefore O(n + k log k) time complexity.

        The indices are sorted descending by similarity.

        Parameters
        ----------
        similarity_matrix: npt.NDArray[np.float64]
            Similarity matrix of Tanimoto scores between query and target fingerprints.

        Raises
        ------
        AssertionError
            If the number of neighbors k has not been set. This should happen in the
            fit function.

        Returns
        -------
        tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]
            Indices of the query's k-nearest neighbors in the target fingerprints and
            the corresponding similarities.

        """
        # Get the indices of the k-nearest neighbors.
        # argpartition returns them unsorted.
        if self.k is None:
            raise AssertionError(
                "The number of neighbors k has not been set. This should happen in the"
                " fit function.",
            )
        topk_indices = np.argpartition(similarity_matrix, kth=-self.k, axis=1)[
            :,
            -self.k :,
        ]
        topk_similarities = np.take_along_axis(similarity_matrix, topk_indices, axis=1)
        # sort the topk_indices descending by similarity
        topk_indices_sorted = np.take_along_axis(
            topk_indices,
            np.fliplr(topk_similarities.argsort(axis=1, kind="stable")),
            axis=1,
        )
        topk_similarities_sorted = np.take_along_axis(
            similarity_matrix,
            topk_indices_sorted,
            axis=1,
        )
        return topk_indices_sorted, topk_similarities_sorted

    @staticmethod
    def _reduce_k_equals_n(
        similarity_matrix: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Reduce similarity matrix to k=n nearest neighbors.

        Parameters
        ----------
        similarity_matrix: npt.NDArray[np.float64]
            Similarity matrix of Tanimoto scores between query and target fingerprints.

        Returns
        -------
        tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]
            Indices of the query's k-nearest neighbors in the target fingerprints and
            the corresponding similarities.

        """
        indices = np.fliplr(similarity_matrix.argsort(axis=1, kind="stable"))
        similarities = np.take_along_axis(similarity_matrix, indices, axis=1)
        return indices, similarities

    def _process_batch(
        self,
        query_batch: sparse.csr_matrix,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Process a batch of query fingerprints.

        Parameters
        ----------
        query_batch: sparse.csr_matrix
            Batch of query fingerprints.

        Returns
        -------
        tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]
            Indices of the k-nearest neighbors in the target fingerprints and the
            corresponding similarities.

        Raises
        ------
        AssertionError
            If the target fingerprints are not set.

        """
        if self.target_fingerprints is None:
            raise AssertionError("Target fingerprints are not set.")

        # compute full similarity matrix for the query batch
        similarity_mat_chunk = tanimoto_similarity_sparse(
            query_batch,
            self.target_fingerprints,
        )

        # reduce the similarity matrix to the k nearest neighbors
        if self.k is None:
            raise AssertionError(
                "The number of neighbors k has not been set. This should happen in the"
                " fit function.",
            )
        if self.k == 1:
            return self._reduce_k_equals_1(similarity_mat_chunk)
        if self.k < self.target_fingerprints.shape[0]:
            return self._reduce_k_greater_1_less_n(similarity_mat_chunk)
        return self._reduce_k_equals_n(similarity_mat_chunk)

    def predict(
        self,
        query_fingerprints: sparse.csr_matrix,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Predict the k-nearest neighbors of the query fingerprints.

        Parameters
        ----------
        query_fingerprints: sparse.csr_matrix
            Query fingerprints.

        Returns
        -------
        tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]
            Indices of the k-nearest neighbors in the target fingerprints and the
            corresponding similarities.

        Raises
        ------
        ValueError
            If the model has not been fitted yet or if the number of features in the
            query fingerprints does not match the number of features in the target
            fingerprints.
        AssertionError
            If the number of neighbors k has not been set. This should happen in the
            fit function.

        """
        if self.target_fingerprints is None:
            raise ValueError("The model has not been fitted yet.")
        if self.k is None:
            raise AssertionError(
                "The number of neighbors k has not been set. This should happen in the "
                "fit function.",
            )
        if query_fingerprints.shape[1] != self.target_fingerprints.shape[1]:
            raise ValueError(
                "The number of features in the query fingerprints does not match the "
                "number of features in the target fingerprints.",
            )
        if self.n_jobs > 1:
            # parallel execution
            with Parallel(n_jobs=self.n_jobs) as parallel:
                # the parallelization is not optimal: the self.target_fingerprints
                # (and query_fingerprints) are copied to each child process worker
                res = parallel(
                    delayed(self._process_batch)(
                        query_fingerprints[i : i + self.batch_size],
                    )
                    for i in range(0, query_fingerprints.shape[0], self.batch_size)
                )
                result_indices_tmp, result_similarities_tmp = zip(*res, strict=True)
                result_indices = np.concatenate(result_indices_tmp)
                result_similarities = np.concatenate(result_similarities_tmp)
        else:
            # single process execution
            result_shape = (
                (query_fingerprints.shape[0], self.k)
                if self.k > 1
                else (query_fingerprints.shape[0],)
            )
            result_indices = np.full(result_shape, -1, dtype=np.int64)
            result_similarities = np.full(result_shape, np.nan, dtype=np.float64)
            for i in range(0, query_fingerprints.shape[0], self.batch_size):
                query_batch = query_fingerprints[i : i + self.batch_size]
                (
                    result_indices[i : i + self.batch_size],
                    result_similarities[i : i + self.batch_size],
                ) = self._process_batch(query_batch)

        return result_indices, result_similarities
