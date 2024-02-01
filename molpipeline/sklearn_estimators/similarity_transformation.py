"""Sklearn estimators for computing similarity and distance matrices."""

from __future__ import annotations

from typing import Any

try:
    from typing import Self  # pylint: disable=no-name-in-module
except ImportError:
    from typing_extensions import Self

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

from molpipeline.utils.kernel import tanimoto_similarity_sparse


class TanimotoSimilarityToTraining(BaseEstimator, TransformerMixin):
    """Transformer for precomputing tanimoto similarity matrices.

    Attributes
    ----------
    training_matrix: npt.NDArray[np.float_] | csr_matrix | None
        Features seen during fit.
    """

    training_matrix: npt.NDArray[np.float_] | csr_matrix | None

    def __init__(self) -> None:
        """Initialize TanimotoSimilarityToTraining.

        Returns
        -------
        None
        """
        self.training_matrix = None

    def _sim(
        self,
        matrix_a: npt.NDArray[np.float_] | csr_matrix,
        matrix_b: npt.NDArray[np.float_] | csr_matrix,
    ) -> npt.NDArray[np.float_]:
        if not isinstance(matrix_a, csr_matrix):
            matrix_a = csr_matrix(matrix_a)
        if not isinstance(matrix_b, csr_matrix):
            matrix_b = csr_matrix(matrix_b)
        return tanimoto_similarity_sparse(matrix_a, matrix_b)

    def fit(
        self,
        X: npt.NDArray[np.float_] | csr_matrix,  # pylint: disable=invalid-name
        y: npt.NDArray[np.float_] | None = None,  # pylint: disable=unused-argument
    ) -> Self:
        """Fit the model.

        Parameters
        ----------
        X : npt.NDArray[np.float_] | csr_matrix
            Feature matrix to which the similarity matrix is computed.
        y : npt.NDArray[np.float_] | None, optional
            Labels, by default None and never used

        Returns
        -------
        Self
            Fitted model.
        """
        self.training_matrix = X
        return self

    def transform(
        self, X: npt.NDArray[np.float_] | csr_matrix  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float_]:
        """Transform the data.

        Parameters
        ----------
        X : npt.NDArray[np.float_] | csr_matrix
            Feature matrix to which the similarity matrix is computed.

        Returns
        -------
        npt.NDArray[np.float_]
            Similarity matrix of X to the training matrix.
        """
        if self.training_matrix is None:
            raise ValueError("Please fit the transformer before transforming!")
        return self._sim(X, self.training_matrix)

    def fit_transform(
        self,
        X: npt.NDArray[np.float_] | csr_matrix,  # pylint: disable=invalid-name
        y: npt.NDArray[np.float_] | None = None,
        **fit_params: Any,
    ) -> npt.NDArray[np.float_]:
        """Fit the model and transform the data.

        Parameters
        ----------
        X: npt.NDArray[np.float_] | csr_matrix
            Feature matrix to fit the model. Is returended as similarity matrix to itself.
        y: npt.NDArray[np.float_] | None, optional
            Labels, by default None and never used
        **fit_params: Any
            Additional fit parameters. Ignored.

        Returns
        -------
        npt.NDArray[np.float_]
            Similarity matrix of X to itself.
        """
        self.fit(X, y)
        return self.transform(X)


class PrecomputedTanimotoDistance(BaseEstimator, TransformerMixin):
    """Transformer for precomputing tanimoto distance matrices.

    Attributes
    ----------
    training_matrix: npt.NDArray[np.float_] | csr_matrix | None
        Features seen during fit.
    """

    training_matrix: npt.NDArray[np.float_] | csr_matrix | None

    def __init__(self) -> None:
        """Initialize PrecomputedTanimotoDistance.

        Returns
        -------
        None
        """
        self.training_matrix = None

    def _dist(
        self,
        matrix_a: npt.NDArray[np.float_] | csr_matrix,
        matrix_b: npt.NDArray[np.float_] | csr_matrix,
    ) -> npt.NDArray[np.float_]:
        if not isinstance(matrix_a, csr_matrix):
            matrix_a = csr_matrix(matrix_a)
        if not isinstance(matrix_b, csr_matrix):
            matrix_b = csr_matrix(matrix_b)
        return 1 - tanimoto_similarity_sparse(matrix_a, matrix_b)

    def fit(
        self,
        X: npt.NDArray[np.float_] | csr_matrix,  # pylint: disable=invalid-name
        y: npt.NDArray[np.float_] | None = None,  # pylint: disable=unused-argument
    ) -> Self:
        """Fit the model.

        Parameters
        ----------
        X : npt.NDArray[np.float_] | csr_matrix
            Feature matrix to which the distance matrix is computed.
        y : npt.NDArray[np.float_] | None, optional
            Labels, by default None and never used

        Returns
        -------
        Self
            Fitted model.
        """
        self.training_matrix = X
        return self

    def transform(
        self, X: npt.NDArray[np.float_] | csr_matrix  # pylint: disable=invalid-name
    ) -> npt.NDArray[np.float_]:
        """Transform the data.

        Parameters
        ----------
        X : npt.NDArray[np.float_] | csr_matrix
            Feature matrix to which the distance matrix is computed.

        Returns
        -------
        npt.NDArray[np.float_]
            Distance matrix of X to the training matrix.
        """
        if self.training_matrix is None:
            raise ValueError("Please fit the transformer before transforming!")
        return self._dist(X, self.training_matrix)

    def fit_transform(
        self,
        X: npt.NDArray[np.float_] | csr_matrix,  # pylint: disable=invalid-name
        y: npt.NDArray[np.float_] | None = None,
        **fit_params: Any,
    ) -> npt.NDArray[np.float_]:
        """Fit the model and transform the data.

        Parameters
        ----------
        X: npt.NDArray[np.float_] | csr_matrix
            Feature matrix to fit the model. Is returended as distance matrix to itself.
        y: npt.NDArray[np.float_] | None, optional
            Labels, by default None and never used
        **fit_params: Any
            Additional fit parameters. Ignored.

        Returns
        -------
        npt.NDArray[np.float_]
            Distance matrix of X to itself.
        """
        self.fit(X, y)
        return self.transform(X)
