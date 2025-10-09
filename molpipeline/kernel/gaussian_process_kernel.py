"""Kernel objects for use with sklearn GaussianProcess models."""

try:
    from typing import override  # type: ignore
except ImportError:
    from typing_extensions import override

import numpy as np
from numpy import typing as npt
from sklearn.gaussian_process.kernels import (
    Hyperparameter,
    Kernel,
    StationaryKernelMixin,
)

from molpipeline.kernel.tanimoto_functions import tanimoto_similarity_sparse


class TanimotoKernel(StationaryKernelMixin, Kernel):
    """Tanimoto kernel which can be used with sklearn GaussianProcess models."""

    def __init__(self) -> None:
        """Initialize the Tanimoto kernel."""

    @override
    def __call__(  # type: ignore
        self,
        X: npt.NDArray[np.int_],
        Y: npt.NDArray[np.int_] | None = None,
        eval_gradient: bool = False,
    ) -> (
        npt.NDArray[np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ):
        """Calculate the Tanimoto similarity between X and Y.

        Parameters
        ----------
        X : npt.NDArray[np.int_], shape (n_samples_X, n_features)
            First input data.
        Y : npt.NDArray[np.int_], shape (n_samples_Y, n_features), optional
            Second input data. If None, Y is set to X.
        eval_gradient: bool, default: False
            Whether to compute the gradient with respect to the kernel hyperparameters.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel matrix.
        K_gradient : array, shape (n_samples_X, n_samples_Y, n_hyperparameters)
            The gradient of the kernel matrix with respect to the hyperparameters.
            Only returned when eval_gradient is True.

        """
        if Y is None:
            Y = X  # noqa: N806
        sim = tanimoto_similarity_sparse(X, Y)
        if not eval_gradient:
            return sim
        return sim, np.empty((X.shape[0], Y.shape[0], 0))

    @override
    def diag(
        self,
        X: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.float64]:
        """Return the diagonal of the kernel k(X, X).

        Parameters
        ----------
        X : array-like, shape (n_samples_X, n_features)
            Input data.

        Returns
        -------
        npt.NDArray[np.float64]
            Diagonal of the kernel matrix.

        """
        return np.ones(X.shape[0])


class ExponentialTanimotoKernel(StationaryKernelMixin, Kernel):
    """Kernel which can be used with sklearn GaussianProcess models.

    In contrast to the TanimotoKernel, for this kernel the similarity is raised to the
    power of the hyperparameter `exponent`. Exponents greater than 1 decrease small
    similarities stronger than large ones. Thus lead to more broadly distributed
    similarity values. Exponents smaller than 1 lead to similarity values which are
    closer to 1 with a more narrow distribution.

    """

    @property
    def hyperparameter_exponent(self) -> Hyperparameter:
        """The hyperparameter representing the exponent of the kernel."""
        return Hyperparameter("exponent", "numeric", self.exponent_bounds)

    def __init__(
        self,
        exponent: float = 1.0,
        exponent_bounds: tuple[float, float] = (0.001, 5),
    ) -> None:
        """Initialize the Tanimoto kernel.

        Parameters
        ----------
        exponent : float, default = 1.0
            Exponent to which the Tanimoto similarity is raised.
        exponent_bounds : tuple[float, float], default = (0.001, 5)
            Bounds for the exponent hyperparameter during optimization.

        """
        self.exponent = exponent
        self.exponent_bounds = exponent_bounds

    def __repr__(self) -> str:
        """Return string representation of the kernel.

        Returns
        -------
        str
            String representation of the kernel.

        """
        return f"{self.__class__.__name__}(exponent={self.exponent:.3g})"

    @override
    def __call__(
        self,
        X: npt.NDArray[np.int_],
        Y: npt.NDArray[np.int_] | None = None,
        eval_gradient: bool = False,
    ) -> (
        npt.NDArray[np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ):
        """Calculate the Tanimoto similarity between X and Y.

        Parameters
        ----------
        X : npt.NDArray[np.int_], shape (n_samples_X, n_features)
            First input data.
        Y : npt.NDArray[np.int_], shape (n_samples_Y, n_features), optional
            Second input data. If None, Y is set to X.
        eval_gradient: bool, default: False
            Whether to compute the gradient with respect to the kernel hyperparameters.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel matrix.
        K_gradient : array, shape (n_samples_X, n_samples_Y, n_hyperparameters)
            The gradient of the kernel matrix with respect to the hyperparameters.
            Only returned when eval_gradient is True.

        """
        if Y is None:
            Y = X  # noqa: N806
        tanimoto = tanimoto_similarity_sparse(X, Y)
        sim = np.power(tanimoto, self.exponent)
        if not eval_gradient:
            return sim
        grad = np.power(tanimoto, self.exponent) * np.log10(tanimoto)
        grad += np.eye(X.shape[0]) * 1e-9
        sim += np.eye(X.shape[0]) * 1e-9
        return sim, grad[:, :, np.newaxis]

    @override
    def diag(
        self,
        X: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.float64]:
        """Return the diagonal of the kernel k(X, X).

        Parameters
        ----------
        X : array-like, shape (n_samples_X, n_features)
            Input data.

        Returns
        -------
        npt.NDArray[np.float64]
            Diagonal of the kernel matrix.

        """
        return np.ones(X.shape[0])
