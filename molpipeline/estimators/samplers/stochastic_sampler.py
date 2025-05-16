"""Stochastic Sampler."""

from collections.abc import Sequence
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_X_y

from molpipeline.estimators.samplers.stochastic_filter import StochasticFilter


class StochasticSampler(BaseEstimator, TransformerMixin):
    """Sampler that uses stochastic filters for sampling.

    Note that combination_method "product" assumes independence of the filters.

    """

    def __init__(
        self,
        filters: Sequence[StochasticFilter],
        n_samples: int,
        combination_method: Literal["product", "mean"] = "product",
        random_state: int | None = None,
    ):
        """Create a new StochasticSampler.

        Parameters
        ----------
        filters : list of StochasticFilter
            List of filter policies to apply.
        n_samples : int
            Number of samples to generate.
        combination_method : str, default="product"
            Method to combine probabilities from multiple filters.
            Options are "product" or "mean".
        random_state : int, RandomState instance or None, default=None
            Controls the randomization of the algorithm.

        Raises
        ------
        ValueError
            If n_samples is not positive or if combination_method is not recognized.

        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if combination_method not in {"product", "mean"}:
            raise ValueError(
                "combination_method must be either 'product' or 'mean'",
            )
        self.filters = filters
        self.n_samples = n_samples
        self.combination_method = combination_method
        self.rng = check_random_state(random_state)

    def _combine_probabilities(
        self,
        filter_probabilities: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Combine probabilities from multiple filters.

        Parameters
        ----------
        filter_probabilities : npt.NDArray[np.float64]
            Array of shape (n_samples, n_filters,).

        Returns
        -------
        combined_probs : array-like
            Combined probabilities for each sample.

        Raises
        ------
        ValueError
            If the combination method is not recognized.

        """
        if self.combination_method == "product":
            combined = np.prod(filter_probabilities, axis=1)
        elif self.combination_method == "mean":
            combined = np.mean(filter_probabilities, axis=1)
        else:
            raise ValueError(f"Invalid combination method: {self.combination_method}")

        combined_sum = combined.sum()
        if combined_sum > 0:
            combined /= combined_sum
        else:
            # fall back to uniform distribution when all probabilities are 0
            combined = np.ones_like(combined) / len(combined)

        return combined

    def calculate_probabilities(
        self,
        X: npt.NDArray[np.float64],  # noqa: N803 # pylint: disable=invalid-name
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculate probabilities for each filter.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            The input samples of shape (n_samples, n_features).
        y : npt.NDArray[np.float64]
            The target values of shape (n_samples,).

        Returns
        -------
        probabilities : npt.NDArray[np.float64]
            The calculated probabilities of shape (n_samples,).

        """
        x_matrix, y = check_X_y(X, y, accept_sparse=["csr", "csc"])

        # get probabilities for each filter
        filter_probabilities = np.zeros((len(x_matrix), len(self.filters)))
        for filter_idx, filter_policy in enumerate(self.filters):
            filter_probabilities[:, filter_idx] = filter_policy.calculate_probabilities(
                x_matrix,
                y,
            )

        return self._combine_probabilities(filter_probabilities)

    def transform(
        self,
        X: npt.NDArray[np.float64],  # noqa: N803 # pylint: disable=invalid-name
        y: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Apply filters and sample from the input data.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            The input samples of shape (n_samples, n_features).
        y : npt.NDArray[np.float64]
            The target values of shape (n_samples,).

        Returns
        -------
        X_sampled : npt.NDArray[np.float64]
            The sampled input of shape (n_samples_new, n_features).
        y_sampled : npt.NDArray[np.float64]
            The sampled targets of shape (n_samples_new,).

        """
        x_matrix, y = check_X_y(X, y, accept_sparse=["csr", "csc"])
        combined_probabilities = self.calculate_probabilities(x_matrix, y)

        # Sample indices based on combined probabilities
        indices = self.rng.choice(
            len(x_matrix),
            size=self.n_samples,
            replace=True,
            p=combined_probabilities,
        )

        # Return sampled data
        return x_matrix[indices], y[indices]

    def fit(
        self,
        _X: npt.NDArray[np.float64],  # noqa: N803 # pylint: disable=invalid-name
        _y: npt.NDArray[np.float64],
    ) -> Self:
        """Maintain scikit-learn API compatibility.

        Parameters
        ----------
        _X : npt.NDArray[np.float64]
            The input samples of shape (n_samples, n_features).
        _y : npt.NDArray[np.float64]
            The target values of shape (n_samples,).

        Returns
        -------
        self : object
            Returns self.

        """
        return self
