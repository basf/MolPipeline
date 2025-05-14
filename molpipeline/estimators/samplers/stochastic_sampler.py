from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_X_y


class StochasticSampler(BaseEstimator, TransformerMixin):
    """Sampler that uses multiple stochastic filters to determine sampling probabilities.

    Parameters
    ----------
    filters : list of StochasticFilter
        List of filter policies to apply.
    n_samples : int or None, default=None
        Number of samples to generate. If None, will match the size of X.
    combination_method : {'product', 'mean', 'min', 'max'}, default='product'
        Method to combine probabilities from multiple filters.
    random_state : int, RandomState instance or None, default=None
        Controls the randomization of the algorithm.

    """

    def __init__(
        self,
        filters: Sequence[StochasticFilter],
        n_samples: int | None = None,
        combination_method: str = "product",
        random_state: int | None = None,
    ):
        self.filters = filters
        self.n_samples = n_samples
        self.combination_method = combination_method
        self.random_state = random_state

        # Validate combination method
        valid_methods = ["product", "mean", "min", "max"]
        if self.combination_method not in valid_methods:
            raise ValueError(f"combination_method must be one of {valid_methods}")

    def _combine_probabilities(self, probabilities_list: npt.NDArray) -> npt.NDArray:
        """Combine probabilities from multiple filters.

        Parameters
        ----------
        probabilities_list : list of array-like
            List of probability arrays from each filter.

        Returns
        -------
        combined_probs : array-like
            Combined probabilities for each sample.

        """
        if not probabilities_list:
            raise ValueError("No probabilities to combine")

        if len(probabilities_list) == 1:
            return probabilities_list[0]

        if self.combination_method == "product":
            combined = np.prod(probabilities_list, axis=0)

        elif self.combination_method == "mean":
            combined = np.mean(probabilities_list, axis=0)

        elif self.combination_method == "min":
            combined = np.min(probabilities_list, axis=0)

        elif self.combination_method == "max":
            combined = np.max(probabilities_list, axis=0)
        else:
            raise AssertionError("Invalid combination method")

        # # Normalize to ensure probabilities sum to 1
        # if combined.sum() > 0:
        #     combined = combined / combined.sum()
        # else:
        #     # If all zeros, use uniform distribution
        #     combined = np.ones_like(combined) / len(combined)

        return combined

    def transform(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Apply filters and sample from the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        X_sampled : array-like of shape (n_samples_new, n_features)
            The sampled input.
        y_sampled : array-like of shape (n_samples_new,)
            The sampled targets.

        """
        X, y = check_X_y(X, y, accept_sparse=["csr", "csc"])
        rng = check_random_state(self.random_state)

        # Determine the number of samples to generate
        n_samples = self.n_samples if self.n_samples is not None else len(X)

        # Apply each filter to get probabilities
        filter_probabilities = np.zeros((len(X), len(self.filters)))
        for filter_idx, filter_policy in enumerate(self.filters):
            filter_probabilities[:, filter_idx] = filter_policy.calculate_probabilities(
                X,
                y,
            )

        # Combine probabilities
        combined_probabilities = self._combine_probabilities(filter_probabilities)

        # normalize combined probabilities because numpy's choices needs it
        if combined_probabilities.sum() > 0:
            combined_probabilities /= combined_probabilities.sum()
        else:
            # if all zeros, use uniform distribution
            combined_probabilities = np.ones_like(combined_probabilities) / len(
                combined_probabilities,
            )

        sample_probabilities = 1 - combined_probabilities
        # Sample indices based on combined probabilities
        indices = rng.choice(
            len(X),
            size=n_samples,
            replace=True,
            p=sample_probabilities,
        )

        # Return sampled data
        return X[indices], y[indices]
