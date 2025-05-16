"""Stochastic filters for providing filter probabilities for sampling."""

from abc import ABC, abstractmethod
from typing import override

import numpy as np
import numpy.typing as npt


class StochasticFilter(ABC):  # pylint: disable=too-few-public-methods
    """Abstract base class for stochastic filter policies.

    A StochasticFilter assigns sampling probabilities to data points based on
    implementation-defined criteria.
    """

    @abstractmethod
    def calculate_probabilities(
        self,
        X: npt.NDArray[np.float64],  # noqa: N803  # pylint: disable=invalid-name
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculate probability for each sample.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            The input samples of shape (n_samples, n_features).
        y : npt.NDArray[np.float64]
            The target values of shape (n_samples,).

        Returns
        -------
        probabilities : npt.NDArray[np.float64]
            Probability for each sample of shape (n_samples,)

        """


class GlobalClassBalanceFilter(
    StochasticFilter,
):  # pylint: disable=too-few-public-methods
    """Provides probabilities inversely proportional to global class frequencies.

    Samples from minority classes receive higher probability values.
    """

    @override
    def calculate_probabilities(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculate probability based on inverse class frequencies.

        Note that probabilities over the sample list do not sum to 1.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            The input samples of shape (n_samples, n_features).
        y : npt.NDArray[np.float64]
            The target values of shape (n_samples,).

        Returns
        -------
        probabilities : npt.NDArray[np.float64]
            Probability for each sample of shape (n_samples,).

        """
        _, class_indices, counts = np.unique(
            y,
            return_inverse=True,
            return_counts=True,
        )

        # calculate inverse class frequencies
        # could also use power functions like inv_freqs = (1.0 / counts) ** p to
        # control the penalization with the parameter p.
        inv_freqs = 1.0 / counts

        # normalize to probability distribution
        normalized_inv_freqs = inv_freqs / inv_freqs.sum()

        # return probabilities in order of samples in the input
        return normalized_inv_freqs[class_indices]


class LocalGroupClassBalanceFilter(
    StochasticFilter,
):  # pylint: disable=too-few-public-methods
    """Filter that returns probabilities based on class balance within groups.

    Samples from minority classes within their group receive higher probabilities.
    """

    def __init__(self, groups: npt.NDArray[np.float64]) -> None:
        """Create a new LocalGroupClassBalanceFilter.

        Parameters
        ----------
        groups : npt.NDArray[np.float64]
            Group labels for the samples of shape (n_samples,).

        """
        self._n_samples = groups.shape[0]
        self._unique_groups, self._group_indices = np.unique(
            groups,
            return_inverse=True,
        )

    @override
    def calculate_probabilities(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculate probability based on inverse class frequencies within groups.

        Note that probabilities over the sample list do not sum to 1.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            The input samples of shape (n_samples, n_features).
        y : npt.NDArray[np.float64]
            The target values  of shape (n_samples,).

        Returns
        -------
        probabilities : npt.NDArray[np.float64]
            Probability for each sample of shape (n_samples,).

        Raises
        ------
        ValueError
            If the length of y does not match the length of groups.

        """
        if self._n_samples != len(y):
            raise ValueError("Provided y must have the same length as groups")

        n_samples = self._n_samples
        unique_groups = self._unique_groups
        group_indices = self._group_indices

        probabilities = np.zeros(n_samples)

        prob_sum = 0
        for group_idx, _ in enumerate(unique_groups):
            # Find samples belonging to current group
            group_mask = group_indices == group_idx

            # get classes for this group and their counts
            group_y = y[group_mask]
            _, group_class_indices, group_counts = np.unique(
                group_y,
                return_inverse=True,
                return_counts=True,
            )

            # calculate normalized inverse frequencies
            group_inv_freqs = 1.0 / group_counts

            # map each sample to its corresponding inverse frequency
            probabilities[group_mask] = group_inv_freqs[group_class_indices]

            # we need to sum the inverse frequencies for normalization
            prob_sum += group_inv_freqs.sum()

        # Normalize the final probabilities. We normalize over the inverse frequencies
        # over all groups. This sets the scale between groups and classes. Groups with
        # fewer samples of a particular class will get higher probabilities than samples
        # with many samples of that class.
        if prob_sum > 0:
            probabilities /= prob_sum
        else:
            probabilities = np.ones(n_samples) / n_samples

        return probabilities


class GroupSizeFilter(StochasticFilter):  # pylint: disable=too-few-public-methods
    """Filter that returns probabilities inversely proportional to group sizes.

    Samples from smaller groups receive higher probabilities.
    """

    def __init__(self, groups: npt.NDArray[np.float64]) -> None:
        """Create a new GroupSizeFilter.

        Parameters
        ----------
        groups : npt.NDArray[np.float64]
            Group labels for the samples of shape (n_samples,).

        """
        self.probabilities = self._compute_group_probs(groups)

    @staticmethod
    def _compute_group_probs(
        groups: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute inverse group sizes and normalize them.

        Note that probabilities over the sample list do not sum to 1.

        Parameters
        ----------
        groups : npt.NDArray[np.float64]
            Group labels for the samples of shape (n_samples,).

        Returns
        -------
        probabilities : npt.NDArray[np.float64]
            Probability for each sample of shape (n_samples,).

        """
        _, group_indices, group_counts = np.unique(
            groups,
            return_counts=True,
            return_inverse=True,
        )
        inv_sizes = 1.0 / group_counts
        normalized_inv_sizes = inv_sizes / inv_sizes.sum()
        return normalized_inv_sizes[group_indices]

    @override
    def calculate_probabilities(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculate probability based on inverse group sizes.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            The input samples of shape (n_samples, n_features).
        y : npt.NDArray[np.float64]
            The target values  of shape (n_samples,).

        Returns
        -------
        probabilities : array-like of shape (n_samples,)
            Probability for each sample.

        Raises
        ------
        ValueError
            If the length of y does not match the length of groups.

        """
        if len(self.probabilities) != len(y):
            raise ValueError("Provided y must be the same length as groups")
        return self.probabilities
