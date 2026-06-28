"""Oversampler that selects samples based on group sizes."""

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_X_y

from molpipeline.estimators.samplers.stochastic_filter import GroupSizeFilter


class GroupRandomOversampler(BaseEstimator, TransformerMixin):
    """Random oversampler that samples proportional to inverse group sizes.

    The oversampling strategy picks minority class samples with a probability inverse
    to size of the group they are in. With this strategy minority samples from smaller
    groups are more often oversampled. The number of samples picked is the difference
    between the number of majority class and minority class samples, such that the
    returned data sets is balanced, i.e. has the  same number of samples for each class.
    Note that, since only minority class samples of the groups are oversampled, this
    strategy can lead to a class imbalance within groups.

    """

    # Currently, only binary classification is supported.
    _EXPECTED_NUMBER_OF_CLASSES = 2

    def __init__(self, random_state: int | None = None) -> None:
        """Create new GroupRandomOversampler.

        Parameters
        ----------
        random_state : int, RandomState instance or None, default=None
            Controls the randomization of the algorithm.

        """
        self.random_state = random_state

    @staticmethod
    def _calculate_probabilities(
        X: npt.NDArray[np.float64],  # noqa: N803 # pylint: disable=invalid-name
        y: npt.NDArray[np.float64],
        groups: npt.NDArray[np.float64],
        minority_class: int,
    ) -> npt.NDArray[np.float64]:
        """Calculate sampling probabilities.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Training data. Will be ignored, only present because of compatibility.
        y : npt.NDArray[np.float64]
            Target values. Must be binary classification labels.
        groups : npt.NDArray[np.float64]
            Group labels for the samples.
        minority_class : int
            The minority class label.

        Returns
        -------
        npt.NDArray[np.float64]
            Sampling probabilities for each sample.

        """
        filter_policy = GroupSizeFilter(groups)
        probs = filter_policy.calculate_probabilities(X, y)
        # set probability of majority class(es) to zero
        probs[y != minority_class] = 0

        # normalize probabilities over all samples
        probs /= np.sum(probs, dtype=np.float64)

        return probs

    def transform(
        self,
        X: npt.NDArray[np.float64],  # noqa: N803 # pylint: disable=invalid-name
        y: npt.NDArray[np.float64],
        groups: npt.NDArray[np.float64],
        return_groups: bool = False,
    ) -> (
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        | tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ]
    ):
        """Transform X and y to oversampled version of the data set.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Training data. Will be ignored, only present because of compatibility.
        y : npt.NDArray[np.float64]
            Target values. Must be binary classification labels.
        groups : npt.NDArray[np.float64]
            Group labels for the samples.
        return_groups : bool, default=False
            If True, return the groups of the resampled data set.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        | tuple[
            npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
        ]
            Resampled data set (X, y) or (X, y, groups) if return_groups is True.

        Raises
        ------
        ValueError
            If the length of y does not match the length of groups.
            If y is not binary classification labels.

        """
        _, y = check_X_y(X, y, accept_sparse=["csr", "csc"])
        rng = check_random_state(self.random_state)

        if len(groups) != y.shape[0]:
            raise ValueError(
                f"Found groups with size {groups.shape[0]}, but expected {y.shape[0]}",
            )

        u_y, u_y_sizes = np.unique(
            y,
            return_counts=True,
        )

        if len(u_y) != self._EXPECTED_NUMBER_OF_CLASSES:
            raise ValueError("Only support oversampling for binary classification.")

        diff = abs(u_y_sizes[0] - u_y_sizes[1])
        minority_class = u_y[np.argmin(u_y_sizes)]

        probs = self._calculate_probabilities(X, y, groups, minority_class)

        # sample indices
        sampled_indices = rng.choice(
            len(X),
            size=diff,
            replace=True,
            p=probs,
        )

        x_resampled = np.empty((X.shape[0] + diff, X.shape[1]), dtype=X.dtype)
        x_resampled[: X.shape[0]] = X
        x_resampled[X.shape[0] :] = X[sampled_indices]

        y_resampled = np.empty(y.shape[0] + diff, dtype=y.dtype)
        y_resampled[: y.shape[0]] = y
        y_resampled[y.shape[0] :] = y[sampled_indices]

        if return_groups:
            groups_resampled = np.empty(groups.shape[0] + diff, dtype=groups.dtype)
            groups_resampled[: groups.shape[0]] = groups
            groups_resampled[groups.shape[0] :] = groups[sampled_indices]
            return x_resampled, y_resampled, groups_resampled

        return x_resampled, y_resampled
