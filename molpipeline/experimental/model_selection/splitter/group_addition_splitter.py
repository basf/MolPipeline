"""AddOneGroupSplit splitter implementation."""

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import BaseCrossValidator
from typing_extensions import override


class GroupAdditionSplit(BaseCrossValidator):  # pylint: disable=abstract-method
    """Add sequentially one group to the training set.

    The ordinal value of the groups is used to determine the order of addition. The
    parameter n_skip determines how many of the groups are part of the initial training
    set, where the group with the next higher value is used as the first test set. All
    other groups are not used in this split. After that, the group which formerly served
    as the test set is added to the training set, and the next higher group is used as
    the new test set. This process is repeated until all groups have been used as test
    set. The highest group is never part of the training set. Setting n_skip to 0 means
    that no training set is used for the first split, but since this is not a common use
    case, the default value for n_skip is 1. The parameter max_splits can be used to
    limit the number of splits from the end, meaning that only the last max_splits
    groups are used as test sets.

    To summarize the splitting strategy:
    - The groups are ordered by their ordinal value.
    - Groups ≤ n_skip: Always in training set
    - Highest group: Always in test set
    - Intermediate groups: Each serves once as test set, then joins training set
    - Training set grows incrementally with each split

    """

    n_skip: int
    max_splits: int | None

    def __init__(
        self,
        n_skip: int = 1,
        max_splits: int | None = None,
    ) -> None:
        """Initialize the AddOneGroupSplit.

        Parameters
        ----------
        n_skip : int, default=1
            Number of initial groups to skip as test sets.
            This means that the n groups are always part of the training set.
        max_splits : int | None, optional
            Maximum number of splits to create, by default None.
            If more splits are possible, only the last splits are returned.

        """
        self.n_skip = n_skip
        self.max_splits = max_splits

    @override
    def split(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        groups: npt.ArrayLike | None = None,
    ) -> Iterator[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        """Get the group split.

        Parameters
        ----------
        X : npt.ArrayLike
            The model input data.
        y : npt.ArrayLike, optional
            The target data, by default None.
        groups : npt.ArrayLike, optional
            The group data, by default None.


        Yields
        ------
        npt.NDArray[np.int64]
            The training indices.
        npt.NDArray[np.int64]
            The test indices.

        Raises
        ------
        ValueError
            If the groups parameter is not provided.
        ValueError
            If not enough groups are available to create any splits.

        """
        if groups is None:
            raise ValueError("The groups parameter is required.")

        unique_groups = sorted(np.unique(groups))
        n_skip = self.n_skip
        test_groups = unique_groups[n_skip:]
        # If max_splits is set, limit the number of splits from the end
        if self.max_splits is not None and len(test_groups) > self.max_splits:
            test_groups = test_groups[-self.max_splits :]
        if not test_groups:
            raise ValueError(
                "Not enough groups to create any splits with the given n_skip.",
            )

        for group in test_groups:
            train_idx = np.where(groups < group)[0]
            test_idx = np.where(groups == group)[0]
            yield train_idx, test_idx

    @override
    def get_n_splits(  # pylint: disable=signature-differs
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        groups: npt.ArrayLike | None = None,
    ) -> int:
        """Get the number of splits.

        Parameters
        ----------
        X : npt.ArrayLike
            The model input data.
        y : npt.ArrayLike, optional
            The target data, by default None.
        groups : npt.ArrayLike, optional
            The group data, by default None.

        Returns
        -------
        int
            The number of splits.

        Raises
        ------
        ValueError
            If the groups parameter is not provided.

        """
        if groups is None:
            raise ValueError("The groups parameter is required.")
        max_possible_splits = len(np.unique(groups)) - self.n_skip
        if self.max_splits is not None:
            return min(max_possible_splits, self.max_splits)
        return max_possible_splits
