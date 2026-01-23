"""AddOneGroupSplit splitter implementation."""

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt


class AddOneGroupSplit:
    """Add sequentially one group to the training set.

    The lowest group is always part of the training set,
    and the highest group is always part of the test set.

    """

    n_skip: int
    max_splits: int | None

    def __init__(
        self,
        n_skip: int = 0,
        max_splits: int | None = None,
    ) -> None:
        """Initialize the AddOneGroupSplit.

        Parameters
        ----------
        n_skip : int, optional
            Number of initial groups to skip, by default 0.
        max_splits : int | None, optional
            Maximum number of splits to create, by default None.
            If more splits are possible, only the last splits are returned.

        """
        self.n_skip = n_skip
        self.max_splits = max_splits

    def split(
        self,
        X: npt.ArrayLike,  # pylint:  # noqa: ARG002,N803
        y: npt.ArrayLike | None = None,  # noqa: ARG002
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

        """
        if groups is None:
            raise ValueError("The groups parameter is required.")

        unique_groups = sorted(np.unique(groups))
        n_skip = self.n_skip + 1  # First group is always in training set
        test_groups = unique_groups[n_skip:]
        if self.max_splits is not None and len(test_groups) > self.max_splits:
            test_groups = test_groups[-self.max_splits :]

        for group in test_groups:
            train_idx = np.where(groups < group)[0]
            test_idx = np.where(groups == group)[0]
            yield train_idx, test_idx

    def get_n_splits(
        self,
        X: npt.ArrayLike,  # noqa: ARG002,N803
        y: npt.ArrayLike | None = None,  # noqa: ARG002
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
        max_possible_splits = len(np.unique(groups)) - 1 - self.n_skip
        if self.max_splits is not None:
            return min(max_possible_splits, self.max_splits)
        return max_possible_splits
