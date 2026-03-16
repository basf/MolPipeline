"""Bootstrap split."""

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import BaseCrossValidator
from typing_extensions import override


class BootstrapSplit(BaseCrossValidator):
    """Splitter where the training set is a bootstrap sample."""

    def __init__(self, n_splits: int, random_state: int | None = None) -> None:
        """Initialize the bootstrap split.

        Parameters
        ----------
        n_splits : int
            Number of splits to create.
        random_state : int | None, optional
            Random state to use.

        """
        self.n_splits = n_splits
        self.random_state = random_state

    @override
    def split(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        groups: npt.ArrayLike | None = None,
    ) -> Iterator[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        """Get the bootstrap split.

        Parameters
        ----------
        X : array-like
            The input data.
        y : array-like, optional
            The target values, by default None.
        groups : array-like, optional
            The group labels for the samples used while splitting the dataset into
            train/test set. Default is None.

        Yields
        ------
        tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]
            The training indices and test indices for each split.

        """
        n_samples = len(np.asarray(X))
        rng = np.random.RandomState(self.random_state)
        for _ in range(self.n_splits):
            train_indices = rng.choice(n_samples, size=n_samples, replace=True)
            test_indices = np.setdiff1d(np.arange(n_samples), train_indices)
            yield train_indices, test_indices

    @override
    def get_n_splits(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        groups: npt.ArrayLike | None = None,
    ) -> int:  # type: ignore
        """Get the number of splits.

        Parameters
        ----------
        X : array-like
            The input data.
        y : array-like, optional
            The target values, by default None.
        groups : array-like, optional
            The group labels for the samples used while splitting the dataset into
            train/test set. Default is None.

        Returns
        -------
        int
            The number of splits.

        """
        return self.n_splits
