"""Bootstrap split."""

from collections.abc import Generator
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse
from sklearn.model_selection import BaseCrossValidator
from typing_extensions import override

from molpipeline.utils.molpipeline_types import XType, YType


class BootstrapSplit(BaseCrossValidator):  # pylint: disable=abstract-method
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
        X: XType,
        y: YType = None,
        groups: YType = None,
    ) -> Generator[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]], Any, None]:
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
        n_samples = X.shape[0] if sparse.issparse(X) else len(np.asarray(X))
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_splits):
            train_indices = rng.choice(n_samples, size=n_samples, replace=True)
            test_indices = np.setdiff1d(np.arange(n_samples), train_indices)
            yield train_indices, test_indices

    @override
    def get_n_splits(  # type: ignore  # pylint: disable=signature-differs
        self,
        X: XType,
        y: YType = None,
        groups: YType | None = None,
    ) -> int:
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
