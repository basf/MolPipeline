"""The DataRepetitionSplit returns only the training set indices.

This is not meant to be used for model validation but for ensembling purposes.

"""

from collections.abc import Generator
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse
from sklearn.model_selection import BaseCrossValidator
from typing_extensions import override


class DataRepetitionSplit(BaseCrossValidator):  # pylint: disable=abstract-method
    """The DataRepetitionSplit returns only the training set indices.

    This is not meant to be used for model validation but for ensembling purposes.

    """

    def __init__(self, n_splits: int) -> None:
        """Initialize the DataRepetitionSplit.

        Parameters
        ----------
        n_splits : int
            Number of splits to create.

        """
        self.n_splits = n_splits

    @override
    def split(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        groups: npt.ArrayLike | None = None,
    ) -> Generator[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]], Any, None]:
        """Get the data repetition split.

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
        for _ in range(self.n_splits):
            train_indices = np.arange(n_samples)
            test_indices = np.array([], dtype=np.int64)
            yield train_indices, test_indices

    @override
    def get_n_splits(  # type: ignore  # pylint: disable=signature-differs
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        groups: npt.ArrayLike | None = None,
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
