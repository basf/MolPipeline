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

    def __init__(
        self,
        n_splits: int,
        max_samples: float | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize the bootstrap split.

        Parameters
        ----------
        n_splits : int
            Number of splits to create.
        max_samples: int | float | None, optional
            The number of samples to draw for each split.
            If int, then max_samples defines the exact number of samples to draw.
            If float, then max_samples defines the proportion of samples to draw.
            If None, all samples are drawn.
        random_state : int | None, optional
            Random state to use.

        Raises
        ------
        ValueError
            If max_samples is a float and not in the range (0.0, 1.0].

        """
        self.n_splits = n_splits
        if isinstance(max_samples, float) and not 0.0 < max_samples <= 1.0:
            raise ValueError(
                "If max_samples is a float, it must be in the range (0.0, 1.0].",
            )
        self.max_samples = max_samples
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
        if isinstance(self.max_samples, int):
            n_draw = min(self.max_samples, n_samples)
        elif isinstance(self.max_samples, float):
            n_draw = int(n_samples * self.max_samples)
        else:
            n_draw = n_samples
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_splits):
            train_indices = rng.choice(n_samples, size=n_draw, replace=True)
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
