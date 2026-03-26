"""Stratified K-Fold splitter for regression tasks."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn.model_selection import StratifiedKFold
from typing_extensions import override


class PercentileStratifiedKFold(StratifiedKFold):  # pylint: disable=abstract-method
    """Stratified K-Fold splitter for regression tasks using percentile-based binning.

    This splitter bins continuous target values into percentiles, which by definition
    are equally sized groups, and then applies stratified sampling to ensure that each
    fold has a balanced distribution of target values across the percentiles.

    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        n_groups: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize the PercentileStratifiedRegressionKFold.

        Parameters
        ----------
        n_splits : int, default=5
            Number of folds for cross-validation.
        n_groups : int, default=10
            Number of percentile groups to create for stratification.
        shuffle : bool, default=True
            Whether to shuffle the data before splitting into batches.
        random_state : int | None, optional
            Random state for reproducibility.

        """
        self.n_groups = n_groups
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    @override
    def split(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        groups: npt.ArrayLike | None = None,
    ) -> Iterator[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : npt.ArrayLike
            The model input data.
        y : npt.ArrayLike
            The target data.
        groups : npt.ArrayLike | None, optional
            The group data, by default None.

        Yields
        ------
        npt.NDArray[np.int_]
            The training indices.
        npt.NDArray[np.int_]
            The test indices.

        Raises
        ------
        ValueError
            If n_groups is greater than the number of unique target values.

        """
        n_groups = self.n_groups
        y = np.asarray(y, dtype=np.float64)
        if self.n_groups > len(np.unique(y)):
            raise ValueError(
                f"n_groups ({self.n_groups}) is greater than the number of unique "
                f"target values ({len(np.unique(y))})!",
            )

        try:
            y_binned = pd.qcut(y, n_groups, labels=False, duplicates="raise")
        except ValueError as e:
            raise ValueError(
                "Too many identical values in y for the specified number of groups!",
            ) from e

        yield from super().split(X, y_binned)
