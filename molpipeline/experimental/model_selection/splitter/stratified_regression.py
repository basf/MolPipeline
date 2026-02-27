"""Stratified K-Fold splitter for regression tasks."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn.model_selection import StratifiedKFold
from typing_extensions import deprecated, override


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

        y_binned = pd.qcut(y, n_groups, labels=False, duplicates="drop")

        yield from super().split(X, y_binned)


@deprecated(
    "Use the StratifiedRegressionKFold class directly instead of this helper function.",
)
def create_continuous_stratified_folds(
    y: npt.NDArray[Any],
    n_splits: int,
    n_groups: int = 10,
    random_state: int | None = None,
) -> list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
    """Create stratified folds for continuous targets using percentile-based binning.

    This method creates stratified cross-validation folds for regression by:
    1. Binning continuous targets into percentile-based groups
    2. Using stratified sampling to ensure balanced target distribution
    3. Returning train/validation index pairs

    Parameters
    ----------
    y : npt.NDArray[Any]
        Continuous target values to stratify.
    n_splits : int
        Number of cross-validation folds.
    n_groups : int, optional
        Number of percentile groups to create for stratification (default: 10).
    random_state : int | None, optional
        Random state for reproducibility.

    Returns
    -------
    list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]
        List of (train_indices, validation_indices) tuples for each fold.

    """
    splitter = PercentileStratifiedKFold(
        n_splits=n_splits,
        n_groups=n_groups,
        shuffle=True,
        random_state=random_state,
    )

    return list(splitter.split(X=np.zeros(len(y)), y=y))
