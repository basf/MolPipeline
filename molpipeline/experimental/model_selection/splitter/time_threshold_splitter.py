"""TimeThresholdSplitter implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from molpipeline.experimental.model_selection.splitter.add_one_group_split import (
    AddOneGroupSplit,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self


class TimeThresholdSplitter(AddOneGroupSplit):
    """Split data based on time thresholds using AddOneGroupSplit strategy.

    This splitter converts time-based data into group indices and applies
    the AddOneGroupSplit strategy. Time data is partitioned into groups
    based on threshold timestamps.

    Splitting strategy:
    - Groups ≤ n_skip: Always in training set
    - Highest group: Always in test set
    - Intermediate groups: Each serves once as test set, then joins training set
    - Training set grows incrementally with each split

    """

    threshold_list: list[pd.Timestamp]

    def __init__(
        self,
        threshold_list: list[pd.Timestamp],
        n_skip: int = 1,
        max_splits: int | None = None,
    ) -> None:
        """Initialize the TimeThresholdSplitter.

        Parameters
        ----------
        threshold_list : list[pd.Timestamp]
            List of time thresholds to partition the data into groups.
            Data points are assigned to groups based on which threshold
            they exceed.
        n_skip : int, default=1
            Number of initial groups to skip as test sets.
            These groups are always part of the training set.
        max_splits : int | None, optional
            Maximum number of splits to create, by default None.
            If more splits are possible, only the last splits are returned.

        Raises
        ------
        ValueError
            If threshold_list is empty.

        """
        super().__init__(n_skip=n_skip, max_splits=max_splits)
        if len(threshold_list) == 0:
            raise ValueError("threshold_list must contain at least one timestamp.")
        self.threshold_list = sorted(threshold_list)

    def _convert_time_to_groups(
        self,
        groups: npt.ArrayLike,
    ) -> npt.NDArray[np.int64]:
        """Convert time data to group indices.

        Parameters
        ----------
        groups : npt.ArrayLike
            Time data to convert to group indices. Should be datetime-like values.

        Returns
        -------
        npt.NDArray[np.int64]
            Group indices for each data point.

        """
        # Convert to pandas Series if it's not already to handle comparisons uniformly
        if not isinstance(groups, pd.Series):
            groups = pd.Series(groups)  # type: ignore[call-overload]

        split_index = np.zeros(len(groups), dtype=np.int64)
        for threshold in self.threshold_list:
            split_index[groups > threshold] += 1
        return split_index

    def split(
        self,
        X: npt.ArrayLike,  # noqa: N803
        y: npt.ArrayLike | None = None,
        groups: npt.ArrayLike | None = None,
    ) -> Iterator[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        """Get the time-based split.

        Parameters
        ----------
        X : npt.ArrayLike
            The model input data.
        y : npt.ArrayLike, optional
            The target data, by default None.
        groups : npt.ArrayLike, optional
            The time data for each sample. Should be numpy datetime64 array
            or pandas Series with datetime values, by default None.

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

        # Convert time data to group indices
        group_indices = self._convert_time_to_groups(groups)

        # Use parent class split method with converted groups
        yield from super().split(X=X, y=y, groups=group_indices)

    def get_n_splits(
        self,
        X: npt.ArrayLike,  # noqa: N803
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
            The time data for each sample. Should be numpy datetime64 array
            or pandas Series with datetime values, by default None.

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

        # Convert time data to group indices
        group_indices = self._convert_time_to_groups(groups)

        # Use parent class get_n_splits method with converted groups
        return super().get_n_splits(X=X, y=y, groups=group_indices)

    @classmethod
    def from_splits_per_year(
        cls,
        splits_per_year: int,
        last_year: int,
        n_years: int = 5,
        n_skip: int = 1,
        max_splits: int | None = None,
    ) -> Self:
        """Create a TimeThresholdSplitter from the number of splits per year.

        Parameters
        ----------
        splits_per_year : int
            Number of splits per year. Must be between 1 and 12.
        last_year : int
            The end year for the splits.
        n_years : int, optional
            Number of years to create the splits for, by default 5.
        n_skip : int, default=1
            Number of initial groups to skip as test sets.
        max_splits : int | None, optional
            Maximum number of splits to create, by default None.

        Returns
        -------
        Self
            The TimeThresholdSplitter instance.

        Raises
        ------
        ValueError
            If splits_per_year is less than 1.

        """
        if splits_per_year < 1:
            raise ValueError("splits_per_year must be at least 1.")

        threshold_list = []
        time_delta = pd.Timedelta(days=365.25 / splits_per_year)

        for year in range(n_years):
            year_start = pd.Timestamp(year=last_year - year, month=1, day=1)
            threshold_list.extend(
                [year_start + split * time_delta for split in range(splits_per_year)],
            )

        return cls(
            threshold_list=sorted(threshold_list),
            n_skip=n_skip,
            max_splits=max_splits,
        )
