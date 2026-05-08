"""TimeThresholdSplitter implementation."""

from collections.abc import Iterator
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from molpipeline.experimental.model_selection.splitter.group_addition_splitter import (
    GroupAdditionSplit,
)
from molpipeline.utils.time_utils import (
    thresholds_for_n_years,
    timestamp_to_group,
)


class TimeThresholdSplitter(GroupAdditionSplit):  # pylint: disable=abstract-method
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
            Explicit list of time thresholds to partition the data into groups.
            Data points are assigned to groups based on which threshold they exceed.
        n_skip : int, default=1
            Number of initial groups to skip as test sets. These groups are always part
            of the training set.
        max_splits : int | None, optional
            Maximum number of splits to create, by default `None`. If more splits are
            possible, only the last ones are returned.

        Raises
        ------
        ValueError
            If `threshold_list` does not contain at least one element.

        """
        super().__init__(n_skip=n_skip, max_splits=max_splits)

        if len(threshold_list) == 0:
            raise ValueError("threshold_list must contain at least one timestamp.")

        self.threshold_list = sorted(threshold_list)

    @classmethod
    def from_final_threshold(
        cls,
        final_threshold: pd.Timestamp | str = "today",
        n_years: int = 5,
        splits_per_year: int = 1,
        n_skip: int = 1,
        max_splits: int | None = None,
        date_precision: str | None = "D",
    ) -> Self:
        """Initialize the TimeThresholdSplitter.

        Parameters
        ----------
        final_threshold : pd.Timestamp | str, optional
            The upper bound for generating thresholds when `threshold_list` is not
            provided. The str "today" uses the current timestamp, while "Q1"-"Q4" use
            the start of the respective quarter in the current year.
        n_years : int, default=5
            Number of years to create the splits for when constructing thresholds
            from `final_threshold`.
        splits_per_year : int, default=1
            Number of splits per year. Must be at least 1 if provided. Used only when
            `threshold_list` is None.
        n_skip : int, default=1
            Number of initial groups to skip as test sets. These groups are always part
            of the training set.
        max_splits : int | None, optional
            Maximum number of splits to create, by default `None`. If more splits are
            possible, only the last ones are returned.
        date_precision : str | None, default="D"
            The default "D" rounds to the beginning of the day.
            If `None`, no rounding is applied. Other options can be used as described in
            the pandas documentation [1].

        Returns
        -------
        TimeThresholdSplitter
            An instance of TimeThresholdSplitter initialized with a threshold_list
            generated from the provided parameters.

        References
        ----------
        [1] https://pandas.pydata.org/docs/user_guide/timeseries.html#period-aliases

        """
        threshold_list = thresholds_for_n_years(
            final_threshold=final_threshold,
            n_years=n_years,
            splits_per_year=splits_per_year,
            date_precision=date_precision,
        )
        return cls(n_skip=n_skip, max_splits=max_splits, threshold_list=threshold_list)

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
        group_indices = timestamp_to_group(groups, self.threshold_list)

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
        group_indices = timestamp_to_group(groups, self.threshold_list)

        # Use parent class get_n_splits method with converted groups
        return super().get_n_splits(X=X, y=y, groups=group_indices)
