"""TimeThresholdSplitter implementation."""

from collections.abc import Iterator
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from molpipeline.experimental.model_selection.splitter.group_addition_splitter import (
    GroupAdditionSplit,
)
from molpipeline.utils.time_utils import (
    NamedTimeStamps,
    resolve_named_time_stamps,
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
        threshold_list: list[pd.Timestamp] | None = None,
        *,
        # Parameters for time-based construction when threshold_list is not provided
        final_threshold: pd.Timestamp | NamedTimeStamps | None = None,
        n_years: int = 5,
        splits_per_year: int = 1,
        round_to: str | None = "d",
        # Generic AddOneGroupSplit parameters
        n_skip: int = 1,
        max_splits: int | None = None,
    ) -> None:
        """Initialize the TimeThresholdSplitter.

        Parameters
        ----------
        threshold_list : list[pd.Timestamp] | None, optional
            Explicit list of time thresholds to partition the data into groups.
            Data points are assigned to groups based on which threshold they exceed.
            If None, thresholds are constructed from ``final_threshold`` and
            related parameters.
        final_threshold : pandas.Timestamp or {"now", "Q1", "Q2", "Q3", "Q4"}, optional
            The upper bound for generating thresholds when ``threshold_list`` is
            not provided. ``"now"`` uses the current timestamp. ``"Q1"``-""Q4"``
            use the start of the respective quarter in the current year.
        n_years : int, default=5
            Number of years to create the splits for when constructing thresholds
            from ``final_threshold``.
        splits_per_year : int, default=1
            Number of splits per year. Must be at least 1 if provided. Used only
            when ``threshold_list`` is None.
        round_to : str | None, default="d"
            Rounding precision for generated thresholds when ``threshold_list`` is
            None. For options please refer to pandas.Timestamp.round.html [1].
        n_skip : int, default=1
            Number of initial groups to skip as test sets. These groups are
            always part of the training set.
        max_splits : int | None, optional
            Maximum number of splits to create, by default ``None``. If more
            splits are possible, only the last ones are returned.

        Raises
        ------
        ValueError
            If both ``threshold_list`` and ``final_threshold`` parameters are
            provided.
        ValueError
            If neither ``threshold_list`` nor ``final_threshold`` are
            provided.
        ValueError
            If ``threshold_list`` is empty after resolution.
        ValueError
            If ``splits_per_year`` is provided and is less than 1.

        References
        ----------
        [1] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.round.html

        """
        super().__init__(n_skip=n_skip, max_splits=max_splits)

        if threshold_list is not None and final_threshold is not None:
            raise ValueError(
                "Provide either 'threshold_list' or 'final_threshold', not both.",
            )

        if threshold_list is None:
            if final_threshold is None:
                raise ValueError(
                    "Either 'threshold_list' must be provided or "
                    "'final_threshold' must be specified to generate thresholds.",
                )
            threshold_list = self._build_thresholds_from_years(
                splits_per_year=splits_per_year,
                final_threshold=final_threshold,
                n_years=n_years,
                round_to=round_to,
            )

        if len(threshold_list) == 0:
            raise ValueError("threshold_list must contain at least one timestamp.")

        self.threshold_list = sorted(threshold_list)

    @staticmethod
    def _build_thresholds_from_years(
        *,
        splits_per_year: int,
        final_threshold: pd.Timestamp | NamedTimeStamps,
        n_years: int,
        round_to: Literal["day", "month", "hour"] | None,
    ) -> list[pd.Timestamp]:
        """Construct a threshold list from year-based configuration.

        Parameters
        ----------
        splits_per_year : int
            Number of splits per year. Must be at least 1.
        final_threshold : pandas.Timestamp or {"now", "today", "Q1", "Q2", "Q3", "Q4"}
            The upper bound for the generated thresholds.
        n_years : int
            Number of years to create the splits for.
        round_to : {"day", "month", "hour"} or None
            Rounding precision for threshold timestamps.

        Returns
        -------
        list[pd.Timestamp]
            A list of constructed threshold timestamps.

        Raises
        ------
        ValueError
            If required parameters are missing or invalid.

        """
        if splits_per_year < 1:
            raise ValueError("splits_per_year must be at least 1.")

        resolved_final = resolve_named_time_stamps(final_threshold)
        constructed_thresholds: list[pd.Timestamp] = []

        time_delta = pd.Timedelta(days=365.2425 / splits_per_year)

        # We go backwards in time from the resolved final threshold over n_years
        for year_offset in range(n_years):
            year_start = pd.Timestamp(
                year=resolved_final.year - year_offset,
                month=resolved_final.month,
                day=resolved_final.day,
                hour=resolved_final.hour,
                minute=resolved_final.minute,
                second=resolved_final.second,
                microsecond=resolved_final.microsecond,
            )
            for split in range(splits_per_year):
                threshold = year_start - split * time_delta
                threshold = threshold.round(round_to) if round_to else threshold
                constructed_thresholds.append(threshold)

        return constructed_thresholds

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
