"""TimeThresholdSplitter implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from molpipeline.experimental.model_selection.splitter.add_one_group_split import (
    AddOneGroupSplit,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


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
        threshold_list: list[pd.Timestamp] | None = None,
        *,
        # Parameters for time-based construction when threshold_list is not provided
        last_year: int | None = None,
        n_years: int = 5,
        splits_per_year: int = 1,
        round_to: Literal["day", "month", "hour"] | None = "day",
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
            If None, thresholds are constructed from `splits_per_year` and
            related parameters.
        last_year : int | None, optional
            The end year for the splits. Used only when `threshold_list` is
            None.
        n_years : int, default=5
            Number of years to create the splits for when constructing thresholds
            from `splits_per_year`.
        splits_per_year : int, default=1
            Number of splits per year. Must be at least 1 if provided. Used only
            when `threshold_list` is None.
        round_to : {"day", "month", "hour"} or None, default="day"
            Rounding precision for threshold timestamps when constructing them
            from ``splits_per_year``. Options: "day" (midnight), "month"
            (start of month), "hour" (start of hour). If ``None``, keep exact
            fractional timestamps.
        n_skip : int, default=1
            Number of initial groups to skip as test sets. These groups are
            always part of the training set.
        max_splits : int | None, optional
            Maximum number of splits to create, by default ``None``. If more
            splits are possible, only the last ones are returned.

        Raises
        ------
        ValueError
            If both ``threshold_list`` and ``splits_per_year``/``last_year``
            parameters are provided.
        ValueError
            If neither ``threshold_list`` nor the per-year parameters are
            provided.
        ValueError
            If ``threshold_list`` is empty after resolution.
        ValueError
            If ``splits_per_year`` is provided and is less than 1.

        """
        super().__init__(n_skip=n_skip, max_splits=max_splits)

        if threshold_list is not None and last_year is not None:
            raise ValueError(
                "Provide either 'threshold_list' or 'last_year', not both.",
            )

        if threshold_list is None:
            if last_year is None:
                raise ValueError(
                    "Either 'threshold_list' must be provided or "
                    "'last_year' must be specified to generate thresholds.",
                )
            threshold_list = self._build_thresholds_from_years(
                splits_per_year=splits_per_year,
                last_year=last_year,
                n_years=n_years,
                round_to=round_to,
            )

        if len(threshold_list) == 0:
            raise ValueError("threshold_list must contain at least one timestamp.")

        self.threshold_list = sorted(threshold_list)

    def _build_thresholds_from_years(
        self,
        *,
        splits_per_year: int,
        last_year: int,
        n_years: int,
        round_to: Literal["day", "month", "hour"] | None,
    ) -> list[pd.Timestamp]:
        """Construct a threshold list from year-based configuration.

        Parameters
        ----------
        splits_per_year : int
            Number of splits per year. Must be at least 1.
        last_year : int
            The end year for the splits.
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

        constructed_thresholds: list[pd.Timestamp] = []
        time_delta = pd.Timedelta(days=365.25 / splits_per_year)

        for year in range(n_years):
            year_start = pd.Timestamp(year=last_year - year, month=1, day=1)
            for split in range(splits_per_year):
                threshold = year_start + split * time_delta
                threshold = self._round_threshold(threshold, round_to)
                constructed_thresholds.append(threshold)

        return constructed_thresholds

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

    @staticmethod
    def _round_threshold(
        threshold: pd.Timestamp,
        round_to: Literal["day", "month", "hour"] | None,
    ) -> pd.Timestamp:
        """Round a timestamp to the requested precision.

        Parameters
        ----------
        threshold : pd.Timestamp
            Timestamp to round.
        round_to : {"day", "month", "hour"} or None
            Target precision. If ``None``, the timestamp is returned unchanged.

        Returns
        -------
        pd.Timestamp
            Rounded timestamp.

        Raises
        ------
        ValueError
            If round_to is not one of the valid options.

        """
        if round_to is not None and round_to not in {"day", "month", "hour"}:
            raise ValueError(
                f"round_to must be 'day', 'month', 'hour', or None, got '{round_to}'",
            )

        if round_to == "day":
            return threshold.normalize()
        if round_to == "hour":
            return threshold.floor("h")
        if round_to == "month":
            return pd.Timestamp(
                year=threshold.year,
                month=threshold.month,
                day=1,
            )
        return threshold
