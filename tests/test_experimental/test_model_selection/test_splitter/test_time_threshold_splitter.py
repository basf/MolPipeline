"""Unit tests for TimeThresholdSplitter."""

import unittest

import numpy as np
import numpy.typing as npt
import pandas as pd

from molpipeline.experimental.model_selection.splitter.time_threshold_splitter import (
    TimeThresholdSplitter,
)


class TestTimeThresholdSplitter(unittest.TestCase):  # noqa: PLR0904
    """Tests for TimeThresholdSplitter."""

    def _assert_splits_equal(  # pylint: disable=duplicate-code
        self,
        actual_splits: list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]],
        expected_splits: list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]],
    ) -> None:
        """Assert that actual splits match expected splits.

        Parameters
        ----------
        actual_splits : list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]
            The actual train/test split indices.
        expected_splits : list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]
            The expected train/test split indices.

        """
        self.assertEqual(len(actual_splits), len(expected_splits))
        for i, (actual, expected) in enumerate(
            zip(actual_splits, expected_splits, strict=True),
        ):
            self.assertTrue(
                np.array_equal(actual[0], expected[0]),
                f"Train indices mismatch at split {i}",
            )
            self.assertTrue(
                np.array_equal(actual[1], expected[1]),
                f"Test indices mismatch at split {i}",
            )

    def test_split_raises_without_groups(self) -> None:
        """Ensure split raises when groups are missing."""
        threshold_list = [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")]
        splitter = TimeThresholdSplitter(threshold_list=threshold_list)
        features = np.ones(3)
        with self.assertRaisesRegex(ValueError, "The groups parameter is required."):
            next(splitter.split(X=features, y=None, groups=None))

    def test_get_n_splits_raises_without_groups(self) -> None:
        """Ensure get_n_splits raises when groups are missing."""
        threshold_list = [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")]
        splitter = TimeThresholdSplitter(threshold_list=threshold_list)
        with self.assertRaisesRegex(ValueError, "The groups parameter is required."):
            splitter.get_n_splits(X=np.ones(2), groups=None)

    def test_empty_threshold_list(self) -> None:
        """Test that an empty threshold list raises ValueError."""
        with self.assertRaisesRegex(
            ValueError,
            "threshold_list must contain at least one",
        ):
            TimeThresholdSplitter(threshold_list=[])

    def test_generates_expected_splits(self) -> None:
        """Verify splitter yields cumulative train groups by default."""
        threshold_list = [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01")]
        splitter = TimeThresholdSplitter(threshold_list=threshold_list)

        # Create time data spanning three groups
        time_data = pd.Series(
            [
                pd.Timestamp("2020-01-01"),  # Group 0
                pd.Timestamp("2020-03-01"),  # Group 0
                pd.Timestamp("2020-09-01"),  # Group 1
                pd.Timestamp("2020-12-01"),  # Group 1
                pd.Timestamp("2021-09-01"),  # Group 2
                pd.Timestamp("2022-01-01"),  # Group 2
            ],
        )

        features = np.ones(len(time_data))
        splits = list(splitter.split(X=features, groups=time_data))

        expected = [
            # Train: Group 0, Test: Group 1
            (np.array([0, 1]), np.array([2, 3])),
            # Train: Groups 0,1, Test: Group 2
            (np.array([0, 1, 2, 3]), np.array([4, 5])),
        ]

        self._assert_splits_equal(splits, expected)

    def test_n_skip(self) -> None:
        """Verify initial splits can be skipped via n_skip."""
        threshold_list = [
            pd.Timestamp("2020-06-01"),
            pd.Timestamp("2021-06-01"),
            pd.Timestamp("2022-06-01"),
        ]
        splitter = TimeThresholdSplitter(threshold_list=threshold_list, n_skip=2)

        time_data = pd.Series(
            [
                pd.Timestamp("2020-01-01"),  # Group 0
                pd.Timestamp("2020-03-01"),  # Group 0
                pd.Timestamp("2020-09-01"),  # Group 1
                pd.Timestamp("2020-12-01"),  # Group 1
                pd.Timestamp("2021-09-01"),  # Group 2
                pd.Timestamp("2022-01-01"),  # Group 2
                pd.Timestamp("2022-09-01"),  # Group 3
                pd.Timestamp("2023-01-01"),  # Group 3
            ],
        )

        features = np.ones(len(time_data))
        splits = list(splitter.split(X=features, groups=time_data))

        # Skip group 0 and 1, start test from group 2
        # Split 1: Train: Groups 0,1, Test: Group 2
        # Split 2: Train: Groups 0,1,2, Test: Group 3
        expected = [
            (np.array([0, 1, 2, 3]), np.array([4, 5])),
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7])),
        ]

        self._assert_splits_equal(splits, expected)

    def test_applies_max_splits_from_end(self) -> None:
        """Ensure max_splits limits the number of yielded splits."""
        threshold_list = [
            pd.Timestamp("2020-06-01"),
            pd.Timestamp("2021-06-01"),
            pd.Timestamp("2022-06-01"),
        ]
        splitter = TimeThresholdSplitter(threshold_list=threshold_list, max_splits=1)

        time_data = pd.Series(
            [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-03-01"),
                pd.Timestamp("2020-09-01"),
                pd.Timestamp("2020-12-01"),
                pd.Timestamp("2021-09-01"),
                pd.Timestamp("2022-01-01"),
                pd.Timestamp("2022-09-01"),
                pd.Timestamp("2023-01-01"),
            ],
        )

        features = np.ones(len(time_data))
        splits = list(splitter.split(X=features, groups=time_data))

        # Train: Groups 0,1,2, Test: Group 3
        expected = [
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7])),
        ]

        self._assert_splits_equal(splits, expected)

    def test_get_n_splits(self) -> None:
        """Check get_n_splits returns correct number of splits."""
        threshold_list = [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01")]
        splitter = TimeThresholdSplitter(threshold_list=threshold_list)

        time_data = pd.Series(
            [
                pd.Timestamp("2020-01-01"),  # Group 0
                pd.Timestamp("2020-09-01"),  # Group 1
                pd.Timestamp("2021-09-01"),  # Group 2
            ],
        )

        n_splits = splitter.get_n_splits(X=np.ones(len(time_data)), groups=time_data)
        # 3 groups created (0, 1, 2), with n_skip=0, all 3 serve as test sets
        self.assertEqual(n_splits, 2)

    def test_all_data_in_same_group(self) -> None:
        """Test when all data falls into the same group."""
        threshold_list = [pd.Timestamp("2025-01-01")]
        splitter = TimeThresholdSplitter(threshold_list=threshold_list)

        time_data = pd.Series(
            [
                pd.Timestamp("2020-01-01"),  # Group 0
                pd.Timestamp("2021-01-01"),  # Group 0
                pd.Timestamp("2022-01-01"),  # Group 0
            ],
        )

        features = np.ones(len(time_data))

        # Only one group, resulting in zero splits, raise ValueError on split attempt
        n_splits = splitter.get_n_splits(X=features, groups=time_data)
        self.assertEqual(n_splits, 0)
        with self.assertRaisesRegex(ValueError, "Not enough groups to create "):
            list(splitter.split(X=features, groups=time_data))

    def test_threshold_list_sorted_on_init(self) -> None:
        """Verify threshold list is sorted during initialization."""
        unsorted_thresholds = [
            pd.Timestamp("2021-06-01"),
            pd.Timestamp("2020-06-01"),
            pd.Timestamp("2022-06-01"),
        ]
        splitter = TimeThresholdSplitter(threshold_list=unsorted_thresholds)

        expected_sorted = [
            pd.Timestamp("2020-06-01"),
            pd.Timestamp("2021-06-01"),
            pd.Timestamp("2022-06-01"),
        ]

        self.assertEqual(splitter.threshold_list, expected_sorted)

    def test_convert_time_to_groups(self) -> None:
        """Test _convert_time_to_groups helper method."""
        threshold_list = [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01")]
        splitter = TimeThresholdSplitter(threshold_list=threshold_list)

        time_data = pd.Series(
            [
                pd.Timestamp("2020-01-01"),  # Group 0
                pd.Timestamp("2020-03-01"),  # Group 0
                pd.Timestamp("2020-09-01"),  # Group 1
                pd.Timestamp("2021-09-01"),  # Group 2
            ],
        )

        group_indices = splitter._convert_time_to_groups(time_data)  # pylint: disable=protected-access
        expected = np.array([0, 0, 1, 2])

        self.assertTrue(np.array_equal(group_indices, expected))

    def test_with_numpy_datetime64(self) -> None:
        """Test that numpy datetime64 arrays work as groups."""
        threshold_list = [pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01")]
        splitter = TimeThresholdSplitter(threshold_list=threshold_list)

        time_data = np.array(
            [
                "2020-01-01",  # Group 0
                "2020-09-01",  # Group 1
                "2021-09-01",  # Group 2
            ],
            dtype="datetime64[D]",
        )

        features = np.ones(len(time_data))
        splits = list(splitter.split(X=features, groups=time_data))

        expected = [
            (np.array([0]), np.array([1])),
            (np.array([0, 1]), np.array([2])),
        ]

        self._assert_splits_equal(splits, expected)

    def test_from_splits_per_year_basic(self) -> None:
        """Test constructing from per-year parameters with basic settings.

        In default, timestamps are rounded to midnight (00:00:00).

        """
        splitter = TimeThresholdSplitter(
            splits_per_year=2,
            last_year=2022,
            n_years=2,
        )

        # Should create 4 thresholds (2 per year * 2 years)
        self.assertEqual(len(splitter.threshold_list), 4)
        # By default, timestamps are rounded to midnight (00:00:00)
        expected_thresholds = [
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-07-02"),  # Mid-year, rounded to midnight
            pd.Timestamp("2022-01-01"),
            pd.Timestamp("2022-07-02"),  # Mid-year, rounded to midnight
        ]
        for expected, actual in zip(
            expected_thresholds,
            splitter.threshold_list,
            strict=False,
        ):
            self.assertEqual(expected, actual)

    def test_from_splits_per_year_with_n_skip_and_max_splits(self) -> None:
        """Test constructor with n_skip and max_splits parameters."""
        splitter = TimeThresholdSplitter(
            splits_per_year=4,
            last_year=2022,
            n_years=3,
            n_skip=2,
            max_splits=2,
        )

        self.assertEqual(splitter.n_skip, 2)
        self.assertEqual(splitter.max_splits, 2)
        self.assertEqual(len(splitter.threshold_list), 12)  # 4 * 3

    def test_from_splits_per_year_invalid_splits_too_low(self) -> None:
        """Ensure constructor raises when splits_per_year < 1."""
        with self.assertRaisesRegex(ValueError, "splits_per_year must be at least 1"):
            TimeThresholdSplitter(
                splits_per_year=0,
                last_year=2022,
                n_years=2,
            )

    def test_from_splits_per_year_round_to_day(self) -> None:
        """Test constructor with round_to='day' (default)."""
        splitter = TimeThresholdSplitter(
            splits_per_year=2,
            last_year=2022,
            n_years=1,
            round_to="day",
        )

        # All thresholds should be at midnight (00:00:00)
        for threshold in splitter.threshold_list:
            self.assertEqual(threshold.hour, 0)
            self.assertEqual(threshold.minute, 0)
            self.assertEqual(threshold.second, 0)
            self.assertEqual(threshold.microsecond, 0)

    def test_from_splits_per_year_round_to_none(self) -> None:
        """Test constructor with round_to=None."""
        splitter = TimeThresholdSplitter(
            splits_per_year=2,
            last_year=2022,
            n_years=1,
            round_to=None,
        )

        # At least one threshold should have non-zero time components
        # because 365.25 / 2 = 182.625 days
        has_non_zero_time = any(
            threshold.hour != 0
            or threshold.minute != 0
            or threshold.second != 0
            or threshold.microsecond != 0
            for threshold in splitter.threshold_list
        )
        self.assertTrue(has_non_zero_time)

    def test_from_splits_per_year_round_to_default(self) -> None:
        """Test constructor uses round_to='day' by default."""
        splitter = TimeThresholdSplitter(
            splits_per_year=4,
            last_year=2022,
            n_years=1,
        )

        # All thresholds should be at midnight (00:00:00) by default
        for threshold in splitter.threshold_list:
            self.assertEqual(threshold.hour, 0)
            self.assertEqual(threshold.minute, 0)
            self.assertEqual(threshold.second, 0)
            self.assertEqual(threshold.microsecond, 0)

    def test_from_splits_per_year_round_to_hour(self) -> None:
        """Test constructor with round_to='hour'."""
        splitter = TimeThresholdSplitter(
            splits_per_year=2,
            last_year=2022,
            n_years=1,
            round_to="hour",
        )

        # All thresholds should have zero minutes, seconds, and microseconds
        for threshold in splitter.threshold_list:
            self.assertEqual(threshold.minute, 0)
            self.assertEqual(threshold.second, 0)
            self.assertEqual(threshold.microsecond, 0)

    def test_from_splits_per_year_round_to_month(self) -> None:
        """Test constructor with round_to='month'."""
        splitter = TimeThresholdSplitter(
            splits_per_year=3,
            last_year=2022,
            n_years=1,
            round_to="month",
        )

        # All thresholds should be at the first day of the month at midnight
        for threshold in splitter.threshold_list:
            self.assertEqual(threshold.day, 1)
            self.assertEqual(threshold.hour, 0)
            self.assertEqual(threshold.minute, 0)
            self.assertEqual(threshold.second, 0)
            self.assertEqual(threshold.microsecond, 0)

    def test_from_splits_per_year_round_to_invalid(self) -> None:
        """Test constructor raises error for invalid round_to value."""
        with self.assertRaisesRegex(
            ValueError,
            "round_to must be 'day', 'month', 'hour', or None",
        ):
            TimeThresholdSplitter(
                splits_per_year=2,
                last_year=2022,
                n_years=1,
                round_to="year",  # type: ignore[arg-type]
            )

    def test_init_raises_when_neither_threshold_list_nor_last_year_provided(
        self,
    ) -> None:
        """Ensure __init__ raises when neither threshold_list nor last_year is set."""
        with self.assertRaisesRegex(
            ValueError,
            "Either 'threshold_list' must be provided or 'last_year' must be specified",
        ):
            TimeThresholdSplitter()

    def test_init_raises_when_both_threshold_list_and_last_year_provided(self) -> None:
        """Ensure __init__ raises when both threshold_list and last_year are set."""
        threshold_list = [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2021-01-01"),
        ]
        with self.assertRaisesRegex(
            ValueError,
            "Provide either 'threshold_list' or 'last_year', not both.",
        ):
            TimeThresholdSplitter(
                threshold_list=threshold_list,
                last_year=2022,
            )


if __name__ == "__main__":
    unittest.main()
