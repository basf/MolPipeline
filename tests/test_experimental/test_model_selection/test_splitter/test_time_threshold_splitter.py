"""Unit tests for TimeThresholdSplitter."""

import unittest

import numpy as np
import numpy.typing as npt
import pandas as pd

from molpipeline.experimental.model_selection.splitter.time_threshold_splitter import (
    TimeThresholdSplitter,
)


class TestTimeThresholdSplitter(unittest.TestCase):
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

    def test_invalid_init_params(self) -> None:
        """Check that invalid combinations of init parameters raise errors."""
        # Empty threshold list
        with self.assertRaisesRegex(
            ValueError,
            "threshold_list must contain at least one",
        ):
            TimeThresholdSplitter(threshold_list=[])

        # Neither threshold_list nor final_threshold provided
        with self.assertRaisesRegex(
            ValueError,
            "Either 'threshold_list' must be provided or 'final_threshold' must be",
        ):
            TimeThresholdSplitter()

        # Both threshold_list and final_threshold provided
        with self.assertRaisesRegex(
            ValueError,
            "Provide either 'threshold_list' or 'final_threshold', not both.",
        ):
            TimeThresholdSplitter(
                threshold_list=[
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2021-01-01"),
                ],
                final_threshold=pd.Timestamp("2022-01-01"),
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

    def test_builds_thresholds_from_final_threshold_timestamp(self) -> None:
        """Construct thresholds using a concrete Timestamp final_threshold."""
        final_ts = pd.Timestamp("2022-12-31")
        splitter = TimeThresholdSplitter(
            final_threshold=final_ts,
            n_years=1,
            splits_per_year=2,
            round_to="day",
        )

        self.assertGreaterEqual(len(splitter.threshold_list), 2)
        self.assertEqual(splitter.threshold_list[-1], final_ts)
        # All thresholds should be at midnight (00:00:00)
        for threshold in splitter.threshold_list:
            self.assertEqual(threshold.hour, 0)
            self.assertEqual(threshold.minute, 0)
            self.assertEqual(threshold.second, 0)
            self.assertEqual(threshold.microsecond, 0)

    def test_resolve_final_threshold_special_strings(self) -> None:
        """Test that "now" and quarter strings are accepted as final_threshold."""
        splitter_now = TimeThresholdSplitter(final_threshold="now")
        self.assertIsInstance(splitter_now.threshold_list, list)

        for quarter in ["Q1", "Q2", "Q3", "Q4"]:
            splitter_quarter = TimeThresholdSplitter(final_threshold=quarter)  # type: ignore
            self.assertIsInstance(splitter_quarter.threshold_list, list)

    def test_round_threshold_validation(self) -> None:
        """Ensure invalid round_to raises a clear ValueError."""
        with self.assertRaisesRegex(
            ValueError,
            "round_to must be 'day', 'month', 'hour', or None",
        ):
            TimeThresholdSplitter(
                final_threshold=pd.Timestamp("2022-12-31"),
                splits_per_year=1,
                n_years=1,
                round_to="invalid",  # type: ignore[arg-type]
            )

    def test_final_threshold_now_uses_current_year(self) -> None:
        """Ensure final_threshold='now' uses the current year as reference.

        The constructed thresholds should all lie in a window that includes the
        current year and be rounded according to the default ``round_to='day'``.

        """
        now = pd.Timestamp.now()
        splitter = TimeThresholdSplitter(
            final_threshold="now",
            n_years=1,
            splits_per_year=2,
        )

        self.assertEqual(len(splitter.threshold_list), 2)
        self.assertEqual(splitter.threshold_list[-1].year, now.year)
        self.assertEqual(splitter.threshold_list[-1].month, now.month)
        self.assertEqual(splitter.threshold_list[-1].day, now.day)
        self.assertEqual(splitter.threshold_list[-1].hour, 0)
        self.assertEqual(splitter.threshold_list[-1].minute, 0)
        self.assertEqual(splitter.threshold_list[-1].second, 0)
        # All thresholds should be within one year around the current year
        years = {ts.year for ts in splitter.threshold_list}
        self.assertIn(now.year, years)
        # Default round_to='day' => thresholds at midnight
        for threshold in splitter.threshold_list:
            self.assertEqual(threshold.hour, 0)
            self.assertEqual(threshold.minute, 0)
            self.assertEqual(threshold.second, 0)
            self.assertEqual(threshold.microsecond, 0)

    def test_final_threshold_quarters_end_of_quarter_month(self) -> None:
        """Ensure quarter strings resolve to ends of their respective quarters.

        For 'Q1'..'Q4', the internally resolved final threshold should fall into
        the current year and produce at least one threshold. We also check that
        timestamps are day-rounded.

        """
        quarter_month_map = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}

        for quarter, month in quarter_month_map.items():
            splitter = TimeThresholdSplitter(
                final_threshold=quarter,  # type: ignore
                n_years=1,
                splits_per_year=1,
            )
            self.assertEqual(len(splitter.threshold_list), 1)
            self.assertEqual(splitter.threshold_list[-1].month, month)
            self.assertEqual(splitter.threshold_list[-1].day, 1)


if __name__ == "__main__":
    unittest.main()
