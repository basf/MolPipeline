"""Unit tests for TimeThresholdSplitter."""

import unittest

import numpy as np
import pandas as pd

from molpipeline.experimental.model_selection.splitter.time_threshold_splitter import (
    TimeThresholdSplitter,
)


class TestTimeThresholdSplitter(unittest.TestCase):
    """Tests for TimeThresholdSplitter."""

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

        # Expected:
        # Group 0 (<2020-06-01)
        # Group 1 (2020-06-01 to 2021-06-01)
        # Group 2 (>2021-06-01)
        # With n_skip=0, all groups serve as test sets sequentially
        expected = [
            (np.array([]), np.array([0, 1])),  # Train: nothing, Test: Group 0
            (np.array([0, 1]), np.array([2, 3])),  # Train: Group 0, Test: Group 1
            (
                np.array([0, 1, 2, 3]),
                np.array([4, 5]),
            ),  # Train: Groups 0,1, Test: Group 2
        ]

        self.assertEqual(len(splits), 3)
        for i in range(3):
            self.assertTrue(
                np.array_equal(splits[i][0], expected[i][0]),
                f"Train indices mismatch at split {i}",
            )
            self.assertTrue(
                np.array_equal(splits[i][1], expected[i][1]),
                f"Test indices mismatch at split {i}",
            )

    def test_n_skip(self) -> None:
        """Verify initial splits can be skipped via n_skip."""
        threshold_list = [
            pd.Timestamp("2020-06-01"),
            pd.Timestamp("2021-06-01"),
            pd.Timestamp("2022-06-01"),
        ]
        splitter = TimeThresholdSplitter(threshold_list=threshold_list, n_skip=1)

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

        # Skip group 1, start test from group 2
        expected = [
            (
                np.array([0, 1]),
                np.array([2, 3]),
            ),  # Train: Group 0, Test: Group 1
            (
                np.array([0, 1, 2, 3]),
                np.array([4, 5]),
            ),  # Train: Groups 0,1, Test: Group 2
            (
                np.array([0, 1, 2, 3, 4, 5]),
                np.array([6, 7]),
            ),  # Train: Groups 0,1,2, Test: Group 3
        ]

        self.assertEqual(len(splits), 3)
        for i in range(3):
            self.assertTrue(
                np.array_equal(splits[i][0], expected[i][0]),
                f"Train indices mismatch at split {i}",
            )
            self.assertTrue(
                np.array_equal(splits[i][1], expected[i][1]),
                f"Test indices mismatch at split {i}",
            )

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

        # Only last split
        expected = [
            (
                np.array([0, 1, 2, 3, 4, 5]),
                np.array([6, 7]),
            ),  # Train: Groups 0,1,2, Test: Group 3
        ]

        self.assertEqual(len(splits), 1)
        self.assertTrue(np.array_equal(splits[0][0], expected[0][0]))
        self.assertTrue(np.array_equal(splits[0][1], expected[0][1]))

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
        self.assertEqual(n_splits, 3)

    def test_get_n_splits_with_max_splits(self) -> None:
        """Check get_n_splits respects max_splits parameter."""
        threshold_list = [
            pd.Timestamp("2020-06-01"),
            pd.Timestamp("2021-06-01"),
            pd.Timestamp("2022-06-01"),
        ]
        splitter = TimeThresholdSplitter(threshold_list=threshold_list, max_splits=1)

        time_data = pd.Series(
            [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-09-01"),
                pd.Timestamp("2021-09-01"),
                pd.Timestamp("2022-09-01"),
            ],
        )

        n_splits = splitter.get_n_splits(X=np.ones(len(time_data)), groups=time_data)
        self.assertEqual(n_splits, 1)

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

    def test_from_splits_per_year_basic(self) -> None:
        """Test from_splits_per_year class method with basic parameters."""
        splitter = TimeThresholdSplitter.from_splits_per_year(
            splits_per_year=2,
            last_year=2022,
            n_years=2,
        )

        # Should create 4 thresholds (2 per year * 2 years)
        self.assertEqual(len(splitter.threshold_list), 4)

        # Verify thresholds are sorted
        for i in range(len(splitter.threshold_list) - 1):
            self.assertLess(splitter.threshold_list[i], splitter.threshold_list[i + 1])

    def test_from_splits_per_year_with_n_skip_and_max_splits(self) -> None:
        """Test from_splits_per_year with n_skip and max_splits parameters."""
        splitter = TimeThresholdSplitter.from_splits_per_year(
            splits_per_year=4,
            last_year=2022,
            n_years=3,
            n_skip=1,
            max_splits=2,
        )

        self.assertEqual(splitter.n_skip, 1)
        self.assertEqual(splitter.max_splits, 2)
        self.assertEqual(len(splitter.threshold_list), 12)  # 4 * 3

    def test_from_splits_per_year_invalid_splits_too_low(self) -> None:
        """Ensure from_splits_per_year raises when splits_per_year < 1."""
        with self.assertRaisesRegex(ValueError, "splits_per_year must be at least 1"):
            TimeThresholdSplitter.from_splits_per_year(
                splits_per_year=0,
                last_year=2022,
                n_years=2,
            )

    def test_from_splits_per_year_invalid_splits_too_high(self) -> None:
        """Ensure from_splits_per_year raises when splits_per_year > 12."""
        with self.assertRaisesRegex(ValueError, "splits_per_year must be at most 12"):
            TimeThresholdSplitter.from_splits_per_year(
                splits_per_year=13,
                last_year=2022,
                n_years=2,
            )

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

        group_indices = splitter._convert_time_to_groups(time_data)  # noqa: SLF001
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

        # Should create 3 splits (one per group)
        self.assertEqual(len(splits), 3)

    def test_empty_threshold_list(self) -> None:
        """Test behavior with empty threshold list."""
        splitter = TimeThresholdSplitter(threshold_list=[])

        time_data = pd.Series(
            [
                pd.Timestamp("2020-01-01"),  # Group 0
                pd.Timestamp("2021-01-01"),  # Group 0
            ],
        )

        features = np.ones(len(time_data))

        # All data should be in group 0, with n_skip=0, group 0 serves as test set
        n_splits = splitter.get_n_splits(X=features, groups=time_data)
        self.assertEqual(n_splits, 1)

    def test_single_threshold(self) -> None:
        """Test with a single threshold creating two groups."""
        threshold_list = [pd.Timestamp("2020-06-01")]
        splitter = TimeThresholdSplitter(threshold_list=threshold_list)

        time_data = pd.Series(
            [
                pd.Timestamp("2020-01-01"),  # Group 0
                pd.Timestamp("2020-03-01"),  # Group 0
                pd.Timestamp("2020-09-01"),  # Group 1
                pd.Timestamp("2021-01-01"),  # Group 1
            ],
        )

        features = np.ones(len(time_data))
        splits = list(splitter.split(X=features, groups=time_data))

        # Should create 2 splits: one for each group
        expected = [
            (np.array([]), np.array([0, 1])),  # Train: nothing, Test: Group 0
            (np.array([0, 1]), np.array([2, 3])),  # Train: Group 0, Test: Group 1
        ]

        self.assertEqual(len(splits), 2)
        self.assertTrue(np.array_equal(splits[0][0], expected[0][0]))
        self.assertTrue(np.array_equal(splits[0][1], expected[0][1]))
        self.assertTrue(np.array_equal(splits[1][0], expected[1][0]))
        self.assertTrue(np.array_equal(splits[1][1], expected[1][1]))

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

        # Only one group, with n_skip=0, group 0 serves as test set
        n_splits = splitter.get_n_splits(X=features, groups=time_data)
        self.assertEqual(n_splits, 1)


if __name__ == "__main__":
    unittest.main()
