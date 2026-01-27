"""Unit tests for TimeThresholdSplitter."""

import unittest

import numpy as np
import pandas as pd

from molpipeline.experimental.model_selection.splitter.time_threshold_splitter import (
    TimeThresholdSplitter,
)


class TestTimeThresholdSplitter(unittest.TestCase):
    """Tests for TimeThresholdSplitter."""

    def _assert_splits_equal(
        self,
        actual_splits: list[tuple[np.ndarray, np.ndarray]],
        expected_splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Assert that actual splits match expected splits.

        Parameters
        ----------
        actual_splits : list[tuple[np.ndarray, np.ndarray]]
            The actual train/test split indices.
        expected_splits : list[tuple[np.ndarray, np.ndarray]]
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
        expected = [
            # Train: Groups 0,1, Test: Group 2
            (np.array([0, 1, 2, 3]), np.array([4, 5])),
            # Train: Groups 0,1,2, Test: Group 3
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

        # Only last split
        expected = [
            # Train: Groups 0,1,2, Test: Group 3
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
        expected_thresholds = [
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-07-02 15:00"),  # Approx. mid-year
            pd.Timestamp("2022-01-01"),
            pd.Timestamp("2022-07-02 15:00"),  # Approx. mid-year
        ]
        for expected, actual in zip(
            expected_thresholds,
            splitter.threshold_list,
            strict=False,
        ):
            self.assertEqual(expected, actual)

    def test_from_splits_per_year_with_n_skip_and_max_splits(self) -> None:
        """Test from_splits_per_year with n_skip and max_splits parameters."""
        splitter = TimeThresholdSplitter.from_splits_per_year(
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
        """Ensure from_splits_per_year raises when splits_per_year < 1."""
        with self.assertRaisesRegex(ValueError, "splits_per_year must be at least 1"):
            TimeThresholdSplitter.from_splits_per_year(
                splits_per_year=0,
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

        group_indices = splitter._convert_time_to_groups(time_data)  # noqa: SLF001  # pylint: disable=protected-access
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
        self.assertEqual(len(splits), 2)

    def test_empty_threshold_list(self) -> None:
        """Test that an empty threshold list raises ValueError."""
        with self.assertRaisesRegex(
            ValueError,
            "threshold_list must contain at least one",
        ):
            TimeThresholdSplitter(threshold_list=[])

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
        self.assertEqual(n_splits, 0)
        with self.assertRaisesRegex(ValueError, "Not enough groups to create "):
            list(splitter.split(X=features, groups=time_data))


if __name__ == "__main__":
    unittest.main()
