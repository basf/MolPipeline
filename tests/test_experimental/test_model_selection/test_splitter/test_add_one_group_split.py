"""Unit tests for AddOneGroupSplit splitter."""

import unittest
from itertools import product

import numpy as np

from molpipeline.experimental.model_selection.splitter.group_addition_splitter import (
    GroupAdditionSplit,
)


class TestAddOneGroupSplit(unittest.TestCase):
    """Tests for AddOneGroupSplit."""

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
        splitter = GroupAdditionSplit()
        features = np.ones(3)
        with self.assertRaisesRegex(ValueError, "The groups parameter is required."):
            next(splitter.split(X=features, y=None, groups=None))

    def test_get_n_splits_raises_without_groups(self) -> None:
        """Ensure get_n_splits raises when groups are missing."""
        splitter = GroupAdditionSplit()
        with self.assertRaisesRegex(ValueError, "The groups parameter is required."):
            splitter.get_n_splits(X=np.ones(2), groups=None)

    def test_generates_expected_splits(self) -> None:
        """Verify splitter yields cumulative train groups by default."""
        groups = np.array([0, 0, 1, 1, 2, 2])
        splitter = GroupAdditionSplit()
        splits = list(splitter.split(X=np.ones_like(groups), groups=groups))
        expected = [
            (np.array([0, 1]), np.array([2, 3])),
            (np.array([0, 1, 2, 3]), np.array([4, 5])),
        ]

        self._assert_splits_equal(splits, expected)

        shuffled_groups = np.array([2, 0, 1, 0, 2, 1])
        shuffled_splits = list(
            splitter.split(X=np.ones_like(shuffled_groups), groups=shuffled_groups),
        )
        expected = [
            (np.array([1, 3]), np.array([2, 5])),
            (np.array([1, 2, 3, 5]), np.array([0, 4])),
        ]
        self._assert_splits_equal(shuffled_splits, expected)

    def test_n_skip(self) -> None:
        """Verify initial splits can be skipped via n_skip."""
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        splitter = GroupAdditionSplit(n_skip=2)  # Skip first two splits
        splits = list(splitter.split(X=np.ones_like(groups), groups=groups))
        expected = [
            (np.array([0, 1, 2, 3]), np.array([4, 5])),
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7])),
        ]

        self._assert_splits_equal(splits, expected)

    def test_applies_max_splits_from_end(self) -> None:
        """Ensure max_splits limits the number of yielded splits."""
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        splitter = GroupAdditionSplit(max_splits=1)
        splits = list(splitter.split(X=np.ones_like(groups), groups=groups))
        expected = [
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7])),
        ]

        self._assert_splits_equal(splits, expected)

        # Given the 4 groups and max_splits=1, n_skip<3 should yield the same split
        for n_skip in range(3):
            splitter = GroupAdditionSplit(max_splits=1, n_skip=n_skip)
            splits = list(splitter.split(X=np.ones_like(groups), groups=groups))
            self._assert_splits_equal(splits, expected)

    def test_no_test_data_raise_error(self) -> None:
        """Check get_n_splits accounts for n_skip and max_splits."""
        groups = np.array([0, 0, 1, 1])
        splitter = GroupAdditionSplit(n_skip=2)
        with self.assertRaisesRegex(ValueError, "Not enough groups to create"):
            list(splitter.split(X=np.ones_like(groups), groups=groups))

        splitter = GroupAdditionSplit(max_splits=1, n_skip=2)
        with self.assertRaisesRegex(ValueError, "Not enough groups to create"):
            list(splitter.split(X=np.ones_like(groups), groups=groups))

    def test_get_n_splits(self) -> None:
        """Verify get_n_splits returns the correct number of splits."""
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        unique_groups = np.unique(groups)

        max_split_list = [None, 1, 2, 3, 4]
        n_skip_list = [0, 1, 2, 3, 4]

        for max_split, n_skip in product(max_split_list, n_skip_list):
            splitter = GroupAdditionSplit(n_skip=n_skip, max_splits=max_split)
            n_splits = splitter.get_n_splits(X=np.ones_like(groups), groups=groups)

            if max_split is None:
                self.assertEqual(n_splits, len(unique_groups) - n_skip)
            else:
                self.assertLessEqual(n_splits, max_split)
                self.assertEqual(n_splits, min(len(unique_groups) - n_skip, max_split))


if __name__ == "__main__":
    unittest.main()
