"""Unit tests for AddOneGroupSplit splitter."""

import unittest

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

    def test_no_test_data_raise_error(self) -> None:
        """Check get_n_splits accounts for n_skip and max_splits."""
        groups = np.array([0, 0, 1, 1])
        splitter = GroupAdditionSplit(n_skip=2)
        with self.assertRaisesRegex(ValueError, "Not enough groups to create"):
            list(splitter.split(X=np.ones_like(groups), groups=groups))


if __name__ == "__main__":
    unittest.main()
