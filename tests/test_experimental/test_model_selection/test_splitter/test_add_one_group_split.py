"""Unit tests for AddOneGroupSplit splitter."""

import unittest

import numpy as np

from molpipeline.experimental.model_selection.splitter.add_one_group_split import (
    AddOneGroupSplit,
)


class TestAddOneGroupSplit(unittest.TestCase):
    """Tests for AddOneGroupSplit."""

    def test_split_raises_without_groups(self) -> None:
        """Ensure split raises when groups are missing."""
        splitter = AddOneGroupSplit()
        features = np.ones(3)
        with self.assertRaisesRegex(ValueError, "The groups parameter is required."):
            next(splitter.split(X=features, y=None, groups=None))

    def test_get_n_splits_raises_without_groups(self) -> None:
        """Ensure get_n_splits raises when groups are missing."""
        splitter = AddOneGroupSplit()
        with self.assertRaisesRegex(ValueError, "The groups parameter is required."):
            splitter.get_n_splits(X=np.ones(2), groups=None)

    def test_generates_expected_splits(self) -> None:
        """Verify splitter yields cumulative train groups by default."""
        groups = np.array([0, 0, 1, 1, 2, 2])
        splitter = AddOneGroupSplit()
        splits = list(splitter.split(X=np.ones_like(groups), groups=groups))
        expected = [
            (np.array([0, 1]), np.array([2, 3])),
            (np.array([0, 1, 2, 3]), np.array([4, 5])),
        ]

        self.assertEqual(len(splits), 2)
        for i in range(2):
            self.assertTrue(np.array_equal(splits[i][0], expected[i][0]))
            self.assertTrue(np.array_equal(splits[i][1], expected[i][1]))

    def test_n_skip(self) -> None:
        """Verify initial splits can be skipped via n_skip."""
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        splitter = AddOneGroupSplit(n_skip=2)  # Skip first two splits
        splits = list(splitter.split(X=np.ones_like(groups), groups=groups))
        expected = [
            (np.array([0, 1, 2, 3]), np.array([4, 5])),
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7])),
        ]

        self.assertEqual(len(splits), 2)
        for i in range(2):
            self.assertTrue(np.array_equal(splits[i][0], expected[i][0]))
            self.assertTrue(np.array_equal(splits[i][1], expected[i][1]))

    def test_applies_max_splits_from_end(self) -> None:
        """Ensure max_splits limits the number of yielded splits."""
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        splitter = AddOneGroupSplit(max_splits=1)
        splits = list(splitter.split(X=np.ones_like(groups), groups=groups))
        expected = [
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7])),
        ]

        self.assertEqual(len(splits), 1)
        for i in range(1):
            self.assertTrue(np.array_equal(splits[i][0], expected[i][0]))
            self.assertTrue(np.array_equal(splits[i][1], expected[i][1]))

    def test_no_test_data_raise_error(self) -> None:
        """Check get_n_splits accounts for n_skip and max_splits."""
        groups = np.array([0, 0, 1, 1])
        splitter = AddOneGroupSplit(n_skip=2)
        with self.assertRaisesRegex(ValueError, "Not enough groups to create"):
            list(splitter.split(X=np.ones_like(groups), groups=groups))


if __name__ == "__main__":
    unittest.main()
