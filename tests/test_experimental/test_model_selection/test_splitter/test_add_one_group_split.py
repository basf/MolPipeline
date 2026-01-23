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
        with self.assertRaises(ValueError):
            next(splitter.split(X=features, y=None, groups=None))

    def test_get_n_splits_raises_without_groups(self) -> None:
        """Ensure get_n_splits raises when groups are missing."""
        splitter = AddOneGroupSplit()
        with self.assertRaises(ValueError):
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
        for (train_idx, test_idx), (exp_train, exp_test) in zip(
            splits,
            expected,
            strict=True,
        ):
            self.assertTrue(np.array_equal(train_idx, exp_train))
            self.assertTrue(np.array_equal(test_idx, exp_test))
        self.assertEqual(len(splits), 2)

    def test_respects_n_skip(self) -> None:
        """Verify initial splits can be skipped via n_skip."""
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        splitter = AddOneGroupSplit(n_skip=1)
        splits = list(splitter.split(X=np.ones_like(groups), groups=groups))
        expected = [
            (np.array([0, 1, 2, 3]), np.array([4, 5])),
            (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7])),
        ]
        for (train_idx, test_idx), (exp_train, exp_test) in zip(
            splits,
            expected,
            strict=True,
        ):
            self.assertTrue(np.array_equal(train_idx, exp_train))
            self.assertTrue(np.array_equal(test_idx, exp_test))
        self.assertEqual(len(splits), 2)

    def test_applies_max_splits_from_end(self) -> None:
        """Ensure max_splits limits the number of yielded splits."""
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        splitter = AddOneGroupSplit(max_splits=1)
        splits = list(splitter.split(X=np.ones_like(groups), groups=groups))
        self.assertEqual(len(splits), 1)
        train_idx, test_idx = splits[0]
        self.assertTrue(np.array_equal(train_idx, np.array([0, 1, 2, 3, 4, 5])))
        self.assertTrue(np.array_equal(test_idx, np.array([6, 7])))

    def test_get_n_splits_respects_limits(self) -> None:
        """Check get_n_splits accounts for n_skip and max_splits."""
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        base_splitter = AddOneGroupSplit(n_skip=1)
        self.assertEqual(
            base_splitter.get_n_splits(X=np.ones_like(groups), groups=groups),
            2,
        )
        limited_splitter = AddOneGroupSplit(n_skip=1, max_splits=1)
        self.assertEqual(
            limited_splitter.get_n_splits(
                X=np.ones_like(groups),
                groups=groups,
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
